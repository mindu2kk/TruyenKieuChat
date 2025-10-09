# app/verifier.py
# -*- coding: utf-8 -*-
"""Verification helpers to ensure quoted câu thơ khớp với corpus + tự sửa trích dẫn lệch nhỏ."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ------------------- Utilities (normalize/canonicalize) -------------------

def _strip_diacritics(s: str) -> str:
    if not s:
        return ""
    # chuẩn hoá NFC và bỏ dấu; quy 'đ' -> 'd'
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("Đ", "D").replace("đ", "d")
    return unicodedata.normalize("NFC", s)

def _canon(s: str) -> str:
    """Chuẩn hoá để so khớp: lower, bỏ dấu, bỏ ký tự lạ, gộp khoảng trắng."""
    s = (s or "").strip().lower()
    s = _strip_diacritics(s)
    # giữ chữ/số và khoảng trắng
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Hỗ trợ cả "..." và “...”
_QUOTE_ITER = re.compile(r'("([^"\n]{4,200})"|“([^”\n]{4,200})”)')

def _iter_quote_spans(text: str) -> List[Tuple[int, int, str]]:
    """Trả danh sách (start, end, inner_text) theo thứ tự xuất hiện."""
    spans: List[Tuple[int, int, str]] = []
    for m in _QUOTE_ITER.finditer(text or ""):
        start, end = m.span()
        inner = m.group(2) if m.group(2) is not None else (m.group(3) or "")
        spans.append((start, end, inner))
    return spans

def _find_quotes(text: str) -> List[str]:
    return [inner for _, _, inner in _iter_quote_spans(text)]

# --------------------- RapidFuzz / difflib backend ------------------------

try:  # pragma: no cover - rapidfuzz là tuỳ chọn
    from rapidfuzz import fuzz, process  # type: ignore

    def _extract_one(query: str, choices: Dict[str, Tuple[str,int]]) -> Tuple[str, float, int] | None:
        """
        choices: map canonical_text -> (original_text, line_no)
        """
        result = process.extractOne(_canon(query), list(choices.keys()), scorer=fuzz.token_set_ratio)
        if not result:
            return None
        canon_matched, score, _idx = result
        orig_text, line_no = choices[str(canon_matched)]
        return str(orig_text), float(score), int(line_no)

except ImportError:  # pragma: no cover - fallback stdlib
    import difflib

    def _simple_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0

    def _extract_one(query: str, choices: Dict[str, Tuple[str,int]]) -> Tuple[str, float, int] | None:
        best: Tuple[str, float, int] | None = None
        cq = _canon(query)
        for ccanon, (orig, ln) in choices.items():
            score = _simple_ratio(cq, ccanon)
            if best is None or score > best[1]:
                best = (orig, score, ln)
        return best

# --------------------------- Poem lines access -----------------------------

try:  # pragma: no cover
    from .poem_tools import PoemLine, all_poem_lines
except ImportError:  # pragma: no cover
    from poem_tools import PoemLine, all_poem_lines  # type: ignore


@dataclass(frozen=True)
class QuoteCheck:
    quote: str
    score: float
    matched_line: Optional[int]
    matched_text: Optional[str]
    exact: bool


# ------------------------------ API ---------------------------------------

def verify_poem_quotes(answer: str, *, threshold: float = 88.0) -> Dict[str, object]:
    """
    Trả về:
      - quotes: mọi trích dẫn đã kiểm
      - accepted: các trích ≥ threshold
      - non_exact: trích đạt ngưỡng nhưng khác nguyên văn (đề xuất sửa)
      - suggested_fixes: [(old, new), ...]
      - coverage: tỉ lệ đạt
    """
    answer = (answer or "").strip()
    lines = all_poem_lines()
    if not answer or not lines:
        return {"quotes": [], "accepted": [], "non_exact": [], "suggested_fixes": [], "coverage": 0.0}

    found = _find_quotes(answer)
    if not found:
        return {"quotes": [], "accepted": [], "non_exact": [], "suggested_fixes": [], "coverage": 0.0}

    # build canonical choices
    choices: Dict[str, Tuple[str,int]] = {}
    for line in lines:
        orig = getattr(line, "text", "")
        ln = int(getattr(line, "number", 0) or 0)
        c = _canon(orig)
        if c and c not in choices:
            choices[c] = (orig, ln)

    checks: List[QuoteCheck] = []
    non_exact: List[Dict[str, object]] = []
    fixes: List[Tuple[str, str]] = []

    for q in found:
        result = _extract_one(q, choices)
        if not result:
            checks.append(QuoteCheck(quote=q, score=0.0, matched_line=None, matched_text=None, exact=False))
            continue
        matched_text, score, line_no = result
        exact = _canon(q) == _canon(matched_text)
        checks.append(QuoteCheck(quote=q, score=float(score), matched_line=int(line_no), matched_text=str(matched_text), exact=exact))
        if score >= threshold and not exact:
            non_exact.append({
                "quote": q,
                "matched_text": matched_text,
                "matched_line": line_no,
                "score": float(score),
            })
            fixes.append((q, matched_text))

    accepted = [chk for chk in checks if chk.score >= threshold]
    coverage = len(accepted) / max(1, len(checks))

    return {
        "quotes": [chk.__dict__ for chk in checks],
        "accepted": [chk.__dict__ for chk in accepted],
        "non_exact": non_exact,
        "suggested_fixes": fixes,
        "coverage": round(coverage, 3),
    }


def apply_quote_corrections(answer: str, fixes: List[Tuple[str, str]]) -> str:
    """
    Thay thế từng trích dẫn theo thứ tự xuất hiện bằng nguyên văn matched_text.
    """
    if not fixes:
        return answer or ""

    spans = _iter_quote_spans(answer or "")
    if not spans:
        return answer or ""

    out = []
    last = 0
    fix_idx = 0

    for start, end, inner in spans:
        out.append(answer[last:start])  # phần trước dấu mở
        corrected = None
        if fix_idx < len(fixes):
            old, new = fixes[fix_idx]
            if _canon(inner) == _canon(old):
                corrected = new
                fix_idx += 1
        out.append('"' + (corrected if corrected is not None else inner) + '"')
        last = end

    out.append(answer[last:])
    return "".join(out)


def verify_and_autocorrect(answer: str, *, threshold: float = 88.0, autocorrect: bool = True) -> Tuple[str, Dict[str, object]]:
    """
    Trả về (answer_có_thể_đã_sửa, verification_payload).
    """
    ver = verify_poem_quotes(answer, threshold=threshold)
    if autocorrect:
        fixed = apply_quote_corrections(answer, ver.get("suggested_fixes", []))
        if fixed != (answer or ""):
            # tính lại số liệu sau khi sửa
            ver = verify_poem_quotes(fixed, threshold=threshold)
            return fixed, ver
    return answer or "", ver


__all__ = [
    "verify_poem_quotes",
    "verify_and_autocorrect",
    "apply_quote_corrections",
    "QuoteCheck",
]
