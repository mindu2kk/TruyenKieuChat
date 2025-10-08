# app/verifier.py
# -*- coding: utf-8 -*-
"""Verification helpers to ensure quoted câu thơ khớp với corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Utils (ép chuỗi an toàn)
# =========================
def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""


# =========================
# RapidFuzz / difflib bridges
# =========================
try:  # pragma: no cover - rapidfuzz là optional
    from rapidfuzz import fuzz, process  # type: ignore

    def _rf_extract_one(query: str, mapping: Dict[int, str]) -> Tuple[str, float, int] | None:
        """
        Cố gắng gọi RapidFuzz theo 2 cách:
        1) dict {label(int) -> text(str)}  => (label, score, text)
        2) list [text] + map index->label  => (text, score, index)
        Trả về (matched_text, score, line_no) hoặc None.
        """
        # làm sạch & chuẩn hóa mapping
        clean: Dict[int, str] = {}
        for k, v in mapping.items():
            try:
                key = int(k)
            except Exception:
                continue
            s = _as_str(v).strip()
            if s:
                clean[key] = s

        if not clean:
            return None

        q = _as_str(query)

        # Thử dạng dict trước (phù hợp RapidFuzz >=3.x)
        try:
            # RapidFuzz dict form trả (key, score, value)
            key, score, value = process.extractOne(q, clean, scorer=fuzz.token_set_ratio)
            return _as_str(value), float(score), int(key)
        except Exception:
            pass  # thử list form

        # Fallback list form (phiên bản khác/edge-case)
        texts = list(clean.values())
        line_nos = list(clean.keys())
        try:
            matched_text, score, idx = process.extractOne(q, texts, scorer=fuzz.token_set_ratio)
            return _as_str(matched_text), float(score), int(line_nos[int(idx)])
        except Exception:
            return None

    _extract_one = _rf_extract_one

except ImportError:  # pragma: no cover - fallback difflib
    import difflib

    def _simple_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0

    def _df_extract_one(query: str, mapping: Dict[int, str]) -> Tuple[str, float, int] | None:
        best: Tuple[str, float, int] | None = None
        q = _as_str(query)
        for line_no, text in mapping.items():
            t = _as_str(text).strip()
            if not t:
                continue
            score = _simple_ratio(q, t)
            cand = (t, float(score), int(line_no))
            if best is None or cand[1] > best[1]:
                best = cand
        return best

    _extract_one = _df_extract_one  # type: ignore


# =========================
# Poem data
# =========================
try:  # pragma: no cover - cho phép chạy như module hoặc script
    from .poem_tools import PoemLine, all_poem_lines
except ImportError:  # pragma: no cover
    from poem_tools import PoemLine, all_poem_lines  # type: ignore


QUOTE_PATTERN = re.compile(r'"([^"\n]{4,200})"')


@dataclass(frozen=True)
class QuoteCheck:
    quote: str
    score: float
    matched_line: Optional[int]
    matched_text: Optional[str]


def _best_match(quote: str, lines: List[PoemLine]) -> QuoteCheck:
    """
    Dùng mapping {line_no:int -> text:str} để tương thích mọi phiên bản RapidFuzz.
    """
    mapping: Dict[int, str] = {}
    for ln in lines:
        try:
            n = int(getattr(ln, "number", None))
        except Exception:
            continue
        t = _as_str(getattr(ln, "text", "")).strip()
        if not t:
            continue
        mapping[n] = t

    result = _extract_one(quote, mapping)
    if not result:
        return QuoteCheck(quote=_as_str(quote), score=0.0, matched_line=None, matched_text=None)

    matched_text, score, line_no = result
    return QuoteCheck(
        quote=_as_str(quote),
        score=float(score),
        matched_line=int(line_no) if isinstance(line_no, int) else None,
        matched_text=_as_str(matched_text),
    )


def verify_poem_quotes(answer: str, *, threshold: float = 74.0) -> Dict[str, object]:
    """
    Trả về:
      - quotes:  danh sách tất cả trích dẫn (kèm score & match)
      - accepted:những trích dẫn đạt ngưỡng
      - coverage:accepted / tổng số trích dẫn (0..1, làm tròn 3 chữ số)
    """
    answer = _as_str(answer).strip()
    lines = all_poem_lines()
    if not answer or not lines:
        return {"quotes": [], "accepted": [], "coverage": 0.0}

    found = QUOTE_PATTERN.findall(answer)
    if not found:
        return {"quotes": [], "accepted": [], "coverage": 0.0}

    checks: List[QuoteCheck] = [_best_match(q, lines) for q in found]
    accepted = [chk for chk in checks if (chk.score or 0.0) >= float(threshold)]
    coverage = len(accepted) / max(1, len(checks))

    return {
        "quotes": [chk.__dict__ for chk in checks],
        "accepted": [chk.__dict__ for chk in accepted],
        "coverage": round(coverage, 3),
    }


__all__ = ["verify_poem_quotes", "QuoteCheck"]
