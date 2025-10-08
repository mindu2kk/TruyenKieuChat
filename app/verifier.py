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

    def _extract_one(query: str, choices: Dict[int, str]) -> Tuple[str, float, int] | None:
        """
        RapidFuzz khi nhận dict sẽ trả về tuple (key, score, value)
        - key: chính là khóa (ở đây là line_no: int)
        - value: chuỗi text tương ứng
        """
        if not isinstance(choices, dict) or not choices:
            return None
        # Bảo đảm value là chuỗi
        safe_choices: Dict[int, str] = {}
        for k, v in choices.items():
            s = _as_str(v).strip()
            if s:
                safe_choices[int(k)] = s
        if not safe_choices:
            return None

        res = process.extractOne(_as_str(query), safe_choices, scorer=fuzz.token_set_ratio)
        if not res:
            return None

        key, score, value = res  # RapidFuzz: (key, score, value) cho dict
        try:
            line_no = int(key)
        except Exception:
            line_no = None  # hiếm gặp
        matched_text = _as_str(value)
        return matched_text, float(score), int(line_no) if line_no is not None else -1

except ImportError:  # pragma: no cover - fallback difflib
    import difflib

    def _simple_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0

    def _extract_one(query: str, choices: Dict[int, str]) -> Tuple[str, float, int] | None:
        if not isinstance(choices, dict) or not choices:
            return None
        best: Tuple[str, float, int] | None = None
        q = _as_str(query)
        for line_no, text in choices.items():
            s = _as_str(text)
            if not s:
                continue
            score = _simple_ratio(q, s)
            cand = (s, float(score), int(line_no))
            if best is None or cand[1] > best[1]:
                best = cand
        return best


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
    Dùng mapping {line_no:int -> text:str} cho RapidFuzz/difflib.
    LƯU Ý: Không dùng {text -> line_no} vì RapidFuzz sẽ coi value (line_no) là 'sequence' và gọi len().
    """
    choices: Dict[int, str] = {}
    for ln in lines:
        try:
            n = int(getattr(ln, "number", None))
        except Exception:
            continue
        t = _as_str(getattr(ln, "text", ""))
        if not t:
            continue
        choices[n] = t

    result = _extract_one(quote, choices)
    if not result:
        return QuoteCheck(quote=quote, score=0.0, matched_line=None, matched_text=None)

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
