"""Verification helpers to ensure quoted câu thơ khớp với corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover - rapidfuzz is optional for lightweight installs
    from rapidfuzz import fuzz, process  # type: ignore

    def _extract_one(query: str, choices: Dict[str, int]) -> Tuple[str, float, int] | None:
        result = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)
        if not result:
            return None
        matched_text, score, line_no = result
        return str(matched_text), float(score), int(line_no)

except ImportError:  # pragma: no cover - fallback to stdlib difflib
    import difflib

    def _simple_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0

    def _extract_one(query: str, choices: Dict[str, int]) -> Tuple[str, float, int] | None:
        best: Tuple[str, float, int] | None = None
        for text, line_no in choices.items():
            score = _simple_ratio(query, text)
            if best is None or score > best[1]:
                best = (text, score, line_no)
        return best

try:  # pragma: no cover - allow running as module or script
    from .poem_tools import PoemLine, all_poem_lines
except ImportError:  # pragma: no cover
    from poem_tools import PoemLine, all_poem_lines


QUOTE_PATTERN = re.compile(r'"([^"\n]{4,200})"')


@dataclass(frozen=True)
class QuoteCheck:
    quote: str
    score: float
    matched_line: Optional[int]
    matched_text: Optional[str]


def _best_match(quote: str, lines: List[PoemLine]) -> QuoteCheck:
    choices = {line.text: line.number for line in lines}
    result = _extract_one(quote, choices)
    if not result:
        return QuoteCheck(quote=quote, score=0.0, matched_line=None, matched_text=None)
    matched_text, score, line_no = result
    return QuoteCheck(
        quote=quote,
        score=float(score),
        matched_line=int(line_no),
        matched_text=str(matched_text),
    )


def verify_poem_quotes(answer: str, *, threshold: float = 74.0) -> Dict[str, object]:
    answer = (answer or "").strip()
    lines = all_poem_lines()
    if not answer or not lines:
        return {"quotes": [], "coverage": 0.0}

    found = QUOTE_PATTERN.findall(answer)
    if not found:
        return {"quotes": [], "coverage": 0.0}

    checks: List[QuoteCheck] = [_best_match(q, lines) for q in found]
    accepted = [chk for chk in checks if chk.score >= threshold]
    coverage = len(accepted) / max(1, len(checks))

    return {
        "quotes": [chk.__dict__ for chk in checks],
        "accepted": [chk.__dict__ for chk in accepted],
        "coverage": round(coverage, 3),
    }


__all__ = ["verify_poem_quotes", "QuoteCheck"]