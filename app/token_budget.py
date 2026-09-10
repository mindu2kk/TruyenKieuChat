"""Deterministic output-token planning for each routed request."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .router import RouteDecision, normalize_query


@dataclass(frozen=True)
class TokenBudgetPlan:
    max_output_tokens: int
    tier: str
    long_form: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


_BASE_BUDGETS = {
    "chitchat": 320,
    "generic": 420,
    "domain": 600,
    "plot": 800,
    "facts": 720,
    "analysis": 1200,
    "poem_analysis": 1000,
}

_DETERMINISTIC_INTENTS = {"faq", "core_fact", "out_of_scope"}
_SHORT_SIGNALS = ("ngan gon", "tra loi ngan", "mot cau", "khong giai thich")
_DEEP_SIGNALS = ("chi tiet", "phan tich sau", "day du", "lap luan", "chung minh", "binh giang")
_ESSAY_SIGNALS = ("bai van", "bai nghi luan", "mo bai", "than bai", "ket bai", "hoc sinh gioi")


def _requested_count(query: str) -> Optional[int]:
    q = normalize_query(query)
    match = re.search(r"\b(?:dung|gom|liet ke|neu)\s+(\d{1,2})\s+(?:y|luan diem|dac diem|ly do)", q)
    return max(1, min(int(match.group(1)), 20)) if match else None


def _requested_words(query: str) -> Optional[int]:
    q = normalize_query(query)
    match = re.search(r"\b(?:khoang|toi thieu|toi da)?\s*(\d{2,4})\s+(?:tu|chu)\b", q)
    return max(30, min(int(match.group(1)), 1200)) if match else None


def plan_token_budget(
    query: str,
    decision: RouteDecision,
    *,
    long_answer: bool = False,
    requested_max_tokens: Optional[int] = None,
) -> TokenBudgetPlan:
    """Choose a generous but bounded generation budget from observable signals.

    ``requested_max_tokens`` is a hint, not a hard ceiling. A stale UI value
    must never truncate an essay or a multi-part analysis.
    """
    if decision.intent in _DETERMINISTIC_INTENTS or decision.flow == "exact-poem-lookup":
        return TokenBudgetPlan(0, "deterministic", False, ("no-generation-needed",))

    q = normalize_query(query)
    budget = _BASE_BUDGETS.get(decision.intent, 600)
    reasons = [f"intent:{decision.intent}"]

    count = _requested_count(query)
    if count:
        budget = max(budget, 260 + 130 * count)
        reasons.append(f"requested-items:{count}")

    words = _requested_words(query)
    if words:
        budget = max(budget, math.ceil(words * 1.8) + 160)
        reasons.append(f"requested-words:{words}")

    if any(term in q for term in ("so sanh", "doi chieu", "phan biet")):
        budget = max(budget, 1100)
        reasons.append("comparison")

    if any(term in q for term in _DEEP_SIGNALS):
        budget = max(budget, 1450)
        reasons.append("deep-analysis")

    if any(term in q for term in _ESSAY_SIGNALS):
        budget = max(budget, 1900)
        reasons.append("essay")

    if long_answer:
        budget = max(budget, 1600)
        reasons.append("user-long-mode")
        if requested_max_tokens is not None:
            try:
                budget = max(budget, min(int(requested_max_tokens), 2400))
                reasons.append("user-budget-hint")
            except (TypeError, ValueError):
                reasons.append("invalid-user-budget-ignored")

    # Short wording reduces ordinary answers, but never overrides an explicit
    # item/word count or a deep/essay request.
    if any(term in q for term in _SHORT_SIGNALS) and not (count or words or len(reasons) > 1):
        budget = min(budget, 480)
        reasons.append("concise-request")

    budget = max(256, min(int(budget), 2400))
    tier = "short" if budget <= 480 else "standard" if budget <= 900 else "long" if budget <= 1600 else "extended"
    return TokenBudgetPlan(budget, tier, budget >= 1000, tuple(reasons))


__all__ = ["TokenBudgetPlan", "plan_token_budget"]
