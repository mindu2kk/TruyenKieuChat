"""Data-driven unit contracts for Kiều Bot's answer capability.

The JSONL corpus is also used by the standalone quality harness.  Keeping the
same cases in pytest makes answer regressions visible in the normal CI suite
with the failing case ID shown in the test name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.answer_harness import answer_completeness_issues, is_refusal_answer
from app.orchestrator import answer_with_router
from app.poem_tools import get_range
from app.router import parse_poem_request, route_query


ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "data" / "eval" / "chat_harness.jsonl"


def _load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


CASES = _load_cases()
ANSWER_CASES = [case for case in CASES if case.get("answer_contains")]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_answer_router_contract(case):
    decision = route_query(case["query"])

    assert decision.intent == case["intent"]
    assert decision.flow == case["flow"]


@pytest.mark.parametrize("case", ANSWER_CASES, ids=lambda case: case["id"])
def test_deterministic_answer_contract(case):
    result = answer_with_router(case["query"], long_answer=False)
    answer = result.get("answer", "")
    should_answer = bool(case.get("should_answer", True))

    assert answer.strip(), "Kiều Bot returned an empty answer"
    if should_answer:
        assert not is_refusal_answer(answer), "Kiều Bot refused an answerable question"
        assert not answer_completeness_issues(answer, case["query"])
    else:
        assert is_refusal_answer(answer), "Out-of-scope question was not refused"

    for fragment in case.get("answer_contains", []):
        assert fragment.casefold() in answer.casefold()

    if case.get("requires_exact_quote"):
        requested_range = parse_poem_request(case["query"])
        assert requested_range is not None
        request_type = requested_range[0]
        if request_type == "opening":
            start, end = 1, requested_range[1]
        elif request_type == "single":
            start = end = requested_range[1]
        else:
            start, end = requested_range[1:3]
        for line in get_range(start, end):
            assert line in answer


def test_answer_capability_dataset_has_required_coverage():
    """Prevent accidental shrinking of the answer-quality safety net."""

    intents = {case["intent"] for case in CASES}
    required = {
        "chitchat",
        "core_fact",
        "poem",
        "domain",
        "poem_analysis",
        "plot",
        "analysis",
        "facts",
        "out_of_scope",
    }

    assert len(CASES) >= 43
    assert len(ANSWER_CASES) >= 39
    assert required <= intents
