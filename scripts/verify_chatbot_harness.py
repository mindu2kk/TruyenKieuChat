#!/usr/bin/env python
"""Run the deterministic chatbot contract suite without network services."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestrator import answer_with_router
from app.router import route_query


CASES_PATH = ROOT / "data" / "eval" / "chat_harness.jsonl"


def _load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def run() -> int:
    failures: list[str] = []
    cases = _load_cases()

    for case in cases:
        decision = route_query(case["query"])
        if decision.intent != case["intent"]:
            failures.append(f"{case['id']}: intent={decision.intent!r}, expected={case['intent']!r}")
        if decision.flow != case["flow"]:
            failures.append(f"{case['id']}: flow={decision.flow!r}, expected={case['flow']!r}")

        expected_fragments = case.get("answer_contains", [])
        if expected_fragments:
            result = answer_with_router(case["query"], long_answer=False)
            answer = result.get("answer", "")
            for fragment in expected_fragments:
                if fragment.casefold() not in answer.casefold():
                    failures.append(f"{case['id']}: answer missing {fragment!r}")

    passed = len(cases) - len({failure.split(":", 1)[0] for failure in failures})
    print(f"Chat harness: {passed}/{len(cases)} cases passed")
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
