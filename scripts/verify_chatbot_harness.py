#!/usr/bin/env python
"""Run the deterministic chatbot contract suite without network services."""

from __future__ import annotations

import json
import math
import sys
from time import perf_counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestrator import answer_with_router  # noqa: E402
from app.answer_harness import answer_completeness_issues, is_refusal_answer  # noqa: E402
from app.router import route_query  # noqa: E402


CASES_PATH = ROOT / "data" / "eval" / "chat_harness.jsonl"


def _load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def run() -> int:
    failures: list[str] = []
    cases = _load_cases()
    evaluated = 0
    answerable = 0
    answered = 0
    false_refusals = 0
    truncated = 0
    quote_errors = 0
    latencies_ms: list[float] = []

    for case in cases:
        decision = route_query(case["query"])
        if decision.intent != case["intent"]:
            failures.append(f"{case['id']}: intent={decision.intent!r}, expected={case['intent']!r}")
        if decision.flow != case["flow"]:
            failures.append(f"{case['id']}: flow={decision.flow!r}, expected={case['flow']!r}")

        expected_fragments = case.get("answer_contains", [])
        if expected_fragments:
            started = perf_counter()
            result = answer_with_router(case["query"], long_answer=False)
            latencies_ms.append((perf_counter() - started) * 1000)
            evaluated += 1
            answer = result.get("answer", "")
            should_answer = bool(case.get("should_answer", True))
            answerable += int(should_answer)
            refused = is_refusal_answer(answer)
            incomplete = answer_completeness_issues(answer, case["query"])
            if should_answer and answer.strip() and not refused and not incomplete:
                answered += 1
            if should_answer and refused:
                false_refusals += 1
                failures.append(f"{case['id']}: false refusal")
            if incomplete:
                truncated += 1
                failures.append(f"{case['id']}: incomplete answer {list(incomplete)!r}")
            quality = result.get("harness", {}).get("quality", {})
            if case.get("requires_exact_quote") and quality.get("quote_check") == "failed":
                quote_errors += 1
                failures.append(f"{case['id']}: exact quote verification failed")
            for fragment in expected_fragments:
                if fragment.casefold() not in answer.casefold():
                    failures.append(f"{case['id']}: answer missing {fragment!r}")

    passed = len(cases) - len({failure.split(":", 1)[0] for failure in failures})
    print(f"Chat harness: {passed}/{len(cases)} cases passed")
    answer_rate = answered / max(1, answerable)
    ordered = sorted(latencies_ms)
    p95_ms = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else 0.0
    print(
        "Quality metrics: "
        f"evaluated={evaluated}, answer_rate={answer_rate:.1%}, "
        f"false_refusal={false_refusals}, truncated={truncated}, "
        f"quote_errors={quote_errors}, deterministic_p95_ms={p95_ms:.0f}"
    )
    if answer_rate < 0.98:
        failures.append(f"quality-gate: answer_rate={answer_rate:.1%}, required>=98%")
    if false_refusals or truncated or quote_errors:
        failures.append("quality-gate: zero-defect response contract failed")
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
