"""Run the Kiều Bot v1 quality benchmark.

Offline checks are deterministic and network-free. Add ``--live-retrieval``
after embedding the clean corpus to measure Recall@5/10, MRR and nDCG@10 on
MongoDB Atlas.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation import hit_relevance, ranking_metrics
from app.poem_tools import get_range, get_single
from app.router import parse_poem_request, route_query
from app.verifier import verify_and_autocorrect

DEFAULT_CASES = ROOT / "data" / "eval" / "quality_v1.jsonl"


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _offline_pass(case: dict) -> bool | None:
    judge = case["judge"]
    if judge == "exact_poem":
        parsed = parse_poem_request(case["query"])
        if not parsed:
            return False
        start = int(case["gold_line_start"])
        end = int(case["gold_line_end"])
        actual = get_single(start) if start == end else "\n".join(get_range(start, end))
        return actual == case["gold_text"]
    if judge == "quote_verifier":
        supplied = str(case["supplied_quote"])
        corrected, verification = verify_and_autocorrect(f'"{supplied}"', threshold=88.0, autocorrect=True)
        return str(case["gold_text"]) in corrected and bool(verification.get("accepted"))
    if judge == "route":
        decision = route_query(case["query"], has_history=False)
        return decision.flow == case["expected_flow"]
    return None


def run_offline(cases: list[dict]) -> dict:
    by_category: dict[str, list[bool]] = {}
    skipped = Counter()
    for case in cases:
        result = _offline_pass(case)
        if result is None:
            skipped[case["category"]] += 1
            continue
        by_category.setdefault(case["category"], []).append(result)
    rates = {category: sum(values) / len(values) for category, values in by_category.items()}
    evaluated = sum(len(values) for values in by_category.values())
    overall = statistics.mean([value for values in by_category.values() for value in values]) if evaluated else 0.0
    return {"evaluated": evaluated, "skipped": dict(skipped), "pass_rate": overall, "by_category": rates}


def _balanced_live_cases(cases: list[dict], limit: int) -> list[dict]:
    eligible = [
        case
        for case in cases
        if str(case.get("judge") or "").startswith("retrieval")
        if any(key in case for key in ("gold_line_start", "gold_contains", "gold_source_ids"))
    ]
    if limit <= 0 or limit >= len(eligible):
        return eligible
    by_category: dict[str, list[dict]] = {}
    for case in eligible:
        by_category.setdefault(case["category"], []).append(case)
    selected: list[dict] = []
    while len(selected) < limit:
        made_progress = False
        for category in sorted(by_category):
            bucket = by_category[category]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
                made_progress = True
        if not made_progress:
            break
    return selected


def run_live_retrieval(cases: list[dict], max_cases: int = 60) -> dict:
    from app.rag_pipeline import _get_hybrid_retriever
    from app.retrieval_policy import select_retrieval_policy

    rows = []
    rows_by_category: dict[str, list[list[bool]]] = {}
    failed_case_ids: list[str] = []
    evaluated = 0
    selected = _balanced_live_cases(cases, max_cases)
    for case in selected:
        policy = select_retrieval_policy(case["query"])
        hits = _get_hybrid_retriever().search(
            case["query"], top_k=10, filters=policy.mongo_filter(), num_candidates=200
        )
        hit_dicts = [
            {"text": hit.text, "meta": hit.metadata, "score": hit.score}
            for hit in hits
        ]
        relevance = [hit_relevance(case, hit) for hit in hit_dicts]
        rows.append(relevance)
        rows_by_category.setdefault(case["category"], []).append(relevance)
        if not any(relevance):
            failed_case_ids.append(str(case.get("id") or ""))
        evaluated += 1
    return {
        "evaluated": evaluated,
        "available": len(_balanced_live_cases(cases, 0)),
        "sample_limit": max_cases,
        "failed_case_ids": failed_case_ids,
        "by_category": {
            category: ranking_metrics(category_rows).__dict__
            for category, category_rows in rows_by_category.items()
        },
        **ranking_metrics(rows).__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--live-retrieval", action="store_true")
    parser.add_argument(
        "--max-live-cases",
        type=int,
        default=60,
        help="Balanced live sample size; use 0 to evaluate every eligible case.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    categories = Counter(case["category"] for case in cases)
    output = {"cases": len(cases), "categories": dict(categories), "offline": run_offline(cases)}
    if args.live_retrieval:
        output["retrieval"] = run_live_retrieval(cases, args.max_live_cases)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
