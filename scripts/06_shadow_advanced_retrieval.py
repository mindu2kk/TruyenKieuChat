"""A/B benchmark baseline retrieval against HyDE and CRAG-adaptive shadow paths."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.advanced_retrieval import (  # noqa: E402
    RetrievalGrade,
    generate_hypothetical_document,
    grade_retrieval,
    grade_retrieval_with_model,
    should_use_hyde,
)
from app.evaluation import hit_relevance, ranking_metrics  # noqa: E402
from app.generation import generate_answer_gemini, generate_answer_groq  # noqa: E402
from app.rag_pipeline import answer_question  # noqa: E402


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _load_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _available_corpus_labels(path: Path) -> tuple[set[str], set[str]]:
    context_ids: set[str] = set()
    source_ids: set[str] = set()
    if not path.is_dir():
        return context_ids, source_ids
    for chunk_path in path.glob("*.txt"):
        context_ids.add(chunk_path.stem)
        try:
            first_line = chunk_path.open("r", encoding="utf-8").readline().strip()
            if first_line.startswith("###META###"):
                meta = json.loads(first_line.removeprefix("###META###").strip())
                if meta.get("id"):
                    context_ids.add(str(meta["id"]))
                if meta.get("source_id"):
                    source_ids.add(str(meta["source_id"]))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return context_ids, source_ids


def _validate_dataset(cases: list[dict[str, Any]], corpus_path: Path) -> dict[str, Any]:
    if not cases:
        raise ValueError("Bộ eval rỗng; không thể chạy promotion gate.")
    context_ids, source_ids = _available_corpus_labels(corpus_path)
    unsupported: list[str] = []
    stale: list[str] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or index)
        gold_contexts = {str(value) for value in case.get("gold_ctx_ids", []) or []}
        gold_sources = {str(value) for value in case.get("gold_source_ids", []) or []}
        has_content_label = bool(case.get("gold_contains"))
        has_line_label = isinstance(case.get("gold_line_start"), int)
        if not (gold_contexts or gold_sources or has_content_label or has_line_label):
            unsupported.append(case_id)
            continue
        if gold_contexts and not (gold_contexts & context_ids):
            stale.append(case_id)
        if gold_sources and not (gold_sources & source_ids):
            stale.append(case_id)
    if unsupported or stale:
        messages = []
        if unsupported:
            messages.append(f"không có nhãn retrieval: {', '.join(unsupported)}")
        if stale:
            messages.append(f"nhãn không tồn tại trong corpus hiện tại: {', '.join(sorted(set(stale)))}")
        raise ValueError("Dataset không hợp lệ - " + "; ".join(messages))
    return {
        "validated": True,
        "corpus_path": str(corpus_path),
        "corpus_context_count": len(context_ids),
        "corpus_source_count": len(source_ids),
        "labeled_case_count": len(cases),
    }


def _first_relevant_rank(case: dict[str, Any], contexts: list[dict[str, Any]]) -> int | None:
    for index, hit in enumerate(contexts[:10], start=1):
        if hit_relevance(case, hit):
            return index
    return None


def _top_ids(contexts: list[dict[str, Any]]) -> list[str]:
    return [
        str((hit.get("meta") or hit.get("metadata") or {}).get("id") or "")
        for hit in contexts[:5]
    ]


def _retrieve(query: str, *, expansion: str | None = None) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    result = answer_question(
        query,
        k=10,
        num_candidates=140,
        synthesize=False,
        force_quote=False,
        query_expansions=[expansion] if expansion else None,
    )
    return result, (time.perf_counter() - started) * 1000.0


def _strategy_summary(rows: list[list[bool]], latencies: list[float]) -> dict[str, Any]:
    metrics = asdict(ranking_metrics(rows))
    return {
        **{key: round(value, 4) for key, value in metrics.items()},
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_p95_ms": round(_percentile(latencies, 0.95), 1),
    }


def _promotion_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    latency_limit_ms: float,
    *,
    evaluation_valid: bool,
) -> dict[str, Any]:
    recall_gain = candidate["recall_at_5"] - baseline["recall_at_5"]
    ndcg_gain = candidate["ndcg_at_5"] - baseline["ndcg_at_5"]
    latency_ok = candidate["latency_p95_ms"] <= latency_limit_ms
    eligible = evaluation_valid and recall_gain > 0 and ndcg_gain > 0 and latency_ok
    return {
        "promotion_eligible": eligible,
        "evaluation_valid": evaluation_valid,
        "recall_at_5_gain": round(recall_gain, 4),
        "ndcg_at_5_gain": round(ndcg_gain, 4),
        "latency_within_limit": latency_ok,
        "latency_limit_ms": latency_limit_ms,
        "rule": "recall_at_5_gain > 0 AND ndcg_at_5_gain > 0 AND p95 <= latency_limit_ms",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = _load_jsonl(Path(args.dataset), args.limit)
    validation = _validate_dataset(cases, Path(args.corpus))
    hyde_cache = _load_cache(Path(args.cache))
    crag_cache = _load_cache(Path(args.crag_cache))
    relevance = {"baseline": [], "hyde": [], "crag_adaptive": []}
    latency = {"baseline": [], "hyde": [], "crag_adaptive": []}
    details = []

    for index, case in enumerate(cases, start=1):
        query = str(case["query"])
        baseline, baseline_ms = _retrieve(query)
        baseline_contexts = list(baseline.get("contexts") or [])
        grade = grade_retrieval(query, baseline_contexts)
        crag_grader_ms = 0.0
        crag_grader_error = None
        crag_grade_source = "heuristic"
        if args.crag_grader == "groq":
            top_signature = "|".join(_top_ids(baseline_contexts))
            crag_cache_key = f"{args.crag_grader_model}:{query}:{top_signature}"
            cached_grade = None if args.refresh_crag_cache else crag_cache.get(crag_cache_key)
            if isinstance(cached_grade, dict) and cached_grade.get("label"):
                grade = RetrievalGrade(
                    str(cached_grade["label"]),
                    float(cached_grade.get("score") or 0.0),
                    float(cached_grade.get("query_coverage") or 0.0),
                    int(cached_grade.get("source_diversity") or 0),
                )
                crag_grader_ms = float(cached_grade.get("latency_ms") or 0.0)
                crag_grade_source = "groq-cache"
            else:
                try:
                    grade, crag_grader_ms = grade_retrieval_with_model(
                        query,
                        baseline_contexts,
                        generator=(
                            partial(generate_answer_groq, reasoning_effort="low")
                            if args.crag_grader_model.startswith("openai/gpt-oss-")
                            else generate_answer_groq
                        ),
                        model=args.crag_grader_model,
                    )
                    crag_cache[crag_cache_key] = {
                        **asdict(grade),
                        "latency_ms": round(crag_grader_ms, 1),
                    }
                    _save_cache(Path(args.crag_cache), crag_cache)
                    crag_grade_source = "groq-live"
                except Exception as exc:
                    crag_grader_error = f"{type(exc).__name__}: {exc}"
                    crag_grade_source = "heuristic-fallback"
        eligible_for_hyde = should_use_hyde(query)
        hypothetical = ""
        hyde_generation_ms = 0.0
        hyde_error = None

        if eligible_for_hyde:
            cache_key = f"{args.hyde_provider}:{args.model or ''}:{query}"
            cached = None if args.refresh_cache else hyde_cache.get(cache_key)
            if isinstance(cached, dict):
                hypothetical = str(cached.get("document") or "")
                hyde_generation_ms = float(cached.get("generation_ms") or 0.0)
            if not hypothetical:
                try:
                    generator = (
                        partial(generate_answer_groq, reasoning_effort="low")
                        if args.hyde_provider == "groq"
                        else generate_answer_gemini
                    )
                    hypothetical, hyde_generation_ms = generate_hypothetical_document(
                        query,
                        generator=generator,
                        model=args.model,
                    )
                    if hypothetical:
                        hyde_cache[cache_key] = {
                            "document": hypothetical,
                            "generation_ms": round(hyde_generation_ms, 1),
                            "provider": args.hyde_provider,
                            "model": args.model,
                        }
                        _save_cache(Path(args.cache), hyde_cache)
                except Exception as exc:  # shadow must never break the baseline benchmark
                    hyde_error = f"{type(exc).__name__}: {exc}"

        if hypothetical:
            hyde, hyde_retrieval_ms = _retrieve(query, expansion=hypothetical)
            hyde_contexts = list(hyde.get("contexts") or [])
            hyde_ms = hyde_generation_ms + hyde_retrieval_ms
        else:
            hyde_contexts = baseline_contexts
            hyde_ms = baseline_ms

        correction_used = grade.label != "correct" and bool(hypothetical)
        crag_contexts = hyde_contexts if correction_used else baseline_contexts
        crag_ms = baseline_ms + crag_grader_ms + (hyde_ms if correction_used else 0.0)

        for name, contexts, elapsed in (
            ("baseline", baseline_contexts, baseline_ms),
            ("hyde", hyde_contexts, hyde_ms),
            ("crag_adaptive", crag_contexts, crag_ms),
        ):
            relevance[name].append([hit_relevance(case, hit) for hit in contexts[:10]])
            latency[name].append(elapsed)

        details.append(
            {
                "index": index,
                "id": case.get("id"),
                "query": query,
                "crag_grade": asdict(grade),
                "crag_grade_source": crag_grade_source,
                "crag_grader_error": crag_grader_error,
                "crag_grader_latency_ms": round(crag_grader_ms, 1),
                "hyde_eligible": eligible_for_hyde,
                "hyde_generated": bool(hypothetical),
                "hyde_error": hyde_error,
                "crag_correction_used": correction_used,
                "latency_ms": {
                    "baseline": round(baseline_ms, 1),
                    "hyde": round(hyde_ms, 1),
                    "crag_adaptive": round(crag_ms, 1),
                },
                "first_relevant_rank": {
                    "baseline": _first_relevant_rank(case, baseline_contexts),
                    "hyde": _first_relevant_rank(case, hyde_contexts),
                    "crag_adaptive": _first_relevant_rank(case, crag_contexts),
                },
                "top_5_context_ids": {
                    "baseline": _top_ids(baseline_contexts),
                    "hyde": _top_ids(hyde_contexts),
                    "crag_adaptive": _top_ids(crag_contexts),
                },
            }
        )
        print(f"[{index}/{len(cases)}] {grade.label:9s} hyde={bool(hypothetical)!s:5s} {query}", flush=True)

    summaries = {name: _strategy_summary(relevance[name], latency[name]) for name in relevance}
    hyde_valid = all(not item["hyde_error"] for item in details) and all(
        (not item["hyde_eligible"]) or item["hyde_generated"] for item in details
    )
    crag_valid = all(not item["crag_grader_error"] for item in details)
    report = {
        "mode": "shadow",
        "production_activated": False,
        "dataset": str(args.dataset),
        "dataset_validation": validation,
        "hyde_provider": args.hyde_provider,
        "crag_grader": args.crag_grader,
        "crag_grader_model": args.crag_grader_model,
        "case_count": len(cases),
        "strategies": summaries,
        "gates": {
            "hyde": _promotion_gate(
                summaries["baseline"],
                summaries["hyde"],
                args.latency_limit_ms,
                evaluation_valid=hyde_valid,
            ),
            "crag_adaptive": _promotion_gate(
                summaries["baseline"],
                summaries["crag_adaptive"],
                args.latency_limit_ms,
                evaluation_valid=crag_valid,
            ),
        },
        "cases": details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/eval/advanced_retrieval.jsonl")
    parser.add_argument("--corpus", default="data/rag_chunks_clean")
    parser.add_argument("--output", default="output/shadow_advanced_retrieval.json")
    parser.add_argument("--cache", default=".cache/hyde_shadow.json")
    parser.add_argument("--crag-cache", default=".cache/crag_shadow.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--hyde-provider", choices=("auto", "groq"), default="auto")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--crag-grader", choices=("groq", "heuristic"), default="groq")
    parser.add_argument("--crag-grader-model", default="openai/gpt-oss-20b")
    parser.add_argument("--refresh-crag-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--latency-limit-ms", type=float, default=8000.0)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"strategies": report["strategies"], "gates": report["gates"]}, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
