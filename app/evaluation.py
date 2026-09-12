"""Offline and live evaluation metrics for Kiều Bot."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "").casefold()
    return re.sub(r"\s+", " ", text).strip()


def recall_at_k(relevance: Sequence[bool], k: int) -> float:
    return 1.0 if any(relevance[: max(0, k)]) else 0.0


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    for index, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevance: Sequence[bool], k: int) -> float:
    values = [1.0 if value else 0.0 for value in relevance[: max(0, k)]]
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(values))
    ideal = sorted(values, reverse=True)
    idcg = sum(value / math.log2(index + 2) for index, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def hit_relevance(case: Mapping[str, object], hit: Mapping[str, object]) -> bool:
    meta = hit.get("meta") or hit.get("metadata") or {}
    if not isinstance(meta, Mapping):
        meta = {}
    gold_ids = {str(item) for item in case.get("gold_source_ids", []) or []}
    if gold_ids and str(meta.get("source_id") or "") in gold_ids:
        return True

    expected = normalize(str(case.get("gold_contains") or ""))
    if expected and expected in normalize(str(hit.get("text") or "")):
        return True

    gold_start = case.get("gold_line_start")
    gold_end = case.get("gold_line_end", gold_start)
    hit_start = meta.get("line_start")
    hit_end = meta.get("line_end")
    if all(isinstance(value, int) for value in (gold_start, gold_end, hit_start, hit_end)):
        return int(hit_start) <= int(gold_end) and int(hit_end) >= int(gold_start)
    return False


@dataclass(frozen=True)
class RankingMetrics:
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


def ranking_metrics(relevance_rows: Iterable[Sequence[bool]]) -> RankingMetrics:
    rows = list(relevance_rows)
    if not rows:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0)
    size = len(rows)
    return RankingMetrics(
        recall_at_5=sum(recall_at_k(row, 5) for row in rows) / size,
        recall_at_10=sum(recall_at_k(row, 10) for row in rows) / size,
        mrr=sum(reciprocal_rank(row) for row in rows) / size,
        ndcg_at_10=sum(ndcg_at_k(row, 10) for row in rows) / size,
    )


def citation_precision(citations: Sequence[Mapping[str, object]]) -> float:
    if not citations:
        return 0.0
    supported = sum(bool(item.get("supported")) for item in citations)
    return supported / len(citations)


def grounded_claim_rate(claims: Sequence[Mapping[str, object]]) -> float:
    if not claims:
        return 0.0
    grounded = sum(bool(item.get("evidence_ids")) for item in claims)
    return grounded / len(claims)


__all__ = [
    "RankingMetrics",
    "citation_precision",
    "grounded_claim_rate",
    "hit_relevance",
    "ndcg_at_k",
    "ranking_metrics",
    "recall_at_k",
    "reciprocal_rank",
]
