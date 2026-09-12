import pytest

from app.evaluation import (
    citation_precision,
    grounded_claim_rate,
    hit_relevance,
    ranking_metrics,
)


@pytest.mark.unit
def test_ranking_metrics_cover_recall_mrr_and_ndcg():
    metrics = ranking_metrics(([False, True], [True, False], [False, False]))

    assert metrics.recall_at_5 == pytest.approx(2 / 3)
    assert metrics.recall_at_10 == pytest.approx(2 / 3)
    assert metrics.mrr == pytest.approx(0.5)
    assert 0 < metrics.ndcg_at_5 < 1
    assert 0 < metrics.ndcg_at_10 < 1


def test_hit_relevance_supports_legacy_gold_context_ids():
    case = {"gold_ctx_ids": ["chunk-1795"]}
    hit = {"text": "Sen tàn", "meta": {"id": "chunk-1795"}}

    assert hit_relevance(case, hit) is True


@pytest.mark.unit
def test_hit_relevance_supports_line_overlap_and_content():
    line_case = {"gold_line_start": 20, "gold_line_end": 21}
    hit = {"text": "a", "meta": {"line_start": 19, "line_end": 20}}
    text_case = {"gold_contains": "quạt nồng"}

    assert hit_relevance(line_case, hit) is True
    assert hit_relevance(text_case, {"text": "Quạt nồng ấp lạnh", "meta": {}}) is True


@pytest.mark.unit
def test_grounding_and_citation_metrics_are_explicit():
    assert citation_precision(({"supported": True}, {"supported": False})) == 0.5
    assert grounded_claim_rate(({"evidence_ids": ["a"]}, {"evidence_ids": []})) == 0.5
