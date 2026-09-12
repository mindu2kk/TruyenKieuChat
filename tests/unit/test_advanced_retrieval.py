from unittest.mock import Mock

import pytest

from app.advanced_retrieval import (
    grade_retrieval,
    grade_retrieval_with_model,
    is_strategy_promoted,
    should_use_hyde,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Phân tích cảm thức thời gian trong Truyện Kiều", True),
        ("Vì sao hành động của Hoạn Thư có tính hai mặt?", True),
        ("Trích câu 1795-1796", False),
        ("Quạt nồng ấp lạnh nghĩa là gì?", False),
        ("Truyện Kiều có bao nhiêu câu?", False),
        ("Truyện Kiều thuộc thể thơ nào?", False),
    ],
)
def test_hyde_is_only_used_for_open_semantic_questions(query, expected):
    assert should_use_hyde(query) is expected


@pytest.mark.unit
def test_crag_grader_flags_empty_and_accepts_covered_context():
    assert grade_retrieval("phân tích Thúc Sinh", []).label == "incorrect"

    contexts = [
        {
            "text": "Phân tích nhân vật Thúc Sinh cho thấy tâm trạng nhớ thương và sự nhu nhược.",
            "score": 0.9,
            "meta": {"source_id": "scholar-a"},
        },
        {
            "text": "Thúc Sinh xuất hiện trong mạch truyện gia đình Hoạn Thư.",
            "score": 0.8,
            "meta": {"source_id": "poem"},
        },
    ]
    grade = grade_retrieval("phân tích tâm trạng Thúc Sinh", contexts)

    assert grade.label == "correct"
    assert grade.query_coverage >= 0.75
    assert grade.source_diversity == 2


@pytest.mark.unit
def test_model_crag_grader_parses_strict_label():
    generator = Mock(return_value="ambiguous")

    grade, elapsed_ms = grade_retrieval_with_model(
        "Phân tích Hoạn Thư",
        [{"text": "Hoạn Thư xuất hiện trong Truyện Kiều.", "meta": {"source_id": "a"}}],
        generator=generator,
        model="llama-3.1-8b-instant",
    )

    assert grade.label == "ambiguous"
    assert grade.score == 0.5
    assert elapsed_ms >= 0
    assert generator.call_args.kwargs["max_tokens"] == 512


@pytest.mark.unit
def test_only_strategy_that_passed_gate_is_promoted():
    assert is_strategy_promoted("hyde") is True
    assert is_strategy_promoted("crag_adaptive") is False
