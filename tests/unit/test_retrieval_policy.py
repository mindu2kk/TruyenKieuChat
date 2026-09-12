import pytest

from app.retrieval_policy import extract_line_range, select_retrieval_policy
from app.router import route_query


@pytest.mark.unit
@pytest.mark.parametrize(
    "query,lane",
    (
        ("Trích nguyên văn câu thơ nói về nỗi nhớ", "verse"),
        ("Quạt nồng ấp lạnh nghĩa là gì?", "glossary"),
        ("Phân tích tâm lý nhân vật Thúy Kiều", "character"),
        ("Điều gì xảy ra sau khi Kiều gặp Kim Trọng?", "timeline"),
        ("Bình giảng nghệ thuật đoạn Trao duyên", "literary-analysis"),
    ),
)
def test_select_retrieval_lane(query, lane):
    assert select_retrieval_policy(query).lane == lane


@pytest.mark.unit
def test_verse_lane_only_queries_primary_poem_type():
    policy = select_retrieval_policy("Tìm câu thơ về nỗi nhớ")

    assert policy.allowed_types == ("poem",)
    assert policy.prefer_poem is True


@pytest.mark.unit
def test_deictic_question_without_history_requests_clarification():
    decision = route_query("Phân tích nhân vật đó", has_history=False)

    assert decision.flow == "clarification"


@pytest.mark.unit
def test_deictic_question_with_history_can_continue_conversation():
    decision = route_query("Phân tích nhân vật đó", has_history=True)

    assert decision.flow != "clarification"


@pytest.mark.unit
def test_extract_line_range_for_timeline_query():
    assert extract_line_range("Sự kiện chính từ câu 151 đến 210 là gì?") == (151, 210)
    assert extract_line_range("Từ câu 210 đến 151") is None
