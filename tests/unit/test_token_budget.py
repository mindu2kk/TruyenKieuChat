import pytest

from app.router import route_query
from app.token_budget import plan_token_budget


@pytest.mark.parametrize(
    ("query", "expected"),
    (
        ("Xin chào", 320),
        ("Thúy Kiều là ai?", 600),
        ("Vì sao Thúy Kiều bán mình chuộc cha?", 800),
        ("Phân tích nghệ thuật trong Truyện Kiều", 1200),
    ),
)
def test_budget_follows_intent(query, expected):
    plan = plan_token_budget(query, route_query(query))
    assert plan.max_output_tokens == expected


def test_multi_part_request_gets_more_room_than_plain_facts():
    query = "Liệt kê đúng 8 ý về phẩm chất của Thúy Kiều"
    plan = plan_token_budget(query, route_query(query))
    assert plan.max_output_tokens >= 1300
    assert "requested-items:8" in plan.reasons


def test_deep_comparison_overrides_stale_small_ui_hint():
    query = "So sánh và phân tích sâu chữ hiếu với tình yêu trong quyết định bán mình"
    plan = plan_token_budget(query, route_query(query), requested_max_tokens=640)
    assert plan.max_output_tokens >= 1450
    assert plan.long_form is True


def test_requested_word_count_is_translated_to_token_capacity():
    query = "Viết bài văn khoảng 800 từ phân tích bi kịch Thúy Kiều"
    plan = plan_token_budget(query, route_query(query))
    assert plan.max_output_tokens >= 1900
    assert "requested-words:800" in plan.reasons


def test_exact_poem_lookup_needs_no_generation_budget():
    query = "Trích câu 3251-3254, không giải thích thêm"
    plan = plan_token_budget(query, route_query(query))
    assert plan.max_output_tokens == 0
    assert plan.tier == "deterministic"


@pytest.mark.parametrize(
    ("response_length", "expected_tokens", "expected_tier"),
    (
        ("super_short", 600, "super-short"),
        ("short", 800, "short"),
        ("long", 1200, "long"),
    ),
)
def test_user_response_length_has_exact_budget(response_length, expected_tokens, expected_tier):
    query = "Phân tích sâu tám luận điểm về Truyện Kiều"
    plan = plan_token_budget(query, route_query(query), response_length=response_length)

    assert plan.max_output_tokens == expected_tokens
    assert plan.tier == expected_tier
    assert plan.long_form is (response_length == "long")
