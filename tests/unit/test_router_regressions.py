import pytest

from app.faq import _load_facts, lookup_faq
from app.router import route_query


@pytest.mark.unit
@pytest.mark.parametrize(
    "query",
    (
        "Thúy Vân là ai?",
        "Nhân vật Thúy Vân có vai trò gì?",
        "Giới thiệu Thuý Vân trong Truyện Kiều.",
    ),
)
def test_thuy_van_is_never_confused_with_poetic_rhyme(query):
    decision = route_query(query)

    assert decision.intent == "domain"
    assert decision.requires_poem_evidence is False
    assert decision.requires_exact_quotes is False


@pytest.mark.unit
def test_rhyme_analysis_requests_poem_evidence_without_forcing_a_quote():
    decision = route_query("Phân tích cách gieo vần và nhịp thơ trong đoạn Trao duyên")

    assert decision.intent == "poem_analysis"
    assert decision.requires_poem_evidence is True
    assert decision.requires_exact_quotes is False


@pytest.mark.unit
def test_explicit_quote_request_still_requires_exact_poem_text():
    decision = route_query("Trích nguyên văn câu thơ nói về chị em Thúy Kiều")

    assert decision.intent == "poem_analysis"
    assert decision.requires_exact_quotes is True


@pytest.mark.unit
def test_negated_explanation_stays_an_exact_lookup_only():
    decision = route_query("Trích câu 3251-3254, không giải thích thêm")

    assert decision.intent == "poem"
    assert decision.flow == "exact-poem-lookup"


@pytest.mark.unit
def test_committed_faq_store_is_loaded_and_accent_insensitive():
    _load_facts.cache_clear()

    hit = lookup_faq("THUY VAN LA AI?")

    assert hit is not None
    assert "em gái của Thúy Kiều" in hit["answer"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "query, expected",
    (
        ("Thúy Vân là ai?", "em gái"),
        ("Kim Trọng là ai?", "nho sinh"),
        ("Từ Hải là ai?", "anh hùng"),
        ("Hoạn Thư là ai?", "vợ cả"),
        ("Thúc Sinh là ai?", "vợ lẽ"),
        ("Mã Giám Sinh là ai?", "buôn người"),
        ("Sở Khanh là ai?", "lừa"),
        ("Tú Bà là ai?", "lầu xanh"),
        ("Giác Duyên là ai?", "ni cô"),
        ("Đạm Tiên là ai?", "ca nhi"),
    ),
)
def test_known_character_questions_always_have_direct_answers(query, expected):
    hit = lookup_faq(query)

    assert hit is not None
    assert expected in hit["answer"]
    assert "xác minh" not in hit["answer"]
    assert "corpus" not in hit["answer"]
