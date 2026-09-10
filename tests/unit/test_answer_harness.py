from unittest.mock import Mock, patch
from pathlib import Path

import pytest

from app.answer_harness import (
    answer_completeness_issues,
    is_refusal_answer,
    verify_generated_answer,
    verified_core_answer,
)
from app.orchestrator import answer_with_router
from app.router import parse_poem_request, route_query


@pytest.mark.unit
@pytest.mark.parametrize(
    ("query", "intent", "flow"),
    [
        ("Xin chào Kiều Bot, giúp tôi học nhé", "chitchat", "smalltalk"),
        ("Hãy cho biết nhân vật chính của Truyện Kiều là ai?", "core_fact", "verified-core-fact"),
        ("Nhân vật trung tâm của Truyện Kiều là ai?", "core_fact", "verified-core-fact"),
        ("Hãy trích nguyên văn hai câu mở đầu Truyện Kiều", "poem", "exact-poem-lookup"),
        ("Hãy trích hai câu mở đầu và giải thích ý nghĩa", "poem", "grounded-poem-analysis"),
        ("thuy kieu gap kim trong o dau vay", "plot", "plot-and-character"),
        ("Phân tích nghệ thuật trong Truyện Kiều", "analysis", "literary-analysis"),
        ("Liệt kê 5 phẩm chất của Thúy Kiều", "facts", "structured-domain"),
        ("Tổng thống Hoa Kỳ năm 2026 là ai?", "out_of_scope", "safe-refusal"),
    ],
)
def test_route_query_contract(query, intent, flow):
    decision = route_query(query)
    assert decision.intent == intent
    assert decision.flow == flow


@pytest.mark.unit
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("trích hai câu mở đầu", ("opening", 2)),
        ("trích 10 câu đầu", ("opening", 10)),
        ("câu 241–260", ("range", 241, 260)),
        ("từ câu 1 đến câu 4", ("range", 1, 4)),
        ("cho câu số 1", ("single", 1)),
    ],
)
def test_poem_parser_understands_natural_vietnamese(query, expected):
    assert parse_poem_request(query) == expected


@pytest.mark.unit
def test_verified_core_answer_resolves_ambiguous_origin_wording():
    answer = verified_core_answer("Ai là tác giả của Truyện Kiều? Tên tác phẩm gốc bằng chữ Hán là gì?")
    assert answer is not None
    assert "Nguyễn Du" in answer
    assert "Đoạn trường tân thanh" in answer
    assert "Kim Vân Kiều truyện" in answer
    assert "chữ Nôm" in answer


@pytest.mark.unit
def test_clear_character_question_uses_curated_faq_before_rag():
    from app.faq import _load_facts

    _load_facts.cache_clear()
    result = answer_with_router("Thúy Vân là ai?")

    assert result["intent"] == "faq"
    assert "em gái của Thúy Kiều" in result["answer"]
    assert "xác minh nguyên văn" not in result["answer"]
    assert result["harness"]["quality"]["status"] == "verified"


@pytest.mark.unit
def test_orchestrator_core_fact_does_not_call_rag():
    with patch("app.rag_pipeline.answer_question") as rag:
        result = answer_with_router("Hãy cho biết nhân vật chính của Truyện Kiều là ai?")
    assert result["answer"].startswith("Thúy Kiều")
    assert result["harness"]["flow"] == "verified-core-fact"
    rag.assert_not_called()


@pytest.mark.unit
def test_orchestrator_out_of_scope_fails_closed_without_llm():
    with patch("app.orchestrator._safe_generate") as generate:
        result = answer_with_router("Tổng thống Hoa Kỳ năm 2026 là ai?")
    assert result["intent"] == "out_of_scope"
    assert "ngoài phạm vi Truyện Kiều" in result["answer"]
    generate.assert_not_called()


@pytest.mark.unit
def test_grounded_poem_analysis_keeps_exact_lines_before_explanation():
    query = "Hãy trích nguyên văn hai câu mở đầu Truyện Kiều và giải thích ngắn ý nghĩa."
    with (
        patch("app.cache.get_cached", return_value=None),
        patch("app.cache.set_cached"),
        patch("app.orchestrator._safe_generate") as generate,
    ):
        result = answer_with_router(query)
    assert "Trăm năm, trong cõi người ta," in result["answer"]
    assert "Chữ tài, chữ mệnh" in result["answer"]
    assert "Giải thích ngắn" in result["answer"]
    assert "tài mệnh tương đố" in result["answer"]
    assert result["harness"]["flow"] == "grounded-poem-analysis"
    generate.assert_not_called()


@pytest.mark.unit
@patch("app.verifier.all_poem_lines")
def test_quote_verifier_autocorrects_near_exact_poem(mock_lines):
    mock_lines.return_value = [Mock(text="Trăm năm, trong cõi người ta,", number=1)]
    answer, verification, quality = verify_generated_answer(
        'Nguyễn Du viết: "Tram nam trong coi nguoi ta".',
        require_exact_quotes=True,
        has_evidence=True,
    )
    assert "Trăm năm, trong cõi người ta," in answer
    assert verification["coverage"] == 1.0
    assert quality.quote_check == "passed"


@pytest.mark.unit
@patch("app.verifier.all_poem_lines")
def test_quote_verifier_blocks_unverifiable_poem(mock_lines):
    mock_lines.return_value = [Mock(text="Trăm năm, trong cõi người ta,", number=1)]
    answer, _verification, quality = verify_generated_answer(
        'Câu thơ "Một câu hoàn toàn không tồn tại trong Kiều" rất hay.',
        require_exact_quotes=True,
        has_evidence=True,
    )
    assert quality.status == "blocked-by-verifier"
    assert "không đưa ra câu thơ có thể sai" in answer


@pytest.mark.parametrize(
    ("answer", "query", "expected_issue"),
    [
        ("Thúy Kiều bán mình vì ba lý do:\n\n1.", "Trả lời đúng 3 ý", "dangling-list-marker"),
        ("1. Gia biến", "Liệt kê 3 ý", "requested-items-missing"),
        ("Câu trả lời gồm:", "Giải thích ngắn", "dangling-ending"),
    ],
)
def test_completeness_verifier_detects_truncated_answers(answer, query, expected_issue):
    assert expected_issue in answer_completeness_issues(answer, query)


def test_completeness_verifier_accepts_completed_requested_list():
    answer = "1. Gia biến.\n2. Chữ hiếu.\n3. Sự hy sinh."
    assert answer_completeness_issues(answer, "Trả lời đúng 3 ý") == ()


def test_completeness_verifier_accepts_markdown_numbered_list():
    answer = "**1. Gia biến.**\n**2. Chữ hiếu.**\n**3. Sự hy sinh.**"
    assert answer_completeness_issues(answer, "Trả lời đúng 3 ý") == ()


def test_incomplete_answer_is_not_marked_verified():
    answer, verification, quality = verify_generated_answer(
        "Thúy Kiều bán mình vì ba lý do:\n\n1.",
        require_exact_quotes=False,
        has_evidence=True,
        query="Trả lời đúng 3 ý",
    )

    assert quality.status == "incomplete"
    assert verification["completeness"]["status"] == "failed"


@pytest.mark.unit
@pytest.mark.parametrize(
    "answer",
    (
        "Tôi chưa thể xác minh thông tin này.",
        "Không đủ bằng chứng từ corpus.",
        "Dữ liệu không chứa câu trả lời.",
    ),
)
def test_refusal_answer_detection(answer):
    assert is_refusal_answer(answer)


@pytest.mark.unit
def test_normal_character_answer_is_not_a_refusal():
    assert not is_refusal_answer("Thúy Vân là em gái của Thúy Kiều.")


@pytest.mark.unit
def test_default_rag_answer_does_not_force_a_quote():
    pack = {"answer": "Kiều gặp Kim Trọng trong hội Đạp Thanh.", "sources": [], "evidence": [{"text": "x"}]}
    with (
        patch("app.faq.lookup_faq", return_value=None),
        patch("app.cache.get_cached", return_value=None),
        patch("app.rag_pipeline.answer_question", return_value=pack) as rag,
    ):
        answer_with_router("Thúy Kiều gặp Kim Trọng trong hoàn cảnh nào?")
    assert rag.call_args.kwargs["force_quote"] is False


@pytest.mark.unit
def test_intent_planner_ignores_legacy_large_token_budget_for_standard_question():
    pack = {"answer": "Câu trả lời.", "sources": [], "evidence": [{"text": "x"}]}
    with (
        patch("app.faq.lookup_faq", return_value=None),
        patch("app.cache.get_cached", return_value=None),
        patch("app.rag_pipeline.answer_question", return_value=pack) as rag,
    ):
        answer_with_router("Thúy Kiều là ai?", long_answer=False, max_tokens=8096)
    assert rag.call_args.kwargs["max_tokens"] == 600


@pytest.mark.unit
def test_chat_template_metadata_has_no_broken_alpine_scope():
    template = (Path(__file__).resolve().parents[2] / "chat_UI" / "templates" / "chat.html").read_text(encoding="utf-8")
    assert 'x-text="metaIntent"' not in template
    assert 'x-text="metaElapsed"' not in template
    assert "$root.settings.debug_meta" not in template
    assert "data-meta-quality" in template
    assert "data-meta-budget" in template
    assert '"🧠 legacy"' in template


@pytest.mark.unit
def test_chat_uses_compact_response_length_picker_instead_of_settings_panel():
    template = (Path(__file__).resolve().parents[2] / "chat_UI" / "templates" / "chat.html").read_text(encoding="utf-8")

    assert ">Thiết lập<" not in template
    assert 'aria-label="Mức độ suy nghĩ"' in template
    assert "Siêu ngắn · 600 tokens" in template
    assert "Ngắn · 1200 tokens" in template
    assert "Dài · 1700 tokens" in template
    assert '<input type="range"' not in template
    assert "overflow-x: hidden" in template
    assert "overflow-wrap: anywhere" in template


@pytest.mark.unit
def test_ui_uses_compiled_tailwind_instead_of_production_cdn():
    root = Path(__file__).resolve().parents[2]
    templates = [
        root / "chat_UI" / "templates" / "chat.html",
        root / "chat_UI" / "templates" / "base.html",
        root / "chat_UI" / "templates" / "account" / "login.html",
        root / "chat_UI" / "templates" / "account" / "signup.html",
    ]
    for template_path in templates:
        template = template_path.read_text(encoding="utf-8")
        assert "cdn.tailwindcss.com" not in template
        assert "{% static 'chat_UI/tailwind.css' %}" in template

    compiled_css = root / "chat_UI" / "static" / "chat_UI" / "tailwind.css"
    assert compiled_css.stat().st_size > 10_000
