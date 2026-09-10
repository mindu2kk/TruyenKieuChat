"""
Unit tests cho orchestrator logic.

Test các tính năng:
- Intent routing (FAQ, chitchat, poem, domain, generic, facts)
- Cache handling
- Error handling
- History processing
- Poem mode handling
- Domain RAG pipeline integration
"""

import pytest
from unittest.mock import ANY, Mock, patch, MagicMock
from app.orchestrator import answer_with_router, _make_cache_key, _history_to_text


@pytest.mark.unit
def test_make_cache_key():
    """Test tạo cache key với các tham số khác nhau."""
    key1 = _make_cache_key("test query", long_answer=False, intent="domain")
    key2 = _make_cache_key("test query", long_answer=True, intent="domain")
    key3 = _make_cache_key("test query", long_answer=False, intent="poem")

    assert key1 != key2  # Khác nhau do long_answer
    assert key1 != key3  # Khác nhau do intent
    assert "test query" in key1.lower()
    assert "intent=domain" in key1


@pytest.mark.unit
def test_history_to_text():
    """Test chuyển đổi history sang text format."""
    history = [
        ("user", "Câu hỏi đầu tiên"),
        ("assistant", "Câu trả lời đầu tiên"),
        ("user", "Câu hỏi thứ hai"),
    ]

    text = _history_to_text(history, max_turns=10)
    assert "[USER]" in text
    assert "[ASSISTANT]" in text
    assert "Câu hỏi đầu tiên" in text
    assert "Câu trả lời đầu tiên" in text


@pytest.mark.unit
def test_history_to_text_empty():
    """Test history rỗng."""
    text = _history_to_text(None, max_turns=10)
    assert text == ""

    text = _history_to_text([], max_turns=10)
    assert text == ""


@pytest.mark.unit
def test_history_to_text_max_turns():
    """Test giới hạn số lượng turns trong history."""
    history = []
    for i in range(10):
        history.append(("user", f"Câu hỏi {i}"))
        history.append(("assistant", f"Câu trả lời {i}"))

    text = _history_to_text(history, max_turns=4)
    # Chỉ lấy 4 turns cuối (8 messages)
    assert history[-1][1] in text
    assert history[0][1] not in text


@pytest.mark.unit
@patch("app.faq.lookup_faq")
def test_orchestrator_faq(mock_faq):
    """Test orchestrator trả về FAQ khi tìm thấy."""
    mock_faq.return_value = {"answer": "Đây là câu trả lời FAQ", "question": "Câu hỏi FAQ"}

    result = answer_with_router("Câu hỏi FAQ", k=5)

    assert result["intent"] == "faq"
    assert "Đây là câu trả lời FAQ" in result["answer"]
    mock_faq.assert_called_once_with("Câu hỏi FAQ")


@pytest.mark.unit
@patch("app.cache.get_cached")
@patch("app.faq.lookup_faq")
def test_orchestrator_cache(mock_faq, mock_cache):
    """Test orchestrator trả về cache khi có."""
    mock_faq.return_value = None
    mock_cache.return_value = "Câu trả lời từ cache"

    result = answer_with_router("test query", k=5)

    assert result["intent"] == "cache"
    assert result["answer"] == "Câu trả lời từ cache"
    mock_cache.assert_called()


@pytest.mark.unit
@patch("app.faq.lookup_faq")
@patch("app.cache.get_cached")
def test_orchestrator_chitchat(mock_cache, mock_faq):
    """Test orchestrator xử lý chitchat intent."""
    mock_faq.return_value = None
    mock_cache.return_value = None

    with patch("app.router.route_intent", return_value="chitchat"):
        with patch("app.orchestrator._safe_generate") as mock_gen:
            mock_gen.return_value = ("Xin chào! Tôi có thể giúp gì cho bạn?", None)

            result = answer_with_router("Bạn có thể giúp gì cho tôi?", k=5)

            assert result["intent"] == "chitchat"
            assert "Xin chào" in result["answer"]
            mock_gen.assert_called_once()


@pytest.mark.unit
@patch("app.router.route_intent")
@patch("app.faq.lookup_faq")
@patch("app.cache.get_cached")
@patch("app.poem_tools.poem_ready")
def test_orchestrator_poem_single(mock_poem_ready, mock_cache, mock_faq, mock_route):
    """Test orchestrator xử lý poem intent - single line."""
    mock_faq.return_value = None
    mock_cache.return_value = None
    mock_route.return_value = "poem"
    mock_poem_ready.return_value = True

    with patch("app.router.parse_poem_request") as mock_parse:
        mock_parse.return_value = ("single", 1)

        with patch("app.poem_tools.get_single") as mock_get:
            mock_get.return_value = "Trăm năm trong cõi người ta"

            result = answer_with_router("câu 1", k=5)

            assert result["intent"] == "poem"
            assert "Trăm năm" in result["answer"]
            assert "Câu 1" in result["answer"]
            mock_get.assert_called_once_with(1)


@pytest.mark.unit
@patch("app.router.route_intent")
@patch("app.faq.lookup_faq")
@patch("app.cache.get_cached")
@patch("app.poem_tools.poem_ready")
def test_orchestrator_poem_range(mock_poem_ready, mock_cache, mock_faq, mock_route):
    """Test orchestrator xử lý poem intent - range."""
    mock_faq.return_value = None
    mock_cache.return_value = None
    mock_route.return_value = "poem"
    mock_poem_ready.return_value = True

    with patch("app.router.parse_poem_request") as mock_parse:
        mock_parse.return_value = ("range", 1, 3)

        with patch("app.poem_tools.get_range") as mock_get:
            mock_get.return_value = [
                "Trăm năm trong cõi người ta",
                "Chữ tài chữ mệnh khéo là ghét nhau",
                "Trải qua một cuộc bể dâu",
            ]

            result = answer_with_router("câu 1-3", k=5)

            assert result["intent"] == "poem"
            assert "1–3" in result["answer"]
            assert "Trăm năm" in result["answer"]
            mock_get.assert_called_once_with(1, 3)


@pytest.mark.unit
@patch("app.router.route_intent")
@patch("app.faq.lookup_faq")
@patch("app.cache.get_cached")
@patch("app.poem_tools.poem_ready")
def test_orchestrator_never_verifies_incomplete_poem_range(mock_poem_ready, mock_cache, mock_faq, mock_route):
    mock_faq.return_value = None
    mock_cache.return_value = None
    mock_route.return_value = "poem"
    mock_poem_ready.return_value = True

    with patch("app.router.parse_poem_request", return_value=("range", 3251, 3254)):
        with patch("app.poem_tools.get_range", return_value=[]):
            result = answer_with_router("câu 3251-3254", k=5)

    assert result["harness"]["quality"]["status"] == "not-found"
    assert "0/4 câu" in result["answer"]
    assert "verified" not in result["harness"]["quality"]["status"]


@pytest.mark.unit
@patch("app.router.route_intent")
@patch("app.faq.lookup_faq")
@patch("app.cache.get_cached")
def test_orchestrator_domain_rag(mock_cache, mock_faq, mock_route):
    """Test orchestrator xử lý domain intent với RAG pipeline."""
    mock_faq.return_value = None
    mock_cache.return_value = None
    mock_route.return_value = "domain"

    with patch("app.rag_pipeline.answer_question") as mock_rag:
        mock_rag.return_value = {"answer": "Câu trả lời từ RAG", "sources": ["source1", "source2"], "evidence": []}

        with patch("app.verifier.verify_poem_quotes") as mock_verify:
            mock_verify.return_value = {"quotes": [], "accepted": []}

            result = answer_with_router("Thúy Kiều là ai?", k=5)

            assert result["intent"] == "domain"
            assert "Câu trả lời từ RAG" in result["answer"]
            assert result["harness"]["token_budget"]["max_output_tokens"] == 600
            mock_rag.assert_called_once()
            mock_verify.assert_called_once()


@pytest.mark.unit
@patch("app.orchestrator._safe_generate")
@patch("app.router.route_intent")
@patch("app.faq.lookup_faq")
@patch("app.cache.get_cached")
def test_orchestrator_generation_failure(mock_cache, mock_faq, mock_route, mock_gen):
    """Test orchestrator xử lý lỗi generation."""
    mock_faq.return_value = None
    mock_cache.return_value = None
    mock_route.return_value = "chitchat"
    mock_gen.return_value = (None, {"intent": "chitchat", "answer": "Lỗi generation", "error": "API error"})

    result = answer_with_router("Bạn có thể giúp gì cho tôi?", k=5)

    assert result["intent"] == "chitchat"
    assert "Lỗi" in result["answer"] or "error" in result.get("error", "").lower()


@pytest.mark.unit
@patch("app.router.route_intent")
@patch("app.faq.lookup_faq")
@patch("app.cache.get_cached")
def test_orchestrator_long_answer(mock_cache, mock_faq, mock_route):
    """Test orchestrator với long_answer=True."""
    mock_faq.return_value = None
    mock_cache.return_value = None
    mock_route.return_value = "domain"

    with patch("app.rag_pipeline.answer_question") as mock_rag:
        mock_rag.return_value = {"answer": "Câu trả lời dài", "sources": [], "evidence": []}

        with patch("app.verifier.verify_poem_quotes") as mock_verify:
            mock_verify.return_value = {"quotes": []}

            result = answer_with_router("Câu hỏi", k=5, long_answer=True)

            # Kiểm tra long_answer được truyền vào answer_question
            call_kwargs = mock_rag.call_args[1]
            assert call_kwargs["long_answer"] is True


@pytest.mark.unit
def test_orchestrator_infers_long_mode_and_budget_from_deep_analysis():
    pack = {"answer": "Phân tích hoàn chỉnh.", "sources": [], "evidence": [{"text": "x"}]}
    with (
        patch("app.faq.lookup_faq", return_value=None),
        patch("app.cache.get_cached", return_value=None),
        patch("app.rag_pipeline.answer_question", return_value=pack) as rag,
    ):
        result = answer_with_router(
            "So sánh và phân tích sâu chữ hiếu với tình yêu trong quyết định bán mình",
            long_answer=False,
            max_tokens=640,
        )

    assert rag.call_args.kwargs["long_answer"] is True
    assert rag.call_args.kwargs["max_tokens"] == 1450
    assert result["harness"]["token_budget"]["tier"] == "long"


@pytest.mark.unit
@patch("app.cache.set_cached")
@patch("app.router.route_intent")
@patch("app.faq.lookup_faq")
@patch("app.cache.get_cached")
def test_orchestrator_cache_setting(mock_cache, mock_faq, mock_route, mock_set_cache):
    """Test orchestrator lưu vào cache sau khi trả lời."""
    mock_faq.return_value = None
    mock_cache.return_value = None
    mock_route.return_value = "domain"

    with patch("app.rag_pipeline.answer_question") as mock_rag:
        mock_rag.return_value = {"answer": "Câu trả lời", "sources": [], "evidence": [{"text": "chứng cứ"}]}

        with patch("app.verifier.verify_poem_quotes") as mock_verify:
            mock_verify.return_value = {"quotes": []}

            answer_with_router("test query", k=5)

            # Kiểm tra set_cached được gọi
            mock_set_cache.assert_called()


@pytest.mark.unit
@patch("app.cache.set_cached")
@patch("app.router.route_intent", return_value="domain")
@patch("app.faq.lookup_faq", return_value=None)
@patch("app.cache.get_cached", return_value=None)
def test_orchestrator_retries_false_refusal_when_evidence_exists(mock_cache, mock_faq, mock_route, mock_set_cache):
    pack = {
        "answer": "Tôi chưa thể xác minh nhân vật này từ corpus.",
        "prompt": "RAG prompt with evidence",
        "sources": [],
        "evidence": [{"text": "Thúy Vân là em gái Thúy Kiều."}],
    }

    with (
        patch("app.rag_pipeline.answer_question", return_value=pack),
        patch("app.orchestrator._safe_generate", return_value=("Thúy Vân là em gái của Thúy Kiều.", None)) as generate,
    ):
        result = answer_with_router("Thúy Vân là ai?", k=5)

    assert result["answer"] == "Thúy Vân là em gái của Thúy Kiều."
    assert result["harness"]["quality"]["status"] == "verified"
    generate.assert_called_once()
    mock_set_cache.assert_called_once()


@pytest.mark.unit
@patch("app.cache.set_cached")
@patch("app.router.route_intent", return_value="domain")
@patch("app.faq.lookup_faq", return_value=None)
@patch("app.cache.get_cached", return_value=None)
def test_orchestrator_does_not_cache_answer_without_evidence(mock_cache, mock_faq, mock_route, mock_set_cache):
    pack = {"answer": "Không đủ bằng chứng từ corpus.", "sources": [], "evidence": []}

    with patch("app.rag_pipeline.answer_question", return_value=pack):
        result = answer_with_router("Một câu hỏi chưa có dữ liệu", k=5)

    assert result["harness"]["quality"]["status"] == "insufficient-evidence"
    mock_set_cache.assert_not_called()


@pytest.mark.unit
@patch("app.cache.set_cached")
@patch("app.router.route_intent", return_value="plot")
@patch("app.faq.lookup_faq", return_value=None)
@patch("app.cache.get_cached", return_value=None)
def test_orchestrator_repairs_incomplete_rag_answer_once(mock_cache, mock_faq, mock_route, mock_set_cache):
    truncated = "Thúy Kiều bán mình vì ba lý do:\n\n1."
    repaired = "1. Gia biến.\n2. Chữ hiếu.\n3. Sự hy sinh."
    pack = {
        "answer": truncated,
        "prompt": "RAG prompt",
        "sources": [],
        "evidence": [{"text": "evidence"}],
    }

    with (
        patch("app.rag_pipeline.answer_question", return_value=pack),
        patch("app.orchestrator._safe_generate", return_value=(repaired, None)) as generate,
    ):
        result = answer_with_router("Vì sao Kiều bán mình? Trả lời đúng 3 ý", k=5)

    assert result["answer"] == repaired
    assert result["harness"]["quality"]["status"] == "verified"
    generate.assert_called_once()
    mock_set_cache.assert_called_once_with(ANY, repaired)


@pytest.mark.unit
@patch("app.cache.set_cached")
@patch("app.router.route_intent", return_value="plot")
@patch("app.faq.lookup_faq", return_value=None)
@patch("app.cache.get_cached", return_value=None)
def test_orchestrator_does_not_cache_still_incomplete_answer(mock_cache, mock_faq, mock_route, mock_set_cache):
    truncated = "Thúy Kiều bán mình vì ba lý do:\n\n1."
    pack = {
        "answer": truncated,
        "prompt": "RAG prompt",
        "sources": [],
        "evidence": [{"text": "evidence"}],
    }

    with (
        patch("app.rag_pipeline.answer_question", return_value=pack),
        patch("app.orchestrator._safe_generate", return_value=(truncated, None)),
    ):
        result = answer_with_router("Vì sao Kiều bán mình? Trả lời đúng 3 ý", k=5)

    assert result["harness"]["quality"]["status"] == "incomplete"
    mock_set_cache.assert_not_called()
