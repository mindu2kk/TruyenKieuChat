from unittest.mock import Mock, patch

import pytest

from app.generation import (
    GenerationError,
    _setup_groq,
    generate_answer_gemini,
    generate_answer_groq,
)


@pytest.mark.unit
def test_gemini_25_flash_disables_thinking_for_short_rag_answers(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "0")
    client = Mock()
    client.models.generate_content.return_value.text = "Câu trả lời hoàn chỉnh."

    with patch("app.generation._setup", return_value=client):
        answer = generate_answer_gemini("prompt", model="gemini-2.5-flash", max_tokens=480)

    assert answer == "Câu trả lời hoàn chỉnh."
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens == 480
    assert config.thinking_config.thinking_budget == 0


@pytest.mark.unit
def test_generation_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(GenerationError, match="GOOGLE_API_KEY"):
        generate_answer_gemini("prompt")


@pytest.mark.unit
def test_gemini_failure_falls_back_to_groq_without_retry_delay(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    gemini_client = Mock()
    gemini_client.models.generate_content.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")

    with (
        patch("app.generation._setup", return_value=gemini_client),
        patch("app.generation.generate_answer_groq", return_value="Câu trả lời dự phòng.") as fallback,
        patch("app.generation.time.sleep") as sleep,
    ):
        answer = generate_answer_gemini("prompt", max_tokens=1200)

    assert answer == "Câu trả lời dự phòng."
    fallback.assert_called_once_with("prompt", long_answer=False, max_tokens=1200)
    sleep.assert_not_called()


@pytest.mark.unit
def test_missing_gemini_key_uses_groq(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")

    with patch("app.generation.generate_answer_groq", return_value="Groq hoạt động.") as fallback:
        answer = generate_answer_gemini("prompt")

    assert answer == "Groq hoạt động."
    fallback.assert_called_once()


@pytest.mark.unit
def test_groq_uses_configured_model_and_token_budget(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = Mock()
    client.chat.completions.create.return_value.choices = [Mock(message=Mock(content="Phản hồi từ Groq."))]

    with patch("app.generation._setup_groq", return_value=client):
        answer = generate_answer_groq("prompt", long_answer=True, max_tokens=1450)

    assert answer == "Phản hồi từ Groq."
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-oss-120b"
    assert kwargs["max_tokens"] == 1450
    assert "[PHONG CÁCH]" in kwargs["messages"][0]["content"]


@pytest.mark.unit
def test_exact_item_request_adds_machine_verifiable_contract(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    client = Mock()
    client.chat.completions.create.return_value.choices = [Mock(message=Mock(content="1. A\n2. B\n3. C"))]

    with patch("app.generation._setup_groq", return_value=client):
        answer = generate_answer_groq("Nêu đúng 3 ý về kết thúc Truyện Kiều.", max_tokens=800)

    assert answer == "1. A\n2. B\n3. C"
    sent_prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Trả lời đủ và đúng 3 ý" in sent_prompt
    assert "1., 2., 3." in sent_prompt


@pytest.mark.unit
def test_groq_client_has_bounded_timeout(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "18")

    with patch("app.generation._groq_client_for_key", return_value=Mock()) as build_client:
        _setup_groq()

    build_client.assert_called_once_with("groq-key", 18.0)
