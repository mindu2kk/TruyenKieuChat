from unittest.mock import Mock, patch

import pytest

from app.generation import GenerationError, generate_answer_gemini


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

    with pytest.raises(GenerationError, match="GOOGLE_API_KEY"):
        generate_answer_gemini("prompt")
