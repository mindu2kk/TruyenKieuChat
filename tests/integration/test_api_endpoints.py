"""
Integration tests cho API endpoints.

Test các tính năng:
- Chat API endpoint
- History API endpoint
- Authentication
- Rate limiting
- Error handling
- Request/response format
"""

import pytest
import json
from django.test import Client
from django.contrib.auth.models import User
from django.core.management import call_command
from django.urls import reverse
from unittest.mock import patch, Mock

from chat_UI.models import UserProfile


@pytest.fixture
def client():
    """Django test client."""
    return Client()


@pytest.fixture
def test_user(db):
    """Tạo test user."""
    return User.objects.create_user(username="testuser", password="testpass123")


@pytest.fixture
def authenticated_client(client, test_user):
    """Client đã authenticated."""
    client.force_login(test_user)
    return client


@pytest.mark.integration
@pytest.mark.requires_db
def test_home_view_requires_login(client):
    """Test home view yêu cầu login."""
    response = client.get("/")

    # Nếu chưa login, redirect đến login page
    assert response.status_code in [302, 401]


@pytest.mark.integration
@pytest.mark.requires_db
def test_signup_creates_persistent_user_and_profile(client, db):
    response = client.post(
        reverse("account_signup"),
        data={
            "username": "new_kieu_reader",
            "email": "new-kieu-reader@example.com",
            "password1": "VerySafePass123!",
            "password2": "VerySafePass123!",
        },
    )

    assert response.status_code == 302
    user = User.objects.get(username="new_kieu_reader")
    assert user.email == "new-kieu-reader@example.com"
    assert user.check_password("VerySafePass123!")
    assert UserProfile.objects.filter(user=user).exists()


@pytest.mark.integration
@pytest.mark.requires_db
def test_login_accepts_registered_email(client, db):
    user = User.objects.create_user(
        username="email_login_reader",
        email="email-login@example.com",
        password="VerySafePass123!",
    )

    response = client.post(
        reverse("account_login"),
        data={"login": user.email, "password": "VerySafePass123!"},
    )

    assert response.status_code == 302
    assert response.url == "/"


def test_bootstrap_superuser_creates_administrator(monkeypatch, db):
    monkeypatch.setenv("DJANGO_SUPERUSER_USERNAME", "kieu_admin_test")
    monkeypatch.setenv("DJANGO_SUPERUSER_EMAIL", "kieu-admin-test@example.com")
    monkeypatch.setenv("DJANGO_SUPERUSER_PASSWORD", "VerySafePass123!")

    call_command("bootstrap_superuser")

    user = User.objects.get(username="kieu_admin_test")
    assert user.is_staff
    assert user.is_superuser
    assert user.check_password("VerySafePass123!")


@pytest.mark.integration
@pytest.mark.requires_db
def test_home_view_authenticated(authenticated_client):
    """Test home view khi đã authenticated."""
    response = authenticated_client.get("/")

    assert response.status_code == 200
    assert "gemini_ok" in response.context or "gemini_ok" in str(response.content)
    assert "poem_ok" in response.context or "poem_ok" in str(response.content)


@pytest.mark.integration
@pytest.mark.requires_db
def test_chat_api_get(authenticated_client):
    """Test chat API GET request."""
    response = authenticated_client.get("/api/chat")

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data.get("ok") is True


@pytest.mark.integration
@pytest.mark.requires_db
@patch("chat_UI.views.answer_with_router")
@patch("chat_UI.views.count_user_messages_today")
@patch("chat_UI.views.save_message_to_mongo")
@patch("chat_UI.views.get_history_for_bot", return_value=[])
def test_chat_api_post_success(mock_history, mock_save, mock_count, mock_router, authenticated_client):
    """Test chat API POST request thành công."""
    mock_count.return_value = 0  # Chưa dùng quota
    mock_router.return_value = {"intent": "domain", "answer": "Câu trả lời test", "sources": []}

    payload = {
        "message": "Thúy Kiều là ai?",
        "k": 5,
        "model": "gemini-2.0-flash",
        "response_length": "long",
    }

    response = authenticated_client.post("/api/chat", data=json.dumps(payload), content_type="application/json")

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data.get("ok") is True
    assert "answer" in data
    assert data["answer"] == "Câu trả lời test"
    assert "intent" in data
    mock_router.assert_called_once()
    assert mock_router.call_args.kwargs["response_length"] == "long"
    assert mock_router.call_args.kwargs["max_tokens"] == 1700
    assert mock_router.call_args.kwargs["long_answer"] is True
    mock_save.assert_called()


@pytest.mark.integration
@pytest.mark.requires_db
@patch("chat_UI.views.count_user_messages_today")
def test_chat_api_quota_exceeded(mock_count, authenticated_client):
    """Test chat API khi vượt quota."""
    mock_count.return_value = 20  # Đã dùng hết quota

    payload = {"message": "Test query", "k": 5}

    response = authenticated_client.post("/api/chat", data=json.dumps(payload), content_type="application/json")

    assert response.status_code == 429
    data = json.loads(response.content)
    assert data.get("ok") is False
    assert "quota" in data.get("error", "")
    assert "20 câu hỏi" in data.get("message", "")


@pytest.mark.integration
@pytest.mark.requires_db
def test_chat_api_invalid_json(authenticated_client):
    """Test chat API với invalid JSON."""
    response = authenticated_client.post("/api/chat/", data="invalid json", content_type="application/json")

    assert response.status_code == 400
    data = json.loads(response.content)
    assert data.get("ok") is False
    assert "error" in data


@pytest.mark.integration
@pytest.mark.requires_db
def test_chat_api_empty_message(authenticated_client):
    """Test chat API với empty message."""
    payload = {"message": "", "k": 5}

    response = authenticated_client.post("/api/chat", data=json.dumps(payload), content_type="application/json")

    assert response.status_code == 400
    data = json.loads(response.content)
    assert data.get("ok") is False
    assert "error" in data


@pytest.mark.integration
@pytest.mark.requires_db
@patch("chat_UI.views.answer_with_router")
@patch("chat_UI.views.count_user_messages_today")
@patch("chat_UI.views.save_message_to_mongo")
@patch("chat_UI.views.get_history_for_bot", return_value=[])
def test_chat_api_error_handling(mock_history, mock_save, mock_count, mock_router, authenticated_client):
    """Test chat API xử lý lỗi từ orchestrator."""
    mock_count.return_value = 0
    mock_router.side_effect = Exception("Backend error")

    payload = {"message": "Test query", "k": 5}

    response = authenticated_client.post("/api/chat", data=json.dumps(payload), content_type="application/json")

    assert response.status_code == 500
    data = json.loads(response.content)
    assert data.get("ok") is False
    assert "error" in data
    assert "backend" in data.get("error", "")


@pytest.mark.integration
@pytest.mark.requires_db
@patch("chat_UI.views.get_history_for_api")
def test_history_api_get(mock_get_history, authenticated_client):
    """Test history API GET request."""
    mock_get_history.return_value = [
        {"role": "user", "content": "Câu hỏi 1"},
        {"role": "assistant", "content": "Câu trả lời 1"},
    ]

    response = authenticated_client.get("/api/history/")

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data.get("ok") is True
    assert "messages" in data
    assert len(data["messages"]) == 2


@pytest.mark.integration
@pytest.mark.requires_db
@patch("chat_UI.views.clear_user_history")
def test_history_api_clear(mock_clear, authenticated_client):
    """Test history API clear history."""
    mock_clear.return_value = None

    response = authenticated_client.delete("/api/history/")

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data.get("ok") is True
    assert "messages" in data
    assert len(data["messages"]) == 0
    mock_clear.assert_called_once()


@pytest.mark.integration
@patch("chat_UI.views.connection.cursor")
@patch("chat_UI.views.get_mongo_client")
@patch("chat_UI.views.poem_ready", return_value=True)
@patch("chat_UI.views.is_generation_configured", return_value=True)
@patch("chat_UI.views.is_groq_configured", return_value=True)
@patch("chat_UI.views.is_gemini_configured", return_value=True)
def test_health_api_ready(mock_gemini, mock_groq, mock_generation, mock_poem, mock_client, mock_cursor, client):
    mock_cursor.return_value.__enter__.return_value.fetchone.return_value = (1,)
    mock_client.return_value.admin.command.return_value = {"ok": 1}

    response = client.get("/api/health/")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "database": True,
        "database_status": "ready",
        "mongo": True,
        "mongo_status": "ready",
        "gemini_configured": True,
        "groq_configured": True,
        "generation_configured": True,
        "poem_ready": True,
    }
    mock_cursor.return_value.__enter__.return_value.execute.assert_called_once_with("SELECT 1")


@pytest.mark.integration
@patch("chat_UI.views.connection.cursor")
@patch("chat_UI.views.get_mongo_client")
@patch("chat_UI.views.poem_ready", return_value=True)
@patch("chat_UI.views.is_generation_configured", return_value=True)
@patch("chat_UI.views.is_groq_configured", return_value=True)
@patch("chat_UI.views.is_gemini_configured", return_value=True)
def test_health_api_reuses_short_lived_success(
    mock_gemini, mock_groq, mock_generation, mock_poem, mock_client, mock_cursor, client, monkeypatch, settings
):
    import chat_UI.views as views

    settings.HEALTH_CHECK_CACHE_SECONDS = 10
    monkeypatch.setattr(views, "_health_cache", None)
    mock_cursor.return_value.__enter__.return_value.fetchone.return_value = (1,)
    mock_client.return_value.admin.command.return_value = {"ok": 1}

    assert client.get("/api/health/").status_code == 200
    assert client.get("/api/health/").status_code == 200

    assert mock_cursor.call_count == 1
    assert mock_client.return_value.admin.command.call_count == 1


@pytest.mark.integration
@patch("chat_UI.views.connection.cursor", side_effect=Exception("sensitive database details"))
@patch("chat_UI.views.get_mongo_client")
@patch("chat_UI.views.poem_ready", return_value=True)
@patch("chat_UI.views.is_generation_configured", return_value=True)
@patch("chat_UI.views.is_groq_configured", return_value=True)
@patch("chat_UI.views.is_gemini_configured", return_value=True)
def test_health_api_reports_sql_database_failure(
    mock_gemini, mock_groq, mock_generation, mock_poem, mock_client, mock_cursor, client
):
    mock_client.return_value.admin.command.return_value = {"ok": 1}

    response = client.get("/api/health/")

    assert response.status_code == 503
    assert response.json() == {
        "ok": False,
        "database": False,
        "database_status": "query_failed",
        "mongo": True,
        "mongo_status": "ready",
        "gemini_configured": True,
        "groq_configured": True,
        "generation_configured": True,
        "poem_ready": True,
    }
    assert "sensitive database details" not in response.content.decode("utf-8")


@pytest.mark.integration
@patch("chat_UI.views.connection.cursor")
@patch("chat_UI.views.get_mongo_client", side_effect=ValueError("MONGO_URI missing"))
@patch("chat_UI.views.poem_ready", return_value=True)
@patch("chat_UI.views.is_generation_configured", return_value=True)
@patch("chat_UI.views.is_groq_configured", return_value=True)
@patch("chat_UI.views.is_gemini_configured", return_value=True)
def test_health_api_reports_missing_mongo_configuration(
    mock_gemini, mock_groq, mock_generation, mock_poem, mock_client, mock_cursor, client
):
    mock_cursor.return_value.__enter__.return_value.fetchone.return_value = (1,)
    response = client.get("/api/health/")

    assert response.status_code == 503
    assert response.json()["mongo"] is False
    assert response.json()["mongo_status"] == "not_configured"


@pytest.mark.integration
@pytest.mark.requires_db
@patch("chat_UI.views.answer_with_router")
@patch("chat_UI.views.count_user_messages_today")
@patch("chat_UI.views.save_message_to_mongo")
@patch("chat_UI.views.get_history_for_bot", return_value=[])
def test_chat_api_with_metadata(mock_history, mock_save, mock_count, mock_router, authenticated_client):
    """Test chat API trả về metadata đầy đủ."""
    mock_count.return_value = 0
    mock_router.return_value = {
        "intent": "domain",
        "answer": "Câu trả lời",
        "sources": ["source1"],
        "verification": {"quotes": []},
        "harness": {"flow": "domain-qa", "quality": {"status": "grounded"}},
        "elapsed_ms": 100.0,
    }

    payload = {"message": "Test query", "k": 5}

    response = authenticated_client.post("/api/chat", data=json.dumps(payload), content_type="application/json")

    assert response.status_code == 200
    data = json.loads(response.content)
    assert "intent" in data
    assert "verification" in data
    assert "elapsed_ms" in data
    assert data["harness"]["flow"] == "domain-qa"


@pytest.mark.integration
@pytest.mark.requires_db
def test_chat_api_requires_login(client):
    """Test chat API yêu cầu login."""
    payload = {"message": "Test query", "k": 5}

    response = client.post("/api/chat/", data=json.dumps(payload), content_type="application/json")

    # Redirect to login hoặc 401/403
    assert response.status_code in [302, 401, 403]
