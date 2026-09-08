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
from unittest.mock import patch, Mock


@pytest.fixture
def client():
    """Django test client."""
    return Client()


@pytest.fixture
def test_user(db):
    """Tạo test user."""
    return User.objects.create_user(
        username='testuser',
        password='testpass123'
    )


@pytest.fixture
def authenticated_client(client, test_user):
    """Client đã authenticated."""
    client.force_login(test_user)
    return client


@pytest.mark.integration
@pytest.mark.requires_db
def test_home_view_requires_login(client):
    """Test home view yêu cầu login."""
    response = client.get('/')

    # Nếu chưa login, redirect đến login page
    assert response.status_code in [302, 401]


@pytest.mark.integration
@pytest.mark.requires_db
def test_home_view_authenticated(authenticated_client):
    """Test home view khi đã authenticated."""
    response = authenticated_client.get('/')

    assert response.status_code == 200
    assert 'gemini_ok' in response.context or 'gemini_ok' in str(response.content)
    assert 'poem_ok' in response.context or 'poem_ok' in str(response.content)


@pytest.mark.integration
@pytest.mark.requires_db
def test_chat_api_get(authenticated_client):
    """Test chat API GET request."""
    response = authenticated_client.get('/api/chat')

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data.get("ok") is True


@pytest.mark.integration
@pytest.mark.requires_db
@patch('chat_UI.views.answer_with_router')
@patch('chat_UI.views.count_user_messages_today')
@patch('chat_UI.views.save_message_to_mongo')
@patch('chat_UI.views.get_history_for_bot', return_value=[])
def test_chat_api_post_success(mock_history, mock_save, mock_count, mock_router, authenticated_client):
    """Test chat API POST request thành công."""
    mock_count.return_value = 0  # Chưa dùng quota
    mock_router.return_value = {
        "intent": "domain",
        "answer": "Câu trả lời test",
        "sources": []
    }

    payload = {
        "message": "Thúy Kiều là ai?",
        "k": 5,
        "model": "gemini-2.0-flash",
        "long_answer": False
    }

    response = authenticated_client.post(
        '/api/chat',
        data=json.dumps(payload),
        content_type='application/json'
    )

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data.get("ok") is True
    assert "answer" in data
    assert data["answer"] == "Câu trả lời test"
    assert "intent" in data
    mock_router.assert_called_once()
    mock_save.assert_called()


@pytest.mark.integration
@pytest.mark.requires_db
@patch('chat_UI.views.count_user_messages_today')
def test_chat_api_quota_exceeded(mock_count, authenticated_client):
    """Test chat API khi vượt quota."""
    mock_count.return_value = 20  # Đã dùng hết quota

    payload = {
        "message": "Test query",
        "k": 5
    }

    response = authenticated_client.post(
        '/api/chat',
        data=json.dumps(payload),
        content_type='application/json'
    )

    assert response.status_code == 429
    data = json.loads(response.content)
    assert data.get("ok") is False
    assert "quota" in data.get("error", "")
    assert "20 câu hỏi" in data.get("message", "")


@pytest.mark.integration
@pytest.mark.requires_db
def test_chat_api_invalid_json(authenticated_client):
    """Test chat API với invalid JSON."""
    response = authenticated_client.post(
        '/api/chat/',
        data="invalid json",
        content_type='application/json'
    )

    assert response.status_code == 400
    data = json.loads(response.content)
    assert data.get("ok") is False
    assert "error" in data


@pytest.mark.integration
@pytest.mark.requires_db
def test_chat_api_empty_message(authenticated_client):
    """Test chat API với empty message."""
    payload = {
        "message": "",
        "k": 5
    }

    response = authenticated_client.post(
        '/api/chat',
        data=json.dumps(payload),
        content_type='application/json'
    )

    assert response.status_code == 400
    data = json.loads(response.content)
    assert data.get("ok") is False
    assert "error" in data


@pytest.mark.integration
@pytest.mark.requires_db
@patch('chat_UI.views.answer_with_router')
@patch('chat_UI.views.count_user_messages_today')
@patch('chat_UI.views.save_message_to_mongo')
@patch('chat_UI.views.get_history_for_bot', return_value=[])
def test_chat_api_error_handling(mock_history, mock_save, mock_count, mock_router, authenticated_client):
    """Test chat API xử lý lỗi từ orchestrator."""
    mock_count.return_value = 0
    mock_router.side_effect = Exception("Backend error")

    payload = {
        "message": "Test query",
        "k": 5
    }

    response = authenticated_client.post(
        '/api/chat',
        data=json.dumps(payload),
        content_type='application/json'
    )

    assert response.status_code == 500
    data = json.loads(response.content)
    assert data.get("ok") is False
    assert "error" in data
    assert "backend" in data.get("error", "")


@pytest.mark.integration
@pytest.mark.requires_db
@patch('chat_UI.views.get_history_for_api')
def test_history_api_get(mock_get_history, authenticated_client):
    """Test history API GET request."""
    mock_get_history.return_value = [
        {"role": "user", "content": "Câu hỏi 1"},
        {"role": "assistant", "content": "Câu trả lời 1"}
    ]

    response = authenticated_client.get('/api/history/')

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data.get("ok") is True
    assert "messages" in data
    assert len(data["messages"]) == 2


@pytest.mark.integration
@pytest.mark.requires_db
@patch('chat_UI.views.clear_user_history')
def test_history_api_clear(mock_clear, authenticated_client):
    """Test history API clear history."""
    mock_clear.return_value = None

    response = authenticated_client.delete('/api/history/')

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data.get("ok") is True
    assert "messages" in data
    assert len(data["messages"]) == 0
    mock_clear.assert_called_once()


@pytest.mark.integration
@patch('chat_UI.views.get_mongo_client')
@patch('chat_UI.views.poem_ready', return_value=True)
@patch('chat_UI.views.is_gemini_configured', return_value=True)
def test_health_api_ready(mock_gemini, mock_poem, mock_client, client):
    mock_client.return_value.admin.command.return_value = {"ok": 1}

    response = client.get('/api/health/')

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "mongo": True,
        "mongo_status": "ready",
        "gemini_configured": True,
        "poem_ready": True,
    }


@pytest.mark.integration
@patch('chat_UI.views.get_mongo_client', side_effect=ValueError("MONGO_URI missing"))
@patch('chat_UI.views.poem_ready', return_value=True)
@patch('chat_UI.views.is_gemini_configured', return_value=True)
def test_health_api_reports_missing_mongo_configuration(
    mock_gemini, mock_poem, mock_client, client
):
    response = client.get('/api/health/')

    assert response.status_code == 503
    assert response.json()["mongo"] is False
    assert response.json()["mongo_status"] == "not_configured"


@pytest.mark.integration
@pytest.mark.requires_db
@patch('chat_UI.views.answer_with_router')
@patch('chat_UI.views.count_user_messages_today')
@patch('chat_UI.views.save_message_to_mongo')
@patch('chat_UI.views.get_history_for_bot', return_value=[])
def test_chat_api_with_metadata(mock_history, mock_save, mock_count, mock_router, authenticated_client):
    """Test chat API trả về metadata đầy đủ."""
    mock_count.return_value = 0
    mock_router.return_value = {
        "intent": "domain",
        "answer": "Câu trả lời",
        "sources": ["source1"],
        "verification": {"quotes": []},
        "elapsed_ms": 100.0
    }

    payload = {
        "message": "Test query",
        "k": 5
    }

    response = authenticated_client.post(
        '/api/chat',
        data=json.dumps(payload),
        content_type='application/json'
    )

    assert response.status_code == 200
    data = json.loads(response.content)
    assert "intent" in data
    assert "verification" in data
    assert "elapsed_ms" in data


@pytest.mark.integration
@pytest.mark.requires_db
def test_chat_api_requires_login(client):
    """Test chat API yêu cầu login."""
    payload = {
        "message": "Test query",
        "k": 5
    }

    response = client.post(
        '/api/chat/',
        data=json.dumps(payload),
        content_type='application/json'
    )

    # Redirect to login hoặc 401/403
    assert response.status_code in [302, 401, 403]
