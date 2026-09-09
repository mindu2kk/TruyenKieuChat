"""
Shared test fixtures and configurations for pytest.
"""

import os
import pytest
from unittest.mock import Mock, patch

# Configure Django settings before importing Django modules
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project.settings")

# Configure test database
pytest_plugins = ["pytest_django"]


@pytest.fixture
def mock_mongo_client():
    """Mock MongoDB client for tests."""
    with patch("pymongo.MongoClient") as mock_client:
        mock_db = Mock()
        mock_collection = Mock()
        mock_db.__getitem__ = Mock(return_value=mock_collection)
        mock_client.return_value.__getitem__ = Mock(return_value=mock_db)
        yield mock_client


@pytest.fixture
def mock_gemini_client():
    """Mock Google Gemini API client for tests."""
    with patch("app.generation._setup") as mock_setup:
        mock_response = Mock()
        mock_response.text = "Mocked response from Gemini"
        mock_setup.return_value.models.generate_content.return_value = mock_response
        yield mock_setup


@pytest.fixture
def sample_poem_line():
    """Sample poem line for testing."""
    return "Trăm năm trong cõi người ta"


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "Tìm câu thơ về Thúy Kiều"


@pytest.fixture
def sample_chunk():
    """Sample chunk for testing."""
    return {
        "text": "Trăm năm trong cõi người ta...",
        "metadata": {
            "source": "poem.txt",
            "line_number": 1,
            "char_offset": 0,
        },
    }


@pytest.fixture
def mock_embedding():
    """Mock embedding vector for testing."""
    return [0.1] * 768  # Mock 768-dimensional embedding


@pytest.fixture
def env_vars(monkeypatch):
    """Set environment variables for testing."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017/")
    monkeypatch.setenv("MONGO_DB", "kieu_bot_test")
    monkeypatch.setenv("DEBUG", "0")
    yield
    # Cleanup after test


# Mark all tests as requiring database by default
pytestmark = pytest.mark.django_db
