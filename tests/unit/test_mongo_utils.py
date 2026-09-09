from unittest.mock import Mock, patch

import pytest

import chat_UI.mongo_utils as mongo_utils


@pytest.mark.unit
def test_mongo_client_is_lazy_and_has_bounded_timeouts(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://example.invalid:27017")
    monkeypatch.setenv("MONGO_TIMEOUT_MS", "1500")
    monkeypatch.setattr(mongo_utils, "_mongo_client", None)
    client = Mock()

    with patch("chat_UI.mongo_utils.MongoClient", return_value=client) as mongo_client:
        assert mongo_utils.get_mongo_client() is client

    mongo_client.assert_called_once_with(
        "mongodb://example.invalid:27017",
        serverSelectionTimeoutMS=1500,
        connectTimeoutMS=1500,
        socketTimeoutMS=3000,
        maxPoolSize=20,
    )
    client.admin.command.assert_not_called()
