from app.embedding_cache import QueryEmbeddingCache, embedding_cache_key


def test_embedding_cache_round_trip_without_storing_raw_query(tmp_path):
    cache = QueryEmbeddingCache(tmp_path / "query-cache.sqlite3")
    key = embedding_cache_key("model", "RETRIEVAL_QUERY", 3, "Duyên Tần Tấn")

    cache.put(key, [0.1, 0.2, 0.3])

    assert cache.get(key, 3) == [0.1, 0.2, 0.3]
    assert cache.get(key, 2) is None
    assert "Duyên Tần Tấn" not in (tmp_path / "query-cache.sqlite3").read_bytes().decode(
        "utf-8", errors="ignore"
    )


def test_embedding_cache_key_is_model_and_task_specific():
    first = embedding_cache_key("model-a", "RETRIEVAL_QUERY", 768, "câu hỏi")

    assert first == embedding_cache_key("model-a", "RETRIEVAL_QUERY", 768, "câu hỏi")
    assert first != embedding_cache_key("model-b", "RETRIEVAL_QUERY", 768, "câu hỏi")
