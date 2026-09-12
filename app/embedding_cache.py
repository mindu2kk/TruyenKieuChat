"""Persistent, privacy-conscious cache for query embeddings.

Only a SHA-256 key and the numeric vector are stored; raw user queries are not
written to disk. SQLite gives safe concurrent reads across Django workers.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import unicodedata
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


def embedding_cache_key(model: str, task_type: str, dimensions: int, query: str) -> str:
    normalized = unicodedata.normalize("NFC", query or "").strip()
    payload = f"{model}\0{task_type}\0{dimensions}\0{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QueryEmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS query_embeddings "
                    "(cache_key TEXT PRIMARY KEY, dimensions INTEGER NOT NULL, vector_json TEXT NOT NULL)"
                )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def get(self, cache_key: str, dimensions: int) -> list[float] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT vector_json FROM query_embeddings WHERE cache_key = ? AND dimensions = ?",
                (cache_key, dimensions),
            ).fetchone()
        if not row:
            return None
        vector = json.loads(row[0])
        if not isinstance(vector, list) or len(vector) != dimensions:
            return None
        return [float(value) for value in vector]

    def put(self, cache_key: str, vector: Sequence[float]) -> None:
        values = [float(value) for value in vector]
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO query_embeddings(cache_key, dimensions, vector_json) VALUES (?, ?, ?)",
                    (cache_key, len(values), json.dumps(values, separators=(",", ":"))),
                )


@lru_cache(maxsize=1)
def get_query_embedding_cache() -> QueryEmbeddingCache | None:
    enabled = (os.getenv("QUERY_EMBED_CACHE_ENABLED", "1") or "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    configured = os.getenv("QUERY_EMBED_CACHE_PATH")
    path = Path(configured) if configured else ROOT / ".cache" / "query_embeddings.sqlite3"
    return QueryEmbeddingCache(path.resolve())


__all__ = ["QueryEmbeddingCache", "embedding_cache_key", "get_query_embedding_cache"]
