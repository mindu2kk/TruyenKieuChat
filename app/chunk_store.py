"""Read quality-controlled chunk files without external dependencies."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator

from .corpus_quality import deduplicate_records, enrich_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CHUNK_DIR = PROJECT_ROOT / "data" / "rag_chunks"
CLEAN_CHUNK_DIR = PROJECT_ROOT / "data" / "rag_chunks_clean"


def configured_chunk_dir() -> Path:
    configured = os.getenv("RAG_CHUNKS_DIR", "").strip()
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return CLEAN_CHUNK_DIR if CLEAN_CHUNK_DIR.exists() else LEGACY_CHUNK_DIR


def read_chunk(path: Path) -> Dict[str, Any] | None:
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    if not raw.startswith("###META###"):
        return None
    meta_line, separator, body = raw.partition("\n")
    if not separator:
        return None
    try:
        meta = json.loads(meta_line.replace("###META###", "", 1).strip())
    except (TypeError, ValueError):
        return None
    text = body.strip()
    if not text:
        return None
    meta = enrich_metadata(meta, text)
    return {"_id": meta.get("id") or path.stem, "text": text, "meta": meta}


def iter_chunks(chunk_dir: Path | None = None, *, min_words: int = 1) -> Iterator[Dict[str, Any]]:
    directory = chunk_dir or configured_chunk_dir()
    records = []
    for path in sorted(directory.glob("*.txt")):
        record = read_chunk(path)
        if record is not None and len(record["text"].split()) >= min_words:
            records.append(record)
    yield from deduplicate_records(records)


__all__ = ["configured_chunk_dir", "iter_chunks", "read_chunk"]
