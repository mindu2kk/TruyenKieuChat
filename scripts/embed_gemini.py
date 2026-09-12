# scripts/embed_gemini.py
# -*- coding: utf-8 -*-
"""
Embed corpus bằng Gemini gemini-embedding-001 và upsert vào Mongo Atlas.

ENV cần:
  GOOGLE_API_KEY
  MONGO_URI, MONGO_DB, MONGO_COL

Atlas index: path="vector", dimensions=768, similarity=cosine.
"""

import os, time, random, numbers, re
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

import google.generativeai as genai

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunk_store import configured_chunk_dir, iter_chunks
from app.corpus_quality import content_hash

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("MONGO_DB", "kieu_bot")
COL_NAME  = os.getenv("MONGO_COL", "chunks")
CHUNKS_DIR = configured_chunk_dir()

# Model tên chuẩn của Gemini Embeddings:
EMB_MODEL = os.getenv("GEMINI_EMB_MODEL", "models/gemini-embedding-001")
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "32"))
TASK_TYPE = os.getenv("EMB_TASK_TYPE", "RETRIEVAL_DOCUMENT")  # hoặc RETRIEVAL_QUERY

assert GOOGLE_API_KEY, "GOOGLE_API_KEY chưa có"
assert MONGO_URI, "MONGO_URI chưa có"

genai.configure(api_key=GOOGLE_API_KEY)
client = MongoClient(MONGO_URI)
col = client[DB_NAME][COL_NAME]

# ---------- utilities ----------
def _looks_like_vector(x) -> bool:
    if isinstance(x, (list, tuple)) and x and all(isinstance(v, numbers.Real) for v in x):
        return True
    return False

def _parse_gemini_embed_response(res):
    """Trả về list[float] từ nhiều dạng phản hồi của SDK."""
    # 1) dict dạng mới: {"embedding":{"values":[...]}}
    if isinstance(res, dict):
        if "error" in res:
            raise RuntimeError(f"Gemini error: {res.get('error')}")
        emb = res.get("embedding")
        if isinstance(emb, dict) and _looks_like_vector(emb.get("values")):
            return emb["values"]
        # 2) dict dạng: {"embedding":[...]}
        if _looks_like_vector(emb):
            return list(emb)
        # 3) dict batch: {"embeddings":[{"values":[...]}]}
        embs = res.get("embeddings")
        if isinstance(embs, list) and embs and isinstance(embs[0], dict) and _looks_like_vector(embs[0].get("values")):
            return embs[0]["values"]
    # 4) object: res.embedding.values
    emb = getattr(res, "embedding", None)
    vals = getattr(emb, "values", None) if emb is not None else None
    if _looks_like_vector(vals):
        return list(vals)
    # 5) object batch: res.embeddings[i].values
    embs = getattr(res, "embeddings", None)
    if isinstance(embs, list) and embs:
        v = getattr(embs[0], "values", None)
        if _looks_like_vector(v):
            return list(v)
    # 6) object dạng trực tiếp: res.embedding (list)
    if _looks_like_vector(emb):
        return list(emb)
    raise RuntimeError("Không đọc được embedding từ phản hồi Gemini (single).")

def _parse_gemini_embed_batch(res):
    """Trả về list[list[float]] từ nhiều dạng batch response."""
    # dict batch
    if isinstance(res, dict):
        if "error" in res:
            raise RuntimeError(f"Gemini error: {res.get('error')}")
        direct = res.get("embedding")
        if isinstance(direct, list) and direct and all(_looks_like_vector(item) for item in direct):
            return [list(item) for item in direct]
        embs = res.get("embeddings")
        if isinstance(embs, list):
            out = []
            for e in embs:
                if isinstance(e, dict) and _looks_like_vector(e.get("values")):
                    out.append(e["values"])
                elif _looks_like_vector(e):  # đôi khi e đã là list
                    out.append(list(e))
            if out:
                return out
    # object batch
    embs = getattr(res, "embeddings", None)
    if isinstance(embs, list) and embs:
        out = []
        for e in embs:
            vals = getattr(e, "values", None)
            if _looks_like_vector(vals):
                out.append(list(vals))
            elif _looks_like_vector(e):
                out.append(list(e))
        if out:
            return out
    # single rơi về
    one = _parse_gemini_embed_response(res)
    if _looks_like_vector(one):
        return [one]
    raise RuntimeError("Không đọc được embeddings từ phản hồi Gemini (batch).")

def _embed_single(text: str) -> List[float]:
    # Thêm output_dimensionality cho chắc (một số bản hỗ trợ)
    res = genai.embed_content(
        model=EMB_MODEL,
        content=text,
        task_type=TASK_TYPE,
        output_dimensionality=768
    )
    try:
        return _parse_gemini_embed_response(res)
    except Exception as exc:
        # In gọn 1 phần phản hồi để debug khi cần
        preview = str(res)
        if len(preview) > 300:
            preview = preview[:300] + "…"
        raise RuntimeError(f"Parse single embed fail: {exc}. Raw={preview}")

def embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    last_error = None
    for attempt in range(6):
        try:
            res = genai.embed_content(
                model=EMB_MODEL,
                content=texts,
                task_type=TASK_TYPE,
                output_dimensionality=768,
                request_options={"timeout": 60},
            )
            return _parse_gemini_embed_batch(res)
        except Exception as exc:
            last_error = exc
            if attempt == 5:
                break
            retry_match = re.search(r"retry in ([0-9.]+)s", str(exc), flags=re.IGNORECASE)
            delay = (
                float(retry_match.group(1)) + 1.0
                if retry_match
                else min(30.0, 2.0 ** attempt + random.random())
            )
            print(f"[RETRY] embedding batch attempt={attempt + 2} delay={delay:.1f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"Embedding batch failed after retries: {last_error}")

# ---------- main ----------
def main():
    col.create_index("meta.type")
    col.create_index("meta.source")
    col.create_index("meta.source_id")
    col.create_index("meta.source_tier")
    col.create_index("meta.content_hash")

    remote_vectors = {}
    for remote in col.find(
        {"vector.0": {"$exists": True}},
        {"text": 1, "vector": 1, "meta.content_hash": 1},
    ):
        remote_meta = remote.get("meta") or {}
        digest = remote_meta.get("content_hash") or content_hash(str(remote.get("text") or ""))
        vector = remote.get("vector")
        if digest and isinstance(vector, list) and vector:
            remote_vectors.setdefault(str(digest), vector)
    print(f"[INFO] reusable_vectors={len(remote_vectors)}", flush=True)

    total = 0
    embedded_total = 0
    reused_total = 0
    batch: List[Dict] = []

    def flush():
        nonlocal total, embedded_total, reused_total
        if not batch:
            return
        ids = [item["_id"] for item in batch]
        existing = {
            str(item["_id"]): item
            for item in col.find(
                {"_id": {"$in": ids}},
                {"_id": 1, "meta.content_hash": 1, "vector": {"$slice": 1}},
            )
        }
        pending = []
        metadata_updates = []
        for item in batch:
            remote = existing.get(str(item["_id"])) or {}
            remote_meta = remote.get("meta") or {}
            if remote.get("vector") and remote_meta.get("content_hash") == item["meta"].get("content_hash"):
                # Content vectors remain valid, but quality metadata may have
                # changed (sections, source tier, repaired positions, etc.).
                item["meta"]["embedding_model"] = EMB_MODEL
                metadata_updates.append(
                    UpdateOne(
                        {"_id": item["_id"]},
                        {"$set": {"text": item["text"], "meta": item["meta"]}},
                    )
                )
                continue
            pending.append(item)
        if metadata_updates:
            col.bulk_write(metadata_updates, ordered=False)
        if not pending:
            total += len(batch)
            print(f"[META] {len(metadata_updates)} current docs refreshed (total {total})", flush=True)
            batch.clear()
            return
        reused = []
        needs_embedding = []
        for item in pending:
            digest = str(item["meta"].get("content_hash") or content_hash(item["text"]))
            reusable = remote_vectors.get(digest)
            if reusable:
                item["vector"] = reusable
                reused.append(item)
            else:
                needs_embedding.append(item)

        embed_started = time.monotonic()
        vecs = embed_batch([x["text"] for x in needs_embedding]) if needs_embedding else []
        if len(vecs) != len(needs_embedding):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(needs_embedding)}, received {len(vecs)}"
            )
        ops = []
        for x in reused:
            x["meta"]["embedding_model"] = EMB_MODEL
            ops.append(UpdateOne({"_id": x["_id"]}, {"$set": x}, upsert=True))
        for x, v in zip(needs_embedding, vecs):
            x["vector"] = v
            x["meta"]["embedding_model"] = EMB_MODEL
            ops.append(UpdateOne({"_id": x["_id"]}, {"$set": x}, upsert=True))
        if ops:
            for attempt in range(3):
                try:
                    col.bulk_write(ops, ordered=False)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1.0 * (attempt + 1))
        total += len(batch)
        embedded_total += len(needs_embedding)
        reused_total += len(reused)
        print(
            f"[OK] upsert={len(pending)} reused={len(reused)} embedded={len(needs_embedding)} "
            f"total={total} embedded_total={embedded_total} reused_total={reused_total}",
            flush=True,
        )
        if needs_embedding:
            minimum_interval = len(needs_embedding) * 60.0 / 95.0
            remaining = minimum_interval - (time.monotonic() - embed_started)
            if remaining > 0:
                print(f"[THROTTLE] {remaining:.1f}s to respect free-tier quota", flush=True)
                time.sleep(remaining)
        batch.clear()

    print(f"[INFO] chunk_dir={CHUNKS_DIR} batch={BATCH_SIZE}", flush=True)
    for d in iter_chunks(CHUNKS_DIR):
        batch.append(d)
        if len(batch) >= BATCH_SIZE:
            flush()
    if batch:
        flush()

    print(f"Done. Total docs: {total}. DB: {DB_NAME}.{COL_NAME}", flush=True)

if __name__ == "__main__":
    main()
