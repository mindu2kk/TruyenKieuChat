# scripts/embed_gemini.py
# -*- coding: utf-8 -*-
"""
Embed corpus bằng Gemini text-embedding-004 và upsert vào Mongo Atlas.

ENV cần:
  GOOGLE_API_KEY
  MONGO_URI, MONGO_DB, MONGO_COL

Atlas index: path="vector", dimensions=768, similarity=cosine.
"""

import os, json, time, random, numbers
from pathlib import Path
from typing import Iterator, Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("MONGO_DB", "kieu_bot")
COL_NAME  = os.getenv("MONGO_COL", "chunks")
CHUNKS_DIR = Path("data/rag_chunks")

# Model tên chuẩn của Gemini Embeddings:
EMB_MODEL = os.getenv("GEMINI_EMB_MODEL", "models/text-embedding-004")
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "64"))
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

def iter_chunks() -> Iterator[Dict]:
    for p in sorted(CHUNKS_DIR.glob("*.txt")):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if not raw.startswith("###META###"):
            continue
        meta_line, _, body = raw.partition("\n")
        try:
            meta = json.loads(meta_line.replace("###META###","").strip())
        except Exception:
            meta = {}
        text = body.strip()
        if not text:
            continue
        _id = meta.get("id") or p.stem
        yield {"_id": _id, "text": text, "meta": meta}

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
    except Exception as e:
        # In gọn 1 phần phản hồi để debug khi cần
        preview = str(res)
        if len(preview) > 300:
            preview = preview[:300] + "…"
        raise RuntimeError(f"Parse single embed fail: {e}. Raw={preview}")

def embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    # Thử batch một phát:
    try:
        res = genai.embed_content(
            model=EMB_MODEL,
            content=texts,
            task_type=TASK_TYPE,
            output_dimensionality=768
        )
        return _parse_gemini_embed_batch(res)
    except Exception:
        # Fallback: loop single + retry/backoff
        out = []
        for i, t in enumerate(texts):
            for attempt in range(4):
                try:
                    out.append(_embed_single(t))
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    time.sleep(0.6 * (attempt + 1) + random.random() * 0.3)
        return out

# ---------- main ----------
def main():
    col.create_index("meta.type")
    col.create_index("meta.source")

    total = 0
    batch: List[Dict] = []

    def flush():
        nonlocal batch, total
        if not batch:
            return
        vecs = embed_batch([x["text"] for x in batch])
        ops = []
        for x, v in zip(batch, vecs):
            x["vector"] = v
            ops.append(UpdateOne({"_id": x["_id"]}, {"$set": x}, upsert=True))
        if ops:
            for attempt in range(3):
                try:
                    col.bulk_write(ops, ordered=False)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(1.0 * (attempt + 1))
        total += len(batch)
        print(f"[OK] upsert {len(batch)} docs (total {total})")
        batch.clear()

    for d in iter_chunks():
        batch.append(d)
        if len(batch) >= BATCH_SIZE:
            flush()
    if batch:
        flush()

    print(f"Done. Total docs: {total}. DB: {DB_NAME}.{COL_NAME}")

if __name__ == "__main__":
    main()
