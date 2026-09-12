# scripts/02_embed_and_index_mongo.py
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from sentence_transformers import SentenceTransformer

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunk_store import configured_chunk_dir, iter_chunks

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("MONGO_DB", "kieu_bot")
COL_NAME  = os.getenv("MONGO_COL", "chunks")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")  # <-- base
BATCH_SZ  = int(os.getenv("EMBED_BATCH_SIZE", "128"))
CHUNKS_DIR = configured_chunk_dir()

assert MONGO_URI, "Thiếu MONGO_URI trong .env"

client = MongoClient(MONGO_URI)
col = client[DB_NAME][COL_NAME]

col.create_index("meta.type")
col.create_index("meta.source")
col.create_index("meta.source_id")
col.create_index("meta.source_tier")
col.create_index("meta.content_hash")
col.create_index([("text", "text")]) 

embedder = SentenceTransformer(EMB_MODEL)

def embed_texts_passage(texts: List[str]) -> List[List[float]]:
    # E5-spec: prefix "passage: "
    texts2 = [("passage: " + t) for t in texts]
    return embedder.encode(
        texts2,
        normalize_embeddings=True,
        batch_size=BATCH_SZ,
        show_progress_bar=False
    ).tolist()
    
def embed_query(q: str) -> List[float]:
    # E5: prefix "query: "
    return embedder.encode(["query: " + q], normalize_embeddings=True).tolist()[0]

def batched(iterable, n=128):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    # In dim để so với Atlas index
    dim = len(embed_texts_passage(["probe"])[0])
    print(f"[INFO] EMBEDDING_MODEL={EMB_MODEL} | dim={dim} (e5-base=768) | batch={BATCH_SZ}")

    total = 0
    print(f"[INFO] chunk_dir={CHUNKS_DIR}")
    for batch in tqdm(batched(iter_chunks(CHUNKS_DIR, min_words=5), n=BATCH_SZ), desc="Embedding & upserting"):
        texts = [d["text"] for d in batch]
        vecs  = embed_texts_passage(texts)

        ops = []
        for d, v in zip(batch, vecs):
            d["vector"] = v
            ops.append(UpdateOne({"_id": d["_id"]}, {"$set": d}, upsert=True))

        if ops:
            col.bulk_write(ops, ordered=False)
            total += len(ops)
    print(f"[DONE] Upsert {total} chunks → {DB_NAME}.{COL_NAME}")
    print("Nhớ đảm bảo Vector Index của Atlas có numDimensions = 768.")

if __name__ == "__main__":
    main()
