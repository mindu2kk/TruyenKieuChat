# app/retriever.py
# -*- coding: utf-8 -*-
"""
Hybrid retriever cho corpus (MongoDB Atlas Vector + Text), hợp nhất bằng RRF.

ENV cần:
- MONGO_URI
- MONGO_DB  (mặc định: kieu_bot)
- MONGO_COL (mặc định: chunks)
"""

import os
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ====== ENV / Mongo ======
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("MONGO_DB", "kieu_bot")
COL_NAME  = os.getenv("MONGO_COL", "chunks")
assert MONGO_URI, "Thiếu MONGO_URI trong .env"

_client = MongoClient(MONGO_URI)
_col = _client[DB_NAME][COL_NAME]

# ====== Embedding model (E5) ======
_EMB_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
_embedder = SentenceTransformer(_EMB_MODEL)

def embed_query(q: str) -> List[float]:
    """E5 query embedding (prefix 'query: ')"""
    return _embedder.encode(["query: " + q], normalize_embeddings=True).tolist()[0]

# ====== RRF Fuse ======
def rrf_fuse(results_lists: List[List[Dict[str, Any]]], k: int = 60, c: float = 60.0) -> List[Dict[str, Any]]:
    """
    results_lists: danh sách các list kết quả (vector, text ...).
    Mỗi item nên có _id và score (cùng dấu, càng lớn càng tốt).
    """
    agg: Dict[Any, Dict[str, Any]] = {}
    for lst in results_lists:
        for rank, item in enumerate(lst[:k], start=1):
            key = item["_id"]
            found = agg.get(key)
            if not found:
                agg[key] = {"item": item, "rrf": 0.0}
            agg[key]["rrf"] += 1.0 / (c + rank)
    fused = sorted(agg.values(), key=lambda x: x["rrf"], reverse=True)
    return [x["item"] for x in fused]

# ====== Mongo Queries ======
def _vector_search(query_vec: List[float], limit: int = 40, num_candidates: int = 400) -> List[Dict[str, Any]]:
    """
    Yêu cầu đã cấu hình Atlas Vector Search index cho field 'vector'
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": os.getenv("ATLAS_VECTOR_INDEX", "vec_chunks"),
                "path": "vector",
                "queryVector": query_vec,
                "numCandidates": num_candidates,
                "limit": limit,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"text": 1, "meta": 1, "score": 1}},
    ]
    return list(_col.aggregate(pipeline))

def _text_search(query: str, limit: int = 40) -> List[Dict[str, Any]]:
    """
    Cần có text index: db.chunks.createIndex({text: "text"})
    """
    pipeline = [
        {"$match": {"$text": {"$search": query}}},
        {"$addFields": {"score": {"$meta": "textScore"}}},
        {"$sort": {"score": -1}},
        {"$limit": limit},
        {"$project": {"text": 1, "meta": 1, "score": 1}},
    ]
    return list(_col.aggregate(pipeline))

def _boost_poem(results: List[Dict[str, Any]], bonus: float = 0.5) -> List[Dict[str, Any]]:
    """
    Tăng nhẹ score cho chunk thơ (meta.type == 'poem') để ưu tiên khi close-reading.
    """
    out = []
    for r in results:
        m = r.get("meta", {})
        sc = float(r.get("score", 0.0))
        if m.get("type") == "poem":
            sc += bonus
        r2 = dict(r)
        r2["score"] = sc
        out.append(r2)
    return out

def retrieve_hybrid(
    query: str,
    k: int = 60,
    prefer_poem_source: bool = False,
    vector_limit: int = 40,
    text_limit: int = 40,
    num_candidates: int = 400,
) -> List[Dict[str, Any]]:
    """
    Trả list docs: {"_id","text","meta",{...}, "score": float}
    """
    q_vec = embed_query(query)
    vec_hits = _vector_search(q_vec, limit=vector_limit, num_candidates=num_candidates)
    txt_hits = _text_search(query, limit=text_limit)

    if prefer_poem_source:
        vec_hits = _boost_poem(vec_hits, bonus=0.5)
        txt_hits = _boost_poem(txt_hits, bonus=0.5)

    fused = rrf_fuse([vec_hits, txt_hits], k=k, c=60.0)
    return fused
