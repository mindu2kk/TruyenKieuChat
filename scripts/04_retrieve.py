# scripts/04_retrieve.py
# -*- coding: utf-8 -*-
"""
Retrieve ngữ cảnh từ MongoDB Atlas Vector Search.

HỖ TRỢ 2 CHẾ ĐỘ TRUY VẤN VECTORS:
- SBERT/E5:  dùng SentenceTransformer để encode query (mặc định trước đây).
- GEMINI:    dùng Gemini text-embedding-004 với task_type="RETRIEVAL_QUERY".

Cấu hình qua ENV:
  MONGO_URI            : URI Atlas
  MONGO_DB             : tên DB (default: kieu_bot)
  MONGO_COL            : tên collection (default: chunks)
  INDEX_NAME           : tên Search Index cho vector search (default: vector_index)
  RETRIEVER            : "sbert" | "gemini"  (default: "sbert")
  EMBEDDING_MODEL      : (khi sbert) vd "intfloat/multilingual-e5-base"
  GOOGLE_API_KEY       : (khi gemini)

Sử dụng:
  from scripts.04_retrieve import retrieve_context
  hits = retrieve_context("Truyện Kiều có bao nhiêu câu?", k=5, num_candidates=100)
  -> mỗi phần tử: {"text": str, "meta": {...}, "score": float}
"""

from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pymongo import MongoClient

# === Load ENV ===
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

MONGO_URI   = os.getenv("MONGO_URI")
DB_NAME     = os.getenv("MONGO_DB", "kieu_bot")
COL_NAME    = os.getenv("MONGO_COL", "chunks")
INDEX_NAME  = os.getenv("INDEX_NAME", "vector_index")

RETRIEVER   = os.getenv("RETRIEVER", "sbert").strip().lower()   # "sbert" | "gemini"

# SBERT model name (khi RETRIEVER = sbert)
EMB_MODEL   = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# Gemini API key (khi RETRIEVER = gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# ====== Embedding Providers ======
class _SbertProvider:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode_query(self, q: str) -> List[float]:
        # Với e5 đúng chuẩn, có thể tiền tố "query: " để tăng chất lượng.
        query_text = q if "e5" not in (EMB_MODEL or "").lower() else f"query: {q}"
        return self.model.encode(query_text, normalize_embeddings=True).tolist()


class _GeminiProvider:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY trống: không thể dùng chế độ GEMINI.")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        # tên model chuẩn; giữ đồng bộ với script embed_gemini.py
        self.model_name = os.getenv("GEMINI_EMB_MODEL", "models/text-embedding-004")

    @staticmethod
    def _looks_like_vec(x) -> bool:
        from numbers import Real
        return isinstance(x, (list, tuple)) and x and all(isinstance(v, Real) for v in x)

    def _parse_single(self, res) -> List[float]:
        # chấp nhận nhiều format response khác nhau của SDK
        if isinstance(res, dict):
            if "error" in res:
                raise RuntimeError(f"Gemini error: {res.get('error')}")
            emb = res.get("embedding")
            if isinstance(emb, dict) and self._looks_like_vec(emb.get("values")):
                return emb["values"]
            if self._looks_like_vec(emb):
                return list(emb)
        emb_obj = getattr(res, "embedding", None)
        vals = getattr(emb_obj, "values", None) if emb_obj is not None else None
        if self._looks_like_vec(vals):
            return list(vals)
        if self._looks_like_vec(emb_obj):
            return list(emb_obj)
        raise RuntimeError("Không đọc được embedding từ phản hồi Gemini (query).")

    def encode_query(self, q: str) -> List[float]:
        res = self.genai.embed_content(
            model=self.model_name,
            content=q,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,  # nên trùng với vector dim khi indexing
        )
        return self._parse_single(res)

# ====== Clients cache ======
@lru_cache(maxsize=1)
def _get_clients():
    assert MONGO_URI, "Thiếu MONGO_URI trong .env"
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COL_NAME]

    if RETRIEVER == "gemini":
        embedder = _GeminiProvider(GOOGLE_API_KEY)
    else:
        embedder = _SbertProvider(EMB_MODEL)

    return col, embedder

# ====== Public API ======
def retrieve_context(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    num_candidates: int = 100,
    min_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Truy vấn vector từ Atlas:
      - numCandidates: số ứng viên cho ANN
      - limit (k):     số kết quả trả về
    filters: document-level filter Atlas (vd: {"meta.type": {"$in": ["analysis","summary","bio","poem"]}})
    min_score: nếu đặt, lọc bớt các kết quả score < min_score.
    """
    col, embedder = _get_clients()
    qvec = embedder.encode_query(query)

    stage = {
        "$vectorSearch": {
            "index": INDEX_NAME,
            "path": "vector",
            "queryVector": qvec,
            "numCandidates": int(num_candidates),
            "limit": int(k),
        }
    }
    if filters:
        stage["$vectorSearch"]["filter"] = filters

    pipeline = [
        stage,
        {"$project": {
            "_id": 0,
            "text": 1,
            "meta": 1,
            "score": {"$meta": "vectorSearchScore"}
        }},
    ]
    docs = list(col.aggregate(pipeline))

    if min_score is not None:
        docs = [d for d in docs if float(d.get("score", 0.0)) >= float(min_score)]

    return docs

# ====== CLI nhỏ để test nhanh ======
if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]) or "Truyện Kiều có bao nhiêu câu?"
    hs = retrieve_context(q, k=5, num_candidates=100, filters={"meta.type": {"$in": ["analysis","summary","bio","poem"]}})
    for i, h in enumerate(hs, 1):
        meta = h.get("meta", {})
        print(f"{i:2d}. score={h.get('score'):.3f}  src={meta.get('source')}")
    # In JSON đầy đủ (nếu cần)
    # print(json.dumps(hs, ensure_ascii=False, indent=2))
