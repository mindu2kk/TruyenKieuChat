# app/hybrid_retriever.py
# -*- coding: utf-8 -*-
"""Hybrid retriever on MongoDB Atlas: vectorSearch (+ optional text search) with RRF fusion."""

from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient

# load .env ở tầng app
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# ====== ENV / Defaults ======
MONGO_URI    = os.getenv("MONGO_URI")
DB_NAME      = os.getenv("MONGO_DB", "kieu_bot")
COL_NAME     = os.getenv("MONGO_COL", "chunks")
INDEX_NAME   = os.getenv("INDEX_NAME", "vector_index")  # Atlas Vector index name
RETRIEVER    = (os.getenv("RETRIEVER", "sbert") or "sbert").lower()  # "sbert" | "gemini"
EMB_MODEL    = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
RRF_K        = int(os.getenv("RRF_K", "60"))

# ====== Embedding providers ======
class _SbertProvider:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.is_e5 = "e5" in (model_name or "").lower()

    def encode_query(self, q: str) -> List[float]:
        t = f"query: {q}" if self.is_e5 else q
        return self.model.encode(t, normalize_embeddings=True).tolist()

class _GeminiProvider:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY trống — không thể dùng GEMINI retriever.")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        # nên khớp với script embed/query khác của bạn
        self.model_name = os.getenv("GEMINI_EMB_MODEL", "models/text-embedding-004")

    @staticmethod
    def _looks_like_vec(x) -> bool:
        from numbers import Real
        return isinstance(x, (list, tuple)) and x and all(isinstance(v, Real) for v in x)

    def _parse_single(self, res) -> list[float]:
        # Hỗ trợ các layout: dict|object; embedding.values|embedding|res["data"][0]["embedding"] ...
        # 1) Kiểu dict
        if isinstance(res, dict):
            if "error" in res:
                raise RuntimeError(f"Gemini error: {res.get('error')}")
            emb = res.get("embedding")
            if isinstance(emb, dict) and self._looks_like_vec(emb.get("values")):
                return list(emb["values"])
            if self._looks_like_vec(emb):
                return list(emb)
            # một số SDK có res["data"][0]["embedding"]
            data = res.get("data")
            if isinstance(data, list) and data:
                emb2 = data[0].get("embedding") if isinstance(data[0], dict) else None
                if isinstance(emb2, dict) and self._looks_like_vec(emb2.get("values")):
                    return list(emb2["values"])
                if self._looks_like_vec(emb2):
                    return list(emb2)

        # 2) Kiểu object có .embedding hoặc .embedding.values
        emb_obj = getattr(res, "embedding", None)
        vals = getattr(emb_obj, "values", None) if emb_obj is not None else None
        if self._looks_like_vec(vals):
            return list(vals)
        if self._looks_like_vec(emb_obj):
            return list(emb_obj)

        raise RuntimeError("Không đọc được embedding từ phản hồi Gemini (query).")

    def encode_query(self, q: str) -> list[float]:
        res = self.genai.embed_content(
            model=self.model_name,
            content=q,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,  # khớp với Atlas index dim
        )
        return self._parse_single(res)
@lru_cache(maxsize=1)
def _get_clients():
    assert MONGO_URI, "Thiếu MONGO_URI — kiểm tra .env"
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COL_NAME]
    embedder = _SbertProvider(EMB_MODEL) if RETRIEVER != "gemini" else _GeminiProvider(GOOGLE_API_KEY)
    return col, embedder

# ====== Dataclass hit ======
@dataclass
class RetrievalHit:
    text: str
    score: float
    metadata: Dict[str, Any]
    doc_id: Optional[str]
    debug: Dict[str, Any]

# ====== Helper: fusion RRF ======
def _rrf_fuse(*ranked_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    ranked_lists: mỗi danh sách gồm dict {"_key": str, "score": float, "doc": {...}, "debug": {...}}
    """
    fused: Dict[str, Dict[str, Any]] = {}
    for rl in ranked_lists:
        for rank, item in enumerate(rl, start=1):
            key = item["_key"]
            entry = fused.setdefault(key, {"doc": item["doc"], "score": 0.0, "debug": {}})
            entry["score"] += 1.0 / (RRF_K + rank)
            entry["debug"].update(item.get("debug", {}))
    # sort desc by fused score
    return [
        {"_key": k, "doc": v["doc"], "score": v["score"], "debug": v["debug"]}
        for k, v in sorted(fused.items(), key=lambda x: x[1]["score"], reverse=True)
    ]

# ====== Retriever main ======
class HybridRetriever:
    def __init__(self):
        self.col, self.embedder = _get_clients()

    def _vector_search(self, query: str, k: int, num_candidates: int, filters: Optional[Dict[str, Any]]):
        qvec = self.embedder.encode_query(query)
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
        docs = list(self.col.aggregate(pipeline))
        out = []
        for d in docs:
            meta = d.get("meta", {})
            doc_key = meta.get("id") or meta.get("source_id") or meta.get("source") or d.get("text", "")[:60]
            out.append({
                "_key": str(doc_key),
                "score": float(d.get("score", 0.0) or 0.0),
                "doc": d,
                "debug": {"vector": float(d.get("score", 0.0) or 0.0), "index": INDEX_NAME},
            })
        return out

    def _text_search(self, query: str, k: int, filters: Optional[Dict[str, Any]]):
        """Optional lexical path using classic Mongo text index (if exists)."""
        try:
            f = {"$text": {"$search": query}}
            if filters:
                f.update(filters)
            cur = self.col.find(f, {
                "_id": 0, "text": 1, "meta": 1,
                "score": {"$meta": "textScore"}
            }).sort([("score", {"$meta": "textScore"})]).limit(int(k))
            docs = list(cur)
        except Exception:
            return []
        out = []
        for d in docs:
            meta = d.get("meta", {})
            doc_key = meta.get("id") or meta.get("source_id") or meta.get("source") or d.get("text", "")[:60]
            out.append({
                "_key": str(doc_key),
                "score": float(d.get("score", 0.0) or 0.0),
                "doc": d,
                "debug": {"text": float(d.get("score", 0.0) or 0.0)},
            })
        return out

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        num_candidates: int = 120,
    ) -> List[RetrievalHit]:
        if not (query or "").strip():
            return []

        # Vector path luôn có
        vec_ranked = self._vector_search(query, k=max(top_k, 6), num_candidates=num_candidates, filters=filters)
        # Text path nếu có text index
        txt_ranked = self._text_search(query, k=max(top_k, 6), filters=filters)

        # Nếu có cả hai → RRF; nếu chỉ một → dùng một
        ranked = _rrf_fuse(vec_ranked, txt_ranked) if txt_ranked else vec_ranked

        hits: List[RetrievalHit] = []
        for item in ranked[:top_k]:
            d = item["doc"]
            meta = d.get("meta", {})
            hits.append(
                RetrievalHit(
                    text=d.get("text", ""),
                    score=float(item.get("score", 0.0) or 0.0),
                    metadata=meta,
                    doc_id=meta.get("id"),
                    debug=item.get("debug", {}),
                )
            )
        return hits
