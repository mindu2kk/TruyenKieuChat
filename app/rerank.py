# app/rerank.py
# -*- coding: utf-8 -*-
"""
Reranker linh hoạt theo biến môi trường:
  RERANKER=none        (mặc định, nhanh và an toàn)
  RERANKER=cross_encoder | ce
  RERANKER=jina
  RERANKER=cohere
  RERANKER=bge
Tự động fallback về rerank lexical nếu có lỗi hoặc thiếu model/API.
"""
from __future__ import annotations
import os, re
from typing import Any, Dict, List, Sequence

# Mặc định an toàn: none
MODE = (os.getenv("RERANKER") or "none").strip().lower()

_tok = None
_mdl = None
_bge = None
_cross_encoder = None

_WORD = re.compile(r"[\wÀ-ỹ']+")

def _tokens(s: str) -> List[str]:
    return _WORD.findall((s or "").lower())

def _safe_topk(v: Any, default: int = 10) -> int:
    try:
        k = int(v)
    except Exception:
        k = default
    if k <= 0:
        k = 1
    return k

def _lexical_fallback(query: str, hits: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Rerank nhẹ theo overlap từ vựng + trọng số điểm gốc."""
    qset = set(_tokens(query))
    scored: List[Dict[str, Any]] = []
    for h in hits:
        item = dict(h)
        text = str(item.get("text", "") or "")
        base = float(item.get("score", 0.0) or 0.0)
        toks = _tokens(text)
        denom = max(3, len(toks))
        overlap = sum(1 for t in toks if t in qset)
        lex = overlap / denom
        item["re_score"] = 0.6 * lex + 0.4 * base
        scored.append(item)
    scored.sort(key=lambda x: float(x.get("re_score", 0.0) or 0.0), reverse=True)
    return scored[:top_k]

def _ensure_jina():
    global _tok, _mdl
    if _tok is None or _mdl is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _tok = AutoTokenizer.from_pretrained("jinaai/jina-reranker-v2-base-multilingual")
        _mdl = AutoModelForSequenceClassification.from_pretrained("jinaai/jina-reranker-v2-base-multilingual")
    return _tok, _mdl

def _ensure_bge():
    global _bge
    if _bge is None:
        from FlagEmbedding import FlagReranker
        _bge = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _bge

def rerank(query: str, hits: Sequence[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    # Chuẩn hoá đầu vào
    if not hits:
        return list(hits)
    hits = list(hits)
    k = _safe_topk(top_k, default=10)

    # MODE none: trả theo thứ hạng hiện có (hoặc lexical nhẹ để ổn định)
    if MODE in {"none", "off", "disabled"}:
        # ưu tiên dùng điểm có sẵn nếu có
        try:
            tmp = sorted(hits, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            return tmp[:k]
        except Exception:
            return _lexical_fallback(query, hits, k)

    # Cross-Encoder
    if MODE in {"cross_encoder", "cross-encoder", "ce"}:
        try:
            from sentence_transformers import CrossEncoder
            global _cross_encoder
            if _cross_encoder is None:
                ckpt = os.getenv("CROSS_ENCODER_CKPT", "keepitreal/vietnamese-cross-encoder")
                try:
                    _cross_encoder = CrossEncoder(ckpt, max_length=384)
                except Exception:
                    _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=384)
            pairs = [(query, str(h.get("text", "") or "")) for h in hits]
            scores = _cross_encoder.predict(pairs)
            out = []
            for h, s in zip(hits, scores):
                item = dict(h)
                item["re_score"] = float(s)
                out.append(item)
            out.sort(key=lambda x: float(x.get("re_score", 0.0) or 0.0), reverse=True)
            return out[:k]
        except Exception:
            return _lexical_fallback(query, hits, k)

    # Cohere API
    if MODE == "cohere":
        try:
            import cohere
            api_key = os.getenv("COHERE_API_KEY", "")
            if not api_key:
                raise RuntimeError("Thiếu COHERE_API_KEY")
            co = cohere.Client(api_key)
            resp = co.rerank(model="rerank-multilingual-v3.0", query=query,
                             documents=[str(h.get("text", "") or "") for h in hits], top_n=k)
            # resp.results chứa index + relevance_score
            idxs = [r.index for r in resp.results]
            out = []
            for r in resp.results:
                i = int(r.index)
                item = dict(hits[i])
                item["re_score"] = float(getattr(r, "relevance_score", 0.0) or 0.0)
                out.append(item)
            out.sort(key=lambda x: float(x.get("re_score", 0.0) or 0.0), reverse=True)
            return out[:k] if out else [hits[i] for i in idxs][:k]
        except Exception:
            return _lexical_fallback(query, hits, k)

    # Jina (Transformers)
    if MODE == "jina":
        try:
            import torch
            tok, mdl = _ensure_jina()
            with torch.inference_mode():
                batch = tok(
                    [query] * len(hits),
                    [str(h.get("text", "") or "") for h in hits],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                logits = mdl(**batch).logits.squeeze(-1)
                scores = logits.tolist() if hasattr(logits, "tolist") else list(logits)
            out = []
            for h, s in zip(hits, scores):
                item = dict(h)
                item["re_score"] = float(s)
                out.append(item)
            out.sort(key=lambda x: float(x.get("re_score", 0.0) or 0.0), reverse=True)
            return out[:k]
        except Exception:
            return _lexical_fallback(query, hits, k)

    # BGE FlagEmbedding
    if MODE == "bge":
        try:
            rr = _ensure_bge()
            pairs = [(query, str(h.get("text", "") or "")) for h in hits]
            scores = rr.compute_score(pairs, normalize=True)
            out = []
            for h, s in zip(hits, scores):
                item = dict(h)
                item["re_score"] = float(s)
                out.append(item)
            out.sort(key=lambda x: float(x.get("re_score", 0.0) or 0.0), reverse=True)
            return out[:k]
        except Exception:
            return _lexical_fallback(query, hits, k)

    # MODE không nhận diện -> fallback
    return _lexical_fallback(query, hits, k)
