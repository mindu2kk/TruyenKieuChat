# -*- coding: utf-8 -*-
"""Hybrid retriever combining lexical, dense and pseudo-ColBERT scoring."""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency in some deployments
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import numpy as _np

    NDArray = _np.ndarray[Any, Any]
else:  # pragma: no cover - only needed when numpy missing at runtime
    NDArray = Any


class _SimpleBM25:
    """Fallback lexical scorer when rank_bm25 isn't available."""

    def __init__(self, tokenised_docs: List[List[str]]) -> None:
        self._docs = tokenised_docs

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        if not query_tokens:
            return [0.0] * len(self._docs)

        query_set = set(query_tokens)
        scores: List[float] = []
        for doc_tokens in self._docs:
            if not doc_tokens:
                scores.append(0.0)
                continue
            overlap = sum(1 for token in doc_tokens if token in query_set)
            scores.append(overlap / len(doc_tokens))
        return scores

try:  # pragma: no cover - allow usage both as package and script
    from .corpus_loader import CorpusDocument, load_corpus
except ImportError:  # pragma: no cover
    from corpus_loader import CorpusDocument, load_corpus


def _prepare_hf_cache() -> Path:
    """Ensure Hugging Face models download into a writable directory."""

    repo_root = Path(__file__).resolve().parent / ".." / ".hf_cache"
    home_root = Path.home() / ".cache" / "huggingface"

    candidates: List[Path] = []
    env_home = os.environ.get("HF_HOME")
    if env_home:
        candidates.append(Path(env_home))
    candidates.extend([repo_root, home_root])

    cache_root: Optional[Path] = None
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            continue
        if os.access(path, os.W_OK):
            cache_root = path
            break

    if cache_root is None:
        # Final fallback: use a temporary directory inside the current working tree.
        cache_root = Path.cwd() / ".hf_cache"
        cache_root.mkdir(parents=True, exist_ok=True)

    # Align other libraries that respect their own environment flags.
    os.environ.setdefault("HF_HOME", str(cache_root))

    transformers_cache = cache_root / "transformers"
    sentence_cache = cache_root / "sentence-transformers"
    transformers_cache.mkdir(parents=True, exist_ok=True)
    sentence_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(sentence_cache))

    return cache_root


_HF_CACHE_DIR = _prepare_hf_cache()


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[\wÀ-ỹ']+", text.lower())
    return tokens


@dataclass
class RetrievalHit:
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    debug: Dict[str, float]


class HybridRetriever:
    def __init__(
        self,
        *,
        dense_model: str | None = None,
        colbert_model: str | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.dense_model_name = dense_model or "keepitreal/vietnamese-sbert"
        self.colbert_model_name = colbert_model or self.dense_model_name
        self.rrf_k = rrf_k

        self._corpus: List[CorpusDocument] | None = None
        self._bm25 = None
        self._dense_model = None
        self._dense_embeddings: np.ndarray | None = None
        self._colbert_model = None
        self._colbert_embeddings: np.ndarray | None = None
        self._colbert_doc_index: List[int] | None = None
        self._colbert_segments: List[str] | None = None

    # ------------------------------------------------------------------
    # lazy loaders
    def _ensure_corpus(self) -> List[CorpusDocument]:
        if self._corpus is None:
            self._corpus = load_corpus()
        return self._corpus

    def _ensure_bm25(self):
        if self._bm25 is None:
            corpus = self._ensure_corpus()
            tokenised = [_tokenize(doc.text) for doc in corpus]
            try:
                from rank_bm25 import BM25Okapi

                self._bm25 = BM25Okapi(tokenised)
            except ImportError:  # pragma: no cover - optional dependency
                self._bm25 = _SimpleBM25(tokenised)
        return self._bm25

    def _ensure_dense_model(self):
        if np is None:
            raise RuntimeError("numpy chưa được cài đặt")
        if self._dense_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("sentence-transformers chưa được cài đặt") from exc

            self._dense_model = SentenceTransformer(
                self.dense_model_name,
                cache_folder=str(_HF_CACHE_DIR),
            )
        return self._dense_model

    def _ensure_dense_embeddings(self) -> NDArray:
        if np is None:
            raise RuntimeError("numpy chưa được cài đặt")
        if self._dense_embeddings is None:
            model = self._ensure_dense_model()
            corpus = self._ensure_corpus()
            self._dense_embeddings = model.encode(
                [doc.text for doc in corpus],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return self._dense_embeddings

    def _ensure_colbert_model(self):
        if np is None:
            raise RuntimeError("numpy chưa được cài đặt")
        if self._colbert_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("sentence-transformers chưa được cài đặt") from exc

            self._colbert_model = SentenceTransformer(
                self.colbert_model_name,
                cache_folder=str(_HF_CACHE_DIR),
            )
        return self._colbert_model

    def _ensure_colbert_index(self) -> Tuple[NDArray, List[int], List[str]]:
        if np is None:
            raise RuntimeError("numpy chưa được cài đặt")
        if self._colbert_embeddings is None or self._colbert_doc_index is None or self._colbert_segments is None:
            model = self._ensure_colbert_model()
            corpus = self._ensure_corpus()
            segments: List[str] = []
            mapping: List[int] = []
            splitter = re.compile(r"[\n.!?;]+")
            for idx, doc in enumerate(corpus):
                raw_segments = [seg.strip() for seg in splitter.split(doc.text) if seg.strip()]
                if not raw_segments:
                    raw_segments = [doc.text.strip()]
                for seg in raw_segments:
                    segments.append(seg)
                    mapping.append(idx)
            if not segments:
                segments.append("Truyện Kiều")
                mapping.append(0)
            self._colbert_embeddings = model.encode(
                segments,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._colbert_doc_index = mapping
            self._colbert_segments = segments
        return self._colbert_embeddings, self._colbert_doc_index, self._colbert_segments

    # ------------------------------------------------------------------
    # retrieval strategies
    def _search_bm25(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        bm25 = self._ensure_bm25()
        scores = bm25.get_scores(_tokenize(query))
        if np is not None:
            ranking = np.argsort(scores)[::-1][:top_k]
        else:
            ranking = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in ranking if scores[idx] > 0]

    def _search_dense(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        embeddings = self._ensure_dense_embeddings()
        model = self._ensure_dense_model()
        qvec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        if np is None:
            raise RuntimeError("numpy chưa được cài đặt")
        sims = np.dot(embeddings, qvec)
        ranking = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in ranking if sims[idx] > 0]

    def _search_colbert(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        emb_matrix, mapping, _segments = self._ensure_colbert_index()
        model = self._ensure_colbert_model()
        qvec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        if np is None:
            raise RuntimeError("numpy chưa được cài đặt")
        sims = emb_matrix @ qvec
        best: Dict[int, float] = {}
        for score, doc_idx in zip(sims, mapping):
            score = float(score)
            if doc_idx not in best or score > best[doc_idx]:
                best[doc_idx] = score
        ranking = sorted(best.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return ranking

    # ------------------------------------------------------------------
    def _rrf(self, ranked_lists: Sequence[Tuple[str, List[Tuple[int, float]]]]) -> Dict[str, Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}
        corpus = self._ensure_corpus()
        for label, results in ranked_lists:
            for rank, (doc_idx, raw_score) in enumerate(results, start=1):
                if doc_idx < 0 or doc_idx >= len(corpus):
                    continue
                doc = corpus[doc_idx]
                key = doc.doc_id
                entry = fused.setdefault(
                    key,
                    {
                        "doc": doc,
                        "score": 0.0,
                        "debug": {},
                    },
                )
                entry["score"] += 1.0 / (self.rrf_k + rank)
                entry["debug"][label] = float(raw_score)
        return fused

    @staticmethod
    def _allow_document(doc: CorpusDocument, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        allowed_types: Optional[Sequence[str]] = None
        meta_filters = filters.get("meta.type") if isinstance(filters, dict) else None
        if isinstance(meta_filters, dict) and "$in" in meta_filters:
            val = meta_filters.get("$in")
            if isinstance(val, Sequence):
                allowed_types = [str(v) for v in val]
        if allowed_types:
            return str(doc.metadata.get("type")) in allowed_types
        return True

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        num_candidates: int = 60,
    ) -> List[RetrievalHit]:
        query = (query or "").strip()
        if not query:
            return []

        corpus = self._ensure_corpus()
        if not corpus:
            return []

        k = max(top_k, 1)
        candidates = max(num_candidates, k)

        ranked_lists: List[Tuple[str, List[Tuple[int, float]]]] = []
        ranked_lists.append(("bm25", self._search_bm25(query, candidates)))

        try:
            ranked_lists.append(("dense", self._search_dense(query, candidates)))
        except RuntimeError:
            pass

        try:
            ranked_lists.append(("colbert", self._search_colbert(query, candidates)))
        except RuntimeError:
            pass

        fused = self._rrf(ranked_lists)
        hits: List[RetrievalHit] = []
        for entry in sorted(fused.values(), key=lambda item: item["score"], reverse=True):
            doc = entry["doc"]
            if not self._allow_document(doc, filters):
                continue
            hits.append(
                RetrievalHit(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=float(entry["score"]),
                    metadata=dict(doc.metadata),
                    debug=entry["debug"],
                )
            )
            if len(hits) >= k:
                break

        return hits


__all__ = ["HybridRetriever", "RetrievalHit"]