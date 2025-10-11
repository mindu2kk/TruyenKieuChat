# -*- coding: utf-8 -*-
from typing import Dict, Any, Iterable, List, Optional, Sequence, Tuple
import unicodedata

try:  # pragma: no cover - flexible import paths
    from .rerank import rerank
    from .generation import generate_answer_gemini
    from .prompt_engineering import (
        DEFAULT_LONG_TOKEN_BUDGET,
        DEFAULT_SHORT_TOKEN_BUDGET,
        build_rag_synthesis_prompt,
    )
    from .hybrid_retriever import HybridRetriever, RetrievalHit
except ImportError:  # pragma: no cover
    from rerank import rerank  # type: ignore
    from generation import generate_answer_gemini  # type: ignore
    from prompt_engineering import (  # type: ignore
        DEFAULT_LONG_TOKEN_BUDGET,
        DEFAULT_SHORT_TOKEN_BUDGET,
        build_rag_synthesis_prompt,
    )
    from hybrid_retriever import HybridRetriever, RetrievalHit  # type: ignore


# ====== alias/biến thể tên nhân vật để tăng recall khi tạo query variants ======
_CHARACTER_VARIANTS: Dict[str, Sequence[str]] = {
    "thúy kiều": ("thúy kiều", "thuý kiều", "thuy kieu", "kiều", "thụy kiều"),
    "thúy vân": ("thúy vân", "thuý vân", "thuy van", "vân"),
    "kim trọng": ("kim trọng", "kim trong", "kim-trọng"),
    "từ hải": ("từ hải", "tu hai", "từ-hải"),
    "hoạn thư": ("hoạn thư", "hoan thu"),
    "giác duyên": ("giác duyên", "giac duyen"),
}


# ====== helpers chuẩn hoá văn bản ======
def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalise_space(text: str) -> str:
    return " ".join((text or "").split())


def _build_query_variants(query: str) -> List[str]:
    query = _normalise_space(query)
    lowered = query.lower()
    variants: List[str] = []
    seen: set[str] = set()

    def _push(candidate: str) -> None:
        cand_norm = _normalise_space(candidate)
        if not cand_norm:
            return
        key = cand_norm.lower()
        if key in seen:
            return
        seen.add(key)
        variants.append(cand_norm)

    _push(query)
    ascii_query = _strip_accents(query)
    if ascii_query.lower() != lowered:
        _push(ascii_query)

    ascii_lower = ascii_query.lower()
    for canonical, alias_list in _CHARACTER_VARIANTS.items():
        canonical_ascii = _strip_accents(canonical)
        alias_hits = any(alias in lowered for alias in alias_list) or canonical_ascii in ascii_lower
        if not alias_hits:
            continue

        _push(canonical)
        _push(canonical_ascii)
        for alias in alias_list:
            _push(alias)
            alias_ascii = _strip_accents(alias)
            _push(alias_ascii)

        base_name = canonical.split()[0]
        _push(f"nhân vật {canonical}")
        _push(f"tiểu sử {canonical}")
        _push(f"{canonical} trong truyện kiều")
        _push(f"vai trò của {canonical}")
        _push(f"{base_name} của truyện kiều")

    return variants


# ====== hợp nhất & khử trùng lặp hit ======
def _dedupe_hits(hits: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    for hit in hits:
        meta = hit.get("meta") or hit.get("metadata") or {}
        key_parts: List[str] = []

        # ưu tiên các khoá ổn định
        doc_id = hit.get("doc_id")
        if doc_id:
            key_parts.append(str(doc_id))

        if isinstance(meta, dict):
            for attr in ("id", "_id", "chunk_id", "source_id"):
                value = meta.get(attr)
                if value:
                    key_parts.append(str(value))
                    break

        snippet = hit.get("text") or ""
        if snippet and not key_parts:
            key_parts.append(_strip_accents(snippet)[:160])

        if not key_parts:
            key_parts.append(str(len(deduped)))

        key = "::".join(key_parts)
        if key in seen_keys:
            continue

        seen_keys.add(key)
        deduped.append(hit)

    return deduped


def _as_hit_dict(hit: Dict[str, Any] | RetrievalHit) -> Dict[str, Any]:
    if isinstance(hit, RetrievalHit):
        return {
            "text": hit.text,
            "score": hit.score,
            "meta": dict(hit.metadata or {}),
            "metadata": dict(hit.metadata or {}),
            "doc_id": hit.doc_id,
            "debug": dict(hit.debug or {}),
        }
    return dict(hit)


def _annotate_hits(
    hits: Sequence[Dict[str, Any] | RetrievalHit],
    *,
    variant: str,
    variant_index: int,
    relaxed_filters: bool,
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    base_penalty = 0.06 * variant_index + (0.12 if relaxed_filters else 0.0)

    for local_rank, hit in enumerate(hits):
        new_hit = _as_hit_dict(hit)
        meta = dict(new_hit.get("meta") or new_hit.get("metadata") or {})
        new_hit["meta"] = meta
        new_hit["metadata"] = meta
        new_hit["source_query"] = variant
        new_hit["relaxed_filters"] = relaxed_filters
        new_hit["variant_index"] = variant_index
        raw_score = float(new_hit.get("score", 0.0) or 0.0)
        new_hit["_raw_score"] = raw_score
        score = raw_score - base_penalty - 0.01 * local_rank
        new_hit["score"] = score
        annotated.append(new_hit)

    return annotated


# ====== boost nhẹ cho thơ khi cần close-reading ======
def _boost_poem_hits(collected: List[Dict[str, Any]], bonus: float = 0.5) -> None:
    for h in collected:
        m = h.get("meta", {})
        if isinstance(m, dict) and m.get("type") == "poem":
            try:
                h["score"] = float(h.get("score", 0.0)) + bonus
            except Exception:
                pass


# ====== dựng SOURCES & EVIDENCE ======
def _format_source_label(meta: Dict[str, Any]) -> str:
    src = meta.get("source") or meta.get("src") or meta.get("title") or "unknown"
    # thơ ưu tiên line_start/line_end
    if meta.get("type") == "poem" and meta.get("line_start") and meta.get("line_end"):
        return f"{src}:L{meta['line_start']}-{meta['line_end']}"
    # văn xuôi có offset
    if meta.get("char_start") is not None and meta.get("char_end") is not None:
        return f"{src}:{meta['char_start']}-{meta['char_end']}"
    # fallback chunk index
    if meta.get("chunk_index") is not None:
        return f"{src}#chunk{meta['chunk_index']}"
    return str(src)


def _build_sources_and_evidence(contexts: List[Dict[str, Any]], top_evidence: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    ev = []
    for c in contexts[: max(1, top_evidence)]:
        ev.append({"text": c.get("text", ""), "meta": dict(c.get("meta", {}))})
    # unique sources theo label
    labels: List[str] = []
    seen: set[str] = set()
    for c in contexts:
        lab = _format_source_label(dict(c.get("meta", {})))
        if lab not in seen:
            seen.add(lab)
            labels.append(lab)
    return labels, ev


# ====== Hybrid retriever instance ======
_HYBRID_RETRIEVER = HybridRetriever()


def answer_question(
    query: str,
    k: int = 5,
    filters: Dict[str, Any] | None = None,
    num_candidates: int = 120,
    synthesize: str | bool = "single",
    gen_model: str = "gemini-2.0-flash",
    force_quote: bool = True,
    long_answer: bool = False,
    history_text: str | None = None,
    max_tokens: Optional[int] = None,
    *,
    # ==== bổ sung theo P0 ====
    prefer_poem_source: bool = False,
    top_evidence: int = 6,
    essay_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trả về tối thiểu:
      {
        "query": str,
        "prompt": str,           # nếu không synthesize
        "contexts": list,        # danh sách hit đã rerank
        "answer": str,           # nếu synthesize thành công
        "sources": [label...],   # ví dụ: "poem/poem.txt:L123-126"
        "evidence": [ {"text": "...", "meta": {...}}, ... ],
        "generation_error": "...",   # nếu model lỗi
      }
    """

    # -------- defaults --------
    if filters is None:
        filters = {"meta.type": {"$in": ["analysis", "poem", "summary", "bio"]}}

    if max_tokens is None:
        max_tokens = DEFAULT_LONG_TOKEN_BUDGET if long_answer else DEFAULT_SHORT_TOKEN_BUDGET

    # -------- variants for recall --------
    query_variants = _build_query_variants(query)
    if not query_variants:
        query_variants = [_normalise_space(query) or "Truyện Kiều"]

    collected: List[Dict[str, Any]] = []

    # -------- local collector wrapper --------
    def _maybe_collect(
        variant: str,
        variant_index: int,
        *,
        relaxed: bool,
        limit: int,
        candidates: int,
        active_filters: Optional[Dict[str, Any]],
    ) -> None:
        try:
            hits_local = _HYBRID_RETRIEVER.search(
                variant,
                top_k=limit,
                filters=active_filters,
                num_candidates=candidates,
            )
        except Exception:
            return
        if not hits_local:
            return
        collected.extend(
            _annotate_hits(
                hits_local,
                variant=variant,
                variant_index=variant_index,
                relaxed_filters=relaxed,
            )
        )

    # 1) Primary
    _maybe_collect(
        query_variants[0],
        0,
        relaxed=False,
        limit=max(k, 10),
        candidates=num_candidates,
        active_filters=filters,
    )

    # 2) Variants
    if len(collected) < max(2, k):
        for idx, variant in enumerate(query_variants[1:], start=1):
            _maybe_collect(
                variant,
                idx,
                relaxed=False,
                limit=max(k, 10),
                candidates=num_candidates,
                active_filters=filters,
            )
            if len(collected) >= max(3 * k, 25):
                break

    # 3) Relax filters
    if len(collected) < k and filters:
        relaxed_candidates = max(num_candidates, 180)
        for idx, variant in enumerate(query_variants):
            _maybe_collect(
                variant,
                idx,
                relaxed=True,
                limit=max(k, 12),
                candidates=relaxed_candidates,
                active_filters=None,
            )
            if len(collected) >= max(4 * k, 32):
                break

    if not collected:
        prompt = build_rag_synthesis_prompt(query, [], history_text=history_text, long_answer=long_answer, essay_mode=essay_mode)
        return {"query": query, "prompt": prompt, "contexts": []}

    # ưu tiên thơ nếu cần
    if prefer_poem_source:
        _boost_poem_hits(collected, bonus=0.5)

    # dedupe + sort
    collected = _dedupe_hits(collected)
    collected.sort(key=lambda item: item.get("score", 0.0), reverse=True)

    # sanity-check chất lượng tín hiệu
    top_for_avg = collected[: max(1, k)]
    avg_score = sum(h.get("_raw_score", 0.0) for h in top_for_avg) / max(1, len(top_for_avg))
    if avg_score < 0.06:
        prompt = build_rag_synthesis_prompt(query, [], history_text=history_text, long_answer=long_answer, essay_mode=essay_mode)
        return {"query": query, "prompt": prompt, "contexts": []}

    # rerank sâu
    rerank_depth = min(len(collected), max(k * 2, 12))
    reranked = rerank(query, collected, top_k=rerank_depth)
    contexts = reranked[: max(k, 6 if long_answer else k)]

    # dựng prompt synthesis
    prompt = build_rag_synthesis_prompt(query, contexts, history_text=history_text, long_answer=long_answer, essay_mode=essay_mode)

    # dựng sources/evidence (dùng cho cả synthesize lẫn UI)
    sources, evidence = _build_sources_and_evidence(contexts, top_evidence=top_evidence)

    out: Dict[str, Any] = {
        "query": query,
        "prompt": prompt,
        "contexts": contexts,
        "sources": sources,
        "evidence": evidence,
    }

    # synthesize nếu được yêu cầu
    if synthesize and synthesize != "mapreduce":
        if force_quote:
            prompt += "\n\n[LƯU Ý] Nếu có câu thơ phù hợp, hãy trích đúng nguyên văn 1–2 câu trong ngoặc kép."
        try:
            ans = generate_answer_gemini(
                prompt,
                model=gen_model,
                long_answer=long_answer,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # bắt mọi lỗi SDK
            out["generation_error"] = str(exc)
            return out
        if ans:
            out["answer"] = ans

    return out
