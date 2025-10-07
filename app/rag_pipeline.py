# -*- coding: utf-8 -*-
"""
RAG pipeline (single-shot synthesis).
- retrieve_context từ scripts/04_retrieve.py
- rerank theo env (none/jina/cohere/bge) nếu bạn đã cài
- prompt sinh: ngắn gọn, ép dùng NGỮ CẢNH; có tuỳ chọn long_answer & force_quote
"""
from __future__ import annotations
from typing import List, Dict, Any
import importlib.util, sys
from pathlib import Path
from generation import generate_answer_gemini
from rerank import rerank

ROOT = Path(__file__).resolve().parents[1]
retr_path = ROOT / "scripts" / "04_retrieve.py"
spec = importlib.util.spec_from_file_location("retrieve_mod", retr_path)
retrieve_mod = importlib.util.module_from_spec(spec)
sys.modules["retrieve_mod"] = retrieve_mod
assert spec and spec.loader
spec.loader.exec_module(retrieve_mod)  # type: ignore
retrieve_context = retrieve_mod.retrieve_context

SYNTH_SINGLE_TEMPLATE = """[NGỮ CẢNH]
{ctx}

[HƯỚNG DẪN]
- Trả lời câu hỏi: "{query}" bằng 3–6 gạch đầu dòng, mỗi gạch ≤ 2 câu.
- Chỉ dùng thông tin trong NGỮ CẢNH; nếu thiếu căn cứ, nói 'chưa đủ căn cứ'.
- Nếu có thơ trong NGỮ CẢNH và phù hợp, hãy trích 1–3 câu thơ nguyên văn làm dẫn chứng (ghi trong dấu ngoặc kép).
- Giới hạn khoảng 120–220 từ.
{extra_rules}
"""

def _format_sources(ctx_list: List[Dict[str, Any]]) -> str:
    srcs = []
    for c in ctx_list:
        s = c.get("meta", {}).get("source")
        if s and s not in srcs:
            srcs.append(s)
    return "; ".join(srcs)

def _build_single_shot_prompt(query: str, ctx_list: List[Dict[str, Any]], force_quote=False, long_answer=False, history_text="") -> str:
    merged_ctx = "\n\n---\n\n".join(c["text"] for c in ctx_list) if ctx_list else "(trống)"
    extra = []
    if force_quote:
        extra.append("- Ưu tiên kèm 1–3 câu thơ nguyên văn làm dẫn chứng khi phù hợp.")
    if long_answer:
        extra.append("- Có thể mở bài – thân bài – kết luận ngắn gọn (dạng đoạn văn).")
        extra.append("- Duy trì lập luận mạch lạc, tránh liệt kê khô cứng.")
        extra.append("- Nếu có nhiều quan điểm, so sánh ngắn gọn.")
    if history_text:
        extra.append(f"- Lưu ý bối cảnh hội thoại trước đó:\n{history_text}")
    extra_rules = "\n".join(extra) if extra else "- "
    return SYNTH_SINGLE_TEMPLATE.format(ctx=merged_ctx, query=query, extra_rules=extra_rules)

def answer_question(
    query: str,
    k: int = 4,
    filters: Dict[str, Any] | None = None,
    num_candidates: int = 100,
    synthesize: str | bool = "single",
    gen_model: str = "gemini-2.0-flash",
    force_quote: bool = False,
    long_answer: bool = False,
    history_text: str = "",
) -> Dict[str, Any]:

    if filters is None:
        filters = {"meta.type": {"$in": ["analysis", "poem", "summary", "bio"]}}

    hits = retrieve_context(query, k=max(k, 10), filters=filters, num_candidates=num_candidates)
    if not hits:
        prompt = _build_single_shot_prompt(query, [], force_quote, long_answer, history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    avg_score = sum(h.get("score", 0.0) for h in hits) / max(1, len(hits))
    if avg_score < 0.2:
        prompt = _build_single_shot_prompt(query, [], force_quote, long_answer, history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    hits = rerank(query, hits, top_k=k)
    prompt = _build_single_shot_prompt(query, hits, force_quote, long_answer, history_text)
    result = {"query": query, "prompt": prompt, "contexts": hits}

    if synthesize and synthesize != "mapreduce":
        ans = generate_answer_gemini(prompt, model=gen_model)
    elif synthesize == "mapreduce":
        from synthesis import map_reduce_answer
        ans = map_reduce_answer(query, hits, model=gen_model)
    else:
        ans = None

    if ans:
        srcs = _format_sources(hits)
        if srcs:
            ans = f"{ans}\n\n**Nguồn:** {srcs}"
        result["answer"] = ans
    return result
