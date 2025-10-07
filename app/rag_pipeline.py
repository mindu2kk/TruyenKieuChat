# app/rag_pipeline.py
# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
import importlib.util, sys
from pathlib import Path
from rerank import rerank
from generation import generate_answer_gemini

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
- Trả lời câu hỏi: "{query}".
- {style_line}
- Chỉ dùng thông tin trong NGỮ CẢNH; nếu thiếu căn cứ, nói "chưa đủ căn cứ".
- {quote_line}
{history_block}
"""

def _style_line(long_answer: bool) -> str:
    if long_answer:
        return "Viết mạch lạc theo bố cục nghị luận (mở–thân–kết), 3–5 đoạn, có chuyển ý."
    return "Trình bày 3–6 gạch đầu dòng, mỗi gạch ≤ 2 câu, 120–160 từ."

def _quote_line(force_quote: bool) -> str:
    if force_quote:
        return "Nếu phù hợp, chèn 1–2 câu thơ Kiều NGUYÊN VĂN từ NGỮ CẢNH làm dẫn chứng (đừng bịa)."
    return "Không cần dẫn chứng bắt buộc."

def _history_block(history_text: str) -> str:
    if not history_text:
        return ""
    return f"""[NGỮ CẢNH HỘI THOẠI TRƯỚC]
{history_text}
"""

def _format_sources(ctx_list: List[Dict[str, Any]]) -> str:
    srcs = []
    for c in ctx_list:
        s = c.get("meta", {}).get("source")
        if s and s not in srcs:
            srcs.append(s)
    return "; ".join(srcs)

def _build_single_shot_prompt(query: str, ctx_list: List[Dict[str, Any]],
                              long_answer: bool, force_quote: bool,
                              history_text: str) -> str:
    merged_ctx = "\n\n---\n\n".join(c["text"] for c in ctx_list)
    return SYNTH_SINGLE_TEMPLATE.format(
        ctx=merged_ctx or "(trống)",
        query=query,
        style_line=_style_line(long_answer),
        quote_line=_quote_line(force_quote),
        history_block=_history_block(history_text)
    )

def answer_question(
    query: str,
    k: int = 4,
    filters: Optional[Dict[str, Any]] = None,
    num_candidates: int = 100,
    synthesize: str | bool = "single",
    gen_model: str = "gemini-2.0-flash",
    force_quote: bool = False,
    long_answer: bool = False,
    history_text: str = "",
) -> Dict[str, Any]:

    if filters is None:
        filters = {"meta.type": {"$in": ["analysis", "poem", "summary", "bio"]}}

    # 1) retrieve
    hits = retrieve_context(query, k=max(k, 10), filters=filters, num_candidates=num_candidates)
    if not hits:
        prompt = _build_single_shot_prompt(query, [], long_answer, force_quote, history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    # 2) sanity score
    avg_score = sum(h.get("score", 0.0) for h in hits) / max(1, len(hits))
    if avg_score < 0.2:
        prompt = _build_single_shot_prompt(query, [], long_answer, force_quote, history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    # 3) rerank + top-k
    hits = rerank(query, hits, top_k=k)

    # 4) synthesize
    prompt = _build_single_shot_prompt(query, hits, long_answer, force_quote, history_text)
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
