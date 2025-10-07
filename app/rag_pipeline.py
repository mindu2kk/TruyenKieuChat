# -*- coding: utf-8 -*-
"""
RAG single-shot có Prompt Engineering:
- Ưu tiên trích NGẮN 1–2 câu thơ (nếu có trong ngữ cảnh) để làm chứng.
- Không chèn “Nguồn:” để tiết kiệm token hiển thị.
"""
from typing import List, Dict, Any, Optional
import importlib.util, sys
from pathlib import Path
from rerank import rerank
from generation import generate_answer_gemini

# dynamic import retriever
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
Bạn là người chấm bài nghị luận văn học.
- Trả lời câu hỏi: "{query}" thành 1 đoạn mạch lạc (5–10 câu), có mở–thân–kết, lập luận rõ.
- CHỈ dùng thông tin trong NGỮ CẢNH; nếu thiếu căn cứ, nói "chưa đủ căn cứ".
- NẾU trong NGỮ CẢNH có thơ phù hợp: hãy CHÈN 1–2 câu thơ nguyên văn trong dấu ngoặc kép (không thêm nguồn).
- Tránh liệt kê máy móc. Ưu tiên nêu luận điểm → dẫn chứng (thơ) → phân tích → kết.
"""

def _merge_ctx(ctx_list: List[Dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(c["text"] for c in ctx_list)

def _build_single_shot_prompt(query: str, ctx_list: List[Dict[str, Any]], history_text: Optional[str]) -> str:
    merged_ctx = _merge_ctx(ctx_list)
    core = SYNTH_SINGLE_TEMPLATE.format(ctx=merged_ctx, query=query)
    if history_text:
        return f"{history_text}\n\n{core}"
    return core

def answer_question(
    query: str,
    k: int = 4,
    filters: Dict[str, Any] | None = None,
    num_candidates: int = 100,
    synthesize: str | bool = "single",
    gen_model: str = "gemini-2.0-flash",
    force_quote: bool = True,
    long_answer: bool = False,
    history_text: str | None = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    if filters is None:
        # mở rộng cho cả 'poem' để dễ lấy câu thơ làm chứng
        filters = {"meta.type": {"$in": ["analysis", "poem", "summary", "bio"]}}

    # 1) retrieve
    hits = retrieve_context(query, k=max(k, 10), filters=filters, num_candidates=num_candidates)
    if not hits:
        prompt = _build_single_shot_prompt(query, [], history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    # 2) quick trust
    avg_score = sum(h.get("score", 0.0) for h in hits) / max(1, len(hits))
    if avg_score < 0.2:
        prompt = _build_single_shot_prompt(query, [], history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    # 3) rerank → top-k
    hits = rerank(query, hits, top_k=k)

    # 4) synthesize
    prompt = _build_single_shot_prompt(query, hits, history_text)
    result = {"query": query, "prompt": prompt, "contexts": hits}

    if synthesize and synthesize != "mapreduce":
        if force_quote:
            # nhấn mạnh lần nữa để model thật sự chèn thơ nếu có
            prompt += "\n\n[LƯU Ý] Nếu trong NGỮ CẢNH có câu thơ phù hợp, bắt buộc trích 1–2 câu trong ngoặc kép."
        ans = generate_answer_gemini(
            prompt,
            model=gen_model,
            long_answer=long_answer,
            max_tokens=max_tokens,
        )
    elif synthesize == "mapreduce":
        from synthesis import map_reduce_answer
        ans = map_reduce_answer(query, hits, model=gen_model, max_tokens=max_tokens)
    else:
        ans = None

    if ans:
        # KHÔNG gắn nguồn để dồn token cho nội dung
        result["answer"] = ans

    return result
