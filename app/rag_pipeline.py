# -*- coding: utf-8 -*-
"""
RAG pipeline (single-shot synthesis mặc định).
- retrieve_context từ scripts/04_retrieve.py
- rerank theo env (none/jina/cohere/bge)
- single-shot synthesis (1 call) → nhanh
- Ẩn nguồn để nhường token cho phần trả lời (theo yêu cầu).
"""
from typing import List, Dict, Any, Optional
import importlib.util, sys
from pathlib import Path
from rerank import rerank
from generation import generate_answer_gemini

# ==== dynamic import scripts/04_retrieve.py ====
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
- Trả lời câu hỏi: "{query}" thành 1 đoạn rõ ý (5–8 câu). Ưu tiên lập luận mạch lạc.
- Nếu có trích dẫn thơ trong NGỮ CẢNH, hãy chèn 1–2 câu thơ phù hợp vào dấu ngoặc kép.
- Chỉ dùng thông tin trong NGỮ CẢNH; nếu thiếu căn cứ, nói "chưa đủ căn cứ".
"""

def _build_single_shot_prompt(query: str, ctx_list: List[Dict[str, Any]], history_text: Optional[str]) -> str:
    merged_ctx = "\n\n---\n\n".join(c["text"] for c in ctx_list)
    if history_text:
        return f"{history_text}\n\n{SYNTH_SINGLE_TEMPLATE.format(ctx=merged_ctx, query=query)}"
    return SYNTH_SINGLE_TEMPLATE.format(ctx=merged_ctx, query=query)

def answer_question(
    query: str,
    k: int = 4,
    filters: Dict[str, Any] | None = None,
    num_candidates: int = 100,
    synthesize: str | bool = "single",
    gen_model: str = "gemini-2.0-flash",
    force_quote: bool = False,      # (để dành nếu bạn muốn ép trích dẫn nhiều hơn)
    long_answer: bool = False,
    history_text: str | None = None,
    max_tokens: Optional[int] = None,  # <— THÊM
) -> Dict[str, Any]:
    if filters is None:
        filters = {"meta.type": {"$in": ["analysis", "poem", "summary", "bio"]}}

    # 1) Retrieve rộng vừa đủ
    hits = retrieve_context(query, k=max(k, 10), filters=filters, num_candidates=num_candidates)
    if not hits:
        prompt = _build_single_shot_prompt(query, [], history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    # 2) Độ tin cậy nhanh
    avg_score = sum(h.get("score", 0.0) for h in hits) / max(1, len(hits))
    if avg_score < 0.2:
        prompt = _build_single_shot_prompt(query, [], history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    # 3) Rerank theo env rồi lấy top-k
    hits = rerank(query, hits, top_k=k)

    # 4) Synthesize
    prompt = _build_single_shot_prompt(query, hits, history_text)
    result = {"query": query, "prompt": prompt, "contexts": hits}

    if synthesize and synthesize != "mapreduce":
        ans = generate_answer_gemini(
            prompt,
            model=gen_model,
            long_answer=long_answer,
            max_tokens=max_tokens,  # <— TRUYỀN XUỐNG
        )
    elif synthesize == "mapreduce":
        # Nếu có map-reduce riêng, nhớ thêm max_tokens vào đó
        from synthesis import map_reduce_answer
        ans = map_reduce_answer(query, hits, model=gen_model, max_tokens=max_tokens)
    else:
        ans = None

    if ans:
        # KHÔNG gắn nguồn để tiết kiệm token hiển thị
        result["answer"] = ans

    return result
