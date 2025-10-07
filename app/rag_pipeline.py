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

[CHỈ DẪN]
Bạn là người chấm bài nghị luận văn học, trả lời câu hỏi: "{query}".
- Viết 1 đoạn 6–10 câu có mở–thân–kết, mạch lạc.
- CHỈ dùng thông tin trong NGỮ CẢNH; nếu thiếu căn cứ, nói "chưa đủ căn cứ".
- Nếu NGỮ CẢNH có câu thơ phù hợp: trích 1–2 câu nguyên văn trong ngoặc kép (không kèm nguồn).
- Tránh gạch đầu dòng; diễn đạt mềm mại, có liên kết luận điểm–luận cứ–dẫn chứng–bình luận.
"""

def _merge_ctx(ctx_list: List[Dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(c["text"] for c in ctx_list)

def _build_prompt(query: str, ctx_list: List[Dict[str, Any]], history_text: Optional[str]) -> str:
    merged = _merge_ctx(ctx_list)
    core = SYNTH_SINGLE_TEMPLATE.format(ctx=merged, query=query)
    if history_text:
        return f"[HỘI THOẠI GẦN NHẤT]\n{history_text}\n\n{core}"
    return core

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
) -> Dict[str, Any]:

    if filters is None:
        filters = {"meta.type": {"$in": ["analysis", "poem", "summary", "bio"]}}

    hits = retrieve_context(query, k=max(k, 10), filters=filters, num_candidates=num_candidates)
    if not hits:
        prompt = _build_prompt(query, [], history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    avg_score = sum(h.get("score", 0.0) for h in hits) / max(1, len(hits))
    if avg_score < 0.2:
        prompt = _build_prompt(query, [], history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    hits = rerank(query, hits, top_k=k)
    prompt = _build_prompt(query, hits, history_text)

    if synthesize and synthesize != "mapreduce":
        if force_quote:
            prompt += "\n\n[LƯU Ý] Nếu có câu thơ phù hợp, hãy trích 1–2 câu trong ngoặc kép."
        ans = generate_answer_gemini(
            prompt,
            model=gen_model,
            long_answer=long_answer,
            max_tokens=max_tokens,
        )
    else:
        ans = None

    out = {"query": query, "prompt": prompt, "contexts": hits}
    if ans:
        out["answer"] = ans
    return out
