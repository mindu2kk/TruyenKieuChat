# app/rag_pipeline.py
# -*- coding: utf-8 -*-
"""
RAG pipeline (single-shot synthesis).
- retrieve_context từ scripts/04_retrieve.py
- rerank theo env (none/jina/cohere/bge)
- prompt được tối ưu cho Văn (Việt), có yêu cầu dẫn chứng thơ khi phù hợp.
- KHÔNG tự động chèn 'Nguồn:' để dành token cho câu trả lời.
"""

from __future__ import annotations
from typing import List, Dict, Any
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

# ---- Prompt template: bản ngắn và bản luận văn
SYNTH_SINGLE_TEMPLATE = """[SYSTEM]
Bạn là trợ lý học Văn (tiếng Việt). Trả lời mạch lạc, có mở–thân–kết ngắn, dùng thuật ngữ ngữ văn vừa phải, tránh liệt kê khô cứng.

[NGỮ CẢNH]
{ctx}

[CHỈ DẪN]
- Trả lời cho câu hỏi: "{query}".
- Chỉ dùng thông tin trong NGỮ CẢNH và hiểu biết văn học cơ bản; nếu thật sự thiếu căn cứ, nói ngắn gọn 'chưa đủ căn cứ'.
- Nếu câu hỏi cần minh hoạ: trích 1–3 câu thơ NGẮN gọn, đặt blockquote (> ...) và giải thích ngay sau trích dẫn.
- Không thêm mục 'Nguồn' hoặc danh sách link.
- Độ dài: 160–240 từ, tối đa 12 câu.
"""

SYNTH_LONG_TEMPLATE = """[SYSTEM]
Bạn là giảng viên Ngữ văn viết đáp án nghị luận ngắn (tiếng Việt), lập luận mạch lạc, có chuyển ý tự nhiên, ưu tiên dẫn chứng thơ chính xác và phân tích nghệ thuật (tả cảnh ngụ tình, ước lệ, so sánh, điển cố...).

[NGỮ CẢNH]
{ctx}

[CHỈ DẪN]
- Trả lời cho câu hỏi: "{query}".
- Dẫn chứng thơ: trích 1–3 câu phù hợp (nếu chắc chắn), đặt blockquote (> ...); giải thích nghệ thuật và ý nghĩa.
- Không thêm mục 'Nguồn'.
- Bố cục: 1) Đặt vấn đề (1–2 câu)  2) Bình giảng/so sánh (3–6 câu)  3) Kết luận (1–2 câu).
- Độ dài mục tiêu: 280–420 từ.
"""

def _merge_ctx(ctx_list: List[Dict[str, Any]]) -> str:
    if not ctx_list: 
        return "(trống)"
    return "\n\n---\n\n".join(c.get("text", "") for c in ctx_list if c.get("text"))

def _build_prompt(query: str, ctx_list: List[Dict[str, Any]], long_answer: bool) -> str:
    merged_ctx = _merge_ctx(ctx_list)
    tpl = SYNTH_LONG_TEMPLATE if long_answer else SYNTH_SINGLE_TEMPLATE
    return tpl.format(ctx=merged_ctx, query=query)

def answer_question(
    query: str,
    k: int = 4,
    filters: Dict[str, Any] | None = None,
    num_candidates: int = 100,
    synthesize: str | bool = "single",
    gen_model: str = "gemini-2.0-flash",
    force_quote: bool = True,          # không dùng trong mã này, nhưng giữ tham số cho tương thích
    long_answer: bool = False,
    history_text: str | None = None,   # có thể chèn vào ctx nếu muốn
) -> Dict[str, Any]:
    if filters is None:
        filters = {"meta.type": {"$in": ["analysis", "poem", "summary", "bio"]}}

    # 1) Retrieve rộng vừa đủ
    hits = retrieve_context(query, k=max(k, 10), filters=filters, num_candidates=num_candidates)
    if not hits:
        prompt = _build_prompt(query, [], long_answer)
        return {"query": query, "prompt": prompt, "contexts": []}

    # 2) Độ tin cậy nhanh
    avg_score = sum(h.get("score", 0.0) for h in hits) / max(1, len(hits))
    if avg_score < 0.2:
        prompt = _build_prompt(query, [], long_answer)
        return {"query": query, "prompt": prompt, "contexts": []}

    # 3) Rerank theo env rồi lấy top-k
    hits = rerank(query, hits, top_k=k)

    # 4) (tuỳ chọn) ghép history ngắn hạn vào cuối ctx để giữ mạch đối thoại
    if history_text:
        hits = hits + [{"text": f"[LỊCH SỬ HỘI THOẠI]\n{history_text}"}]

    # 5) Synthesize
    prompt = _build_prompt(query, hits, long_answer)
    result = {"query": query, "prompt": prompt, "contexts": hits}

    if synthesize and synthesize != "mapreduce":
        ans = generate_answer_gemini(prompt, model=gen_model, max_output_tokens=(4096 if long_answer else 2048), long_answer=long_answer)
    elif synthesize == "mapreduce":
        # (Nếu muốn, bạn có thể cài synthesis.mapreduce ở đây)
        ans = generate_answer_gemini(prompt, model=gen_model, max_output_tokens=(4096 if long_answer else 2048), long_answer=long_answer)
    else:
        ans = None

    if ans:
        # KHÔNG chèn “Nguồn: …” để dành token cho nội dung
        result["answer"] = ans

    return result
