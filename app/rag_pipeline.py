# app/rag_pipeline.py
# -*- coding: utf-8 -*-
"""
RAG pipeline (single-shot), có chèn 'trích thơ' từ poem_tools khi force_quote=True.
"""
from typing import List, Dict, Any, Optional
import importlib.util, sys
from pathlib import Path
from rerank import rerank
from generation import generate_answer_gemini

# poem quote
from poem_tools import search_lines_by_keywords, search_lines_by_span

# dynamic import scripts/04_retrieve.py
ROOT = Path(__file__).resolve().parents[1]
retr_path = ROOT / "scripts" / "04_retrieve.py"
spec = importlib.util.spec_from_file_location("retrieve_mod", retr_path)
retrieve_mod = importlib.util.module_from_spec(spec)
sys.modules["retrieve_mod"] = retrieve_mod
assert spec and spec.loader
spec.loader.exec_module(retrieve_mod)  # type: ignore
retrieve_context = retrieve_mod.retrieve_context

SYNTH_SINGLE_TMPL = """[LỊCH SỬ HỘI THOẠI]
{history}

[NGỮ CẢNH]
{ctx}

[TRÍCH THƠ THAM CHIẾU]
{quotes}

[HƯỚNG DẪN]
- Trả lời câu hỏi: "{query}" bằng 1 đoạn ngắn gọn, mạch lạc (mở–thân–kết).
- Ưu tiên dẫn chứng NGUYÊN VĂN từ phần TRÍCH THƠ (ghi số câu nếu có).
- Chỉ dùng thông tin trong NGỮ CẢNH + TRÍCH THƠ; nếu thiếu căn cứ, nói 'chưa đủ căn cứ'.
- Tránh liệt kê khô cứng; lập luận gọn, có liên kết ý.
"""

def _merge_ctx(ctx_list: List[Dict[str,Any]]) -> str:
    return "\n\n---\n\n".join(c["text"] for c in ctx_list)

def _format_sources(ctx_list: List[Dict[str, Any]]) -> str:
    srcs = []
    for c in ctx_list:
        s = c.get("meta", {}).get("source")
        if s and s not in srcs:
            srcs.append(s)
    return "; ".join(srcs)

def _pick_quotes(query: str, contexts: List[Dict[str,Any]], top=4):
    """Chọn vài câu thơ liên quan:
       - Từ chính câu hỏi
       - Từ đoạn ngữ cảnh top-2 (nếu có) để lấy từ khoá
    Trả về: list[(lineno, line)]
    """
    picks = []
    # 1) theo query
    for ln, s, sc in search_lines_by_keywords(query, top=top+2):
        picks.append((ln, s, sc))
    # 2) theo ngữ cảnh top-2
    for c in (contexts or [])[:2]:
        span = c.get("text","")
        for ln, s, sc in search_lines_by_span(span, top=2):
            picks.append((ln, s, sc*0.9))
    # unique theo lineno, lấy score cao
    best = {}
    for ln, s, sc in picks:
        if ln not in best or sc > best[ln][1]:
            best[ln] = (s, sc)
    ranked = sorted([(ln, ss[0], ss[1]) for ln, ss in best.items()], key=lambda x:(-x[2], x[0]))[:top]
    return [(ln, s) for ln, s, _ in ranked]

def answer_question(
    query: str,
    k: int = 4,
    filters: Optional[Dict[str, Any]] = None,
    num_candidates: int = 100,
    synthesize: str | bool = "single",
    gen_model: str = "gemini-2.0-flash",
    force_quote: bool = True,
    long_answer: bool = False,       # (chưa dùng ở đây, bạn có thể mở rộng thêm prompt)
    history_text: str = "",
) -> Dict[str, Any]:

    if filters is None:
        filters = {"meta.type": {"$in": ["analysis", "summary", "bio", "poem"]}}

    # 1) retrieve
    hits = retrieve_context(query, k=max(k, 10), filters=filters, num_candidates=num_candidates)
    if not hits:
        ctx_txt = ""
        quotes = _pick_quotes(query, [], top=4) if force_quote else []
    else:
        # 2) độ tin cậy nhanh
        avg_score = sum(h.get("score", 0.0) for h in hits) / max(1, len(hits))
        if avg_score < 0.2:
            hits = []
        ctx_txt = _merge_ctx(hits)
        quotes = _pick_quotes(query, hits, top=4) if force_quote else []

    quotes_txt = "\n".join([f"{n:>4}: {s}" for n, s in quotes]) if quotes else "(không có)"

    prompt = SYNTH_SINGLE_TMPL.format(
        history=history_text or "(trống)",
        ctx=ctx_txt or "(trống)",
        quotes=quotes_txt,
        query=query
    )
    result = {"query": query, "prompt": prompt, "contexts": hits, "quotes": quotes}

    if synthesize:
        ans = generate_answer_gemini(prompt, model=gen_model)
        if ans:
            srcs = _format_sources(hits)
            if srcs:
                ans = f"{ans}\n\n**Nguồn:** {srcs}"
            result["answer"] = ans

    return result
