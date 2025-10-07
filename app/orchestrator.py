# app/orchestrator.py
# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple, Optional
from router import route_intent, parse_poem_request
from rag_pipeline import answer_question
from generation import generate_answer_gemini
from faq import lookup_faq
from cache import get_cached, set_cached
from poem_tools import poem_ready, get_opening, get_range, preview_numbered, total_lines

SMALL_TALK_SYS = "Bạn là một trợ lý thân thiện. Trả lời rất ngắn (≤ 2 câu), lịch sự."
GENERIC_SYS    = "Bạn là một trợ lý kiến thức tổng quát. Trả lời chính xác, ngắn gọn, dễ hiểu."

def _wrap_user_prompt(system: str, user: str) -> str:
    return f"[SYSTEM]\n{system}\n\n[USER]\n{user}"

def _norm_key(q: str) -> str:
    return (q or "").strip().lower()

def _sources_from_ctx(ctx: List[dict]) -> str:
    seen = []
    for c in ctx or []:
        src = (c.get("meta") or {}).get("source")
        if src and src not in seen:
            seen.append(src)
    return "; ".join(seen)

def _history_to_text(history: Optional[List[Tuple[str,str]]], max_turns=6) -> str:
    if not history: return ""
    h = history[-max_turns:]
    lines = []
    for role, txt in h:
        role = "USER" if role=="user" else "ASSISTANT"
        lines.append(f"[{role}]\n{txt}")
    return "\n\n".join(lines)

def _wants_analysis(q: str) -> bool:
    qs = (q or "").lower()
    # nếu có các từ khóa này thì kèm phân tích sau khi trích
    return any(k in qs for k in ["phân tích","bình giảng","cảm nhận","giải thích","bình luận"])

def answer_with_router(
    query: str,
    k: int = 4,
    gemini_model: str = "gemini-2.0-flash",
    history: Optional[List[Tuple[str,str]]] = None,
    long_answer: bool = False,
) -> Dict[str, Any]:

    qkey = _norm_key(query)

    # 0) cache
    cached = get_cached(qkey)
    if cached:
        return {"intent": "cache", "answer": cached, "sources": []}

    # 1) FAQ
    hit = lookup_faq(query)
    if hit:
        ans = hit["answer"]
        set_cached(qkey, ans)
        return {"intent": "faq", "answer": ans, "sources": hit.get("sources", [])}

    # 2) route
    intent = route_intent(query)

    # chitchat
    if intent == "chitchat":
        prompt = _wrap_user_prompt(SMALL_TALK_SYS, query)
        ans = generate_answer_gemini(prompt, model=gemini_model)
        set_cached(qkey, ans)
        return {"intent": intent, "answer": ans, "sources": []}

    # generic
    if intent == "generic":
        prompt = _wrap_user_prompt(GENERIC_SYS, query)
        ans = generate_answer_gemini(prompt, model=gemini_model)
        set_cached(qkey, ans)
        return {"intent": intent, "answer": ans, "sources": []}

    # poem mode (trích chắc chắn + tùy chọn phân tích)
    if intent == "poem":
        if not poem_ready():
            msg = "Kho thơ chưa sẵn sàng (cần data/interim/poem/poem.txt, mỗi câu 1 dòng)."
            set_cached(qkey, msg)
            return {"intent": "poem", "answer": msg, "sources": []}

        spec = parse_poem_request(query)
        if spec and spec[0] == "opening":
            n = max(1, min(int(spec[1]), 600))  # cho phép tới 600 câu đầu
            lines = get_opening(n)
            if len(lines) != n:
                total = total_lines()
                ans = (
                    f"**{n} câu đầu Truyện Kiều (chỉ trích được {len(lines)}/{n} câu; tổng {total} câu trong kho):**\n\n"
                    + preview_numbered(1, lines)
                )
                set_cached(qkey, ans)
                return {"intent": "poem", "answer": ans, "sources": ["poem"]}

            block = preview_numbered(1, lines)
            if _wants_analysis(query):
                # phân tích dựa TRỰC TIẾP khối thơ (grounded)
                sys = (
                    "Bạn là giáo viên Ngữ văn, phân tích ngắn gọn, mạch lạc, có dẫn chứng (trích đúng từng câu trong KHỐI THƠ). "
                    "Không được bịa câu thơ; chỉ trích trong khối thơ đã cho."
                )
                prompt = f"[SYSTEM]\n{sys}\n\n[KHỐI THƠ]\n{block}\n\n[USER]\nPhân tích nội dung và nghệ thuật của đoạn thơ trên (≤ 12 gạch đầu dòng)."
                analysis = generate_answer_gemini(prompt, model=gemini_model)
                ans = f"**{n} câu đầu Truyện Kiều:**\n\n{block}\n\n---\n\n{analysis}"
            else:
                ans = f"**{n} câu đầu Truyện Kiều:**\n\n{block}"
            set_cached(qkey, ans)
            return {"intent": "poem", "answer": ans, "sources": ["poem"]}

        if spec and spec[0] == "range":
            a, b = int(spec[1]), int(spec[2])
            lines = get_range(a, b)
            got = len(lines)
            block = preview_numbered(a, lines)
            if got != (b - a + 1):
                total = total_lines()
                ans = (
                    f"**Các câu {a}–{b} (chỉ trích được {got}/{b-a+1} câu; tổng {total} câu trong kho):**\n\n"
                    + block
                )
                set_cached(qkey, ans)
                return {"intent": "poem", "answer": ans, "sources": ["poem"]}

            if _wants_analysis(query):
                sys = (
                    "Bạn là giáo viên Ngữ văn, phân tích ngắn gọn, mạch lạc, có dẫn chứng (trích đúng từng câu trong KHỐI THƠ). "
                    "Không được bịa câu thơ; chỉ trích trong khối thơ đã cho."
                )
                prompt = f"[SYSTEM]\n{sys}\n\n[KHỐI THƠ]\n{block}\n\n[USER]\nGiải thích và bình giảng các câu trên (≤ 12 gạch đầu dòng)."
                analysis = generate_answer_gemini(prompt, model=gemini_model)
                ans = f"**Các câu {a}–{b} trong Truyện Kiều:**\n\n{block}\n\n---\n\n{analysis}"
            else:
                ans = f"**Các câu {a}–{b} trong Truyện Kiều:**\n\n{block}"

            set_cached(qkey, ans)
            return {"intent": "poem", "answer": ans, "sources": ["poem"]}

        # không parse được — hỏi lại ngắn
        prompt = _wrap_user_prompt(
            "Bạn giúp người dùng trích thơ theo số câu hoặc khoảng câu (vd: '20 câu đầu', 'câu 241–260', 'câu 11'). Nếu họ chưa nêu rõ, hãy hỏi lại rất ngắn.",
            query
        )
        ans = generate_answer_gemini(prompt, model=gemini_model)
        set_cached(qkey, ans)
        return {"intent": "poem", "answer": ans, "sources": []}

    # 3) Domain → RAG
    hist_text = _history_to_text(history, max_turns=8)
    pack = answer_question(
        query,
        k=k,
        synthesize="single",
        gen_model=gemini_model,
        force_quote=True,           # cố gắng chèn trích dẫn trong phần RAG
        long_answer=long_answer,
        history_text=hist_text,
    )
    ans = pack.get("answer")
    if ans:
        set_cached(qkey, ans)
        return {"intent": "domain", "answer": ans, "sources": pack.get("contexts", [])}

    # 4) fallback
    ans = generate_answer_gemini(pack["prompt"], model=gemini_model)
    set_cached(qkey, ans)
    return {"intent": "domain", "answer": ans, "sources": pack.get("contexts", [])}
