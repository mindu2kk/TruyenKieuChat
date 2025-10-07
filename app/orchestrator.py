# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple, Optional
from router import route_intent, parse_poem_request
from rag_pipeline import answer_question
from generation import generate_answer_gemini
from faq import lookup_faq
from cache import get_cached, set_cached
from poem_tools import poem_ready, get_opening, get_range

SMALL_TALK_SYS = "Bạn là một trợ lý thân thiện. Trả lời rất ngắn (≤ 2 câu), lịch sự."
GENERIC_SYS    = "Bạn là một trợ lý kiến thức tổng quát. Trả lời chính xác, ngắn gọn, dễ hiểu."

def _wrap_user_prompt(system: str, user: str) -> str:
    return f"[SYSTEM]\n{system}\n\n[USER]\n{user}"

def _norm_key(q: str) -> str:
    return (q or "").strip().lower()

def _history_to_text(history: Optional[List[Tuple[str,str]]], max_turns=6) -> str:
    if not history: return ""
    h = history[-max_turns:]
    lines = []
    for role, txt in h:
        role = "USER" if role == "user" else "ASSISTANT"
        lines.append(f"[{role}]\n{txt}")
    return "\n\n".join(lines)

def answer_with_router(
    query: str,
    k: int = 4,
    gemini_model: str = "gemini-2.0-flash",
    history: Optional[List[Tuple[str,str]]] = None,
    long_answer: bool = False,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:

    qkey = _norm_key(query)

    # 0) cache
    cached = get_cached(qkey)
    if cached:
        return {"intent": "cache", "answer": cached, "sources": []}

    # 1) FAQ (không hiển thị nguồn)
    hit = lookup_faq(query)
    if hit:
        ans = hit["answer"]
        set_cached(qkey, ans)
        return {"intent": "faq", "answer": ans, "sources": []}

    # 2) định tuyến
    intent = route_intent(query)

    # 2.a) chitchat
    if intent == "chitchat":
        prompt = _wrap_user_prompt(SMALL_TALK_SYS, query)
        ans = generate_answer_gemini(prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens)
        set_cached(qkey, ans)
        return {"intent": intent, "answer": ans, "sources": []}

    # 2.b) generic factual
    if intent == "generic":
        prompt = _wrap_user_prompt(GENERIC_SYS, query)
        ans = generate_answer_gemini(prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens)
        set_cached(qkey, ans)
        return {"intent": intent, "answer": ans, "sources": []}

    # 2.c) poem mode – trích NGUYÊN VĂN, không gọi RAG
    if intent == "poem":
        if not poem_ready():
            msg = "Kho thơ chưa sẵn sàng (cần data/interim/poem/poem.txt, mỗi câu 1 dòng)."
            set_cached(qkey, msg)
            return {"intent": "poem", "answer": msg, "sources": []}

        spec = parse_poem_request(query)
        if spec and spec[0] == "opening":
            n = max(1, min(int(spec[1]), 1000))
            lines = get_opening(n)
            txt = "\n".join(f"{i+1:>4}: {ln}" for i, ln in enumerate(lines))
            ans = f"**{n} câu đầu Truyện Kiều:**\n\n{txt}"
            set_cached(qkey, ans)
            return {"intent": "poem", "answer": ans, "sources": []}

        if spec and spec[0] == "range":
            a, b = int(spec[1]), int(spec[2])
            if a > b: a, b = b, a
            lines = get_range(a, b)
            txt = "\n".join(f"{a+i:>4}: {ln}" for i, ln in enumerate(lines))
            ans = f"**Các câu {a}–{b} trong Truyện Kiều:**\n\n{txt}"
            set_cached(qkey, ans)
            return {"intent": "poem", "answer": ans, "sources": []}

        prompt = _wrap_user_prompt(
            "Bạn giúp người dùng trích thơ theo số câu hoặc khoảng câu (ví dụ: '10 câu đầu', 'câu 241–260'). Nếu họ chưa nêu rõ, hãy hỏi lại rất ngắn.",
            query
        )
        ans = generate_answer_gemini(prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens)
        set_cached(qkey, ans)
        return {"intent": "poem", "answer": ans, "sources": []}

    # 3) Domain → RAG + Prompt Engineering
    hist_text = _history_to_text(history, max_turns=8)
    pack = answer_question(
        query,
        k=k,
        synthesize="single",
        gen_model=gemini_model,
        force_quote=True,           # ưu tiên chèn 1–2 câu thơ nếu có trong context
        long_answer=long_answer,    # văn phong luận khi bật
        history_text=hist_text,     # giữ ngữ cảnh ngắn hạn
        max_tokens=max_tokens,      # dồn token cho câu trả lời
    )

    ans = pack.get("answer")
    if ans:
        set_cached(qkey, ans)
        return {"intent": "domain", "answer": ans, "sources": []}

    # 4) fallback — dùng prompt đã build
    ans = generate_answer_gemini(pack["prompt"], model=gemini_model, long_answer=long_answer, max_tokens=max_tokens)
    set_cached(qkey, ans)
    return {"intent": "domain", "answer": ans, "sources": []}
