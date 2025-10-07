# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import hashlib

try:
    from app.router import route_intent, parse_poem_request
    from app.rag_pipeline import answer_question
    from app.generation import generate_answer_gemini
    from app.faq import lookup_faq
    from app.cache import get_cached, set_cached
    from app.poem_tools import poem_ready, get_opening, get_range
except Exception:
    from router import route_intent, parse_poem_request  # type: ignore
    from rag_pipeline import answer_question            # type: ignore
    from generation import generate_answer_gemini       # type: ignore
    from faq import lookup_faq                          # type: ignore
    from cache import get_cached, set_cached            # type: ignore
    from poem_tools import poem_ready, get_opening, get_range  # type: ignore

SMALL_TALK_SYS = "Bạn là một trợ lý thân thiện. Trả lời rất ngắn (≤ 2 câu), lịch sự."
GENERIC_SYS    = "Bạn là một trợ lý kiến thức tổng quát. Trả lời chính xác, ngắn gọn, dễ hiểu."

def _wrap_user_prompt(system: str, user: str, history_text: str = "") -> str:
    if history_text:
        return f"[SYSTEM]\n{system}\n\n{history_text}\n\n[USER]\n{user}"
    return f"[SYSTEM]\n{system}\n\n[USER]\n{user}"

def _sources_join(ctx: List[dict] | None) -> str:
    seen = []
    for c in ctx or []:
        s = (c.get("meta") or {}).get("source")
        if s and s not in seen:
            seen.append(s)
    return "; ".join(seen)

def _history_to_text(history: Optional[List[Tuple[str, str]]], max_turns=8) -> str:
    if not history:
        return ""
    h = history[-max_turns:]
    lines = []
    for role, txt in h:
        role = "USER" if role == "user" else "ASSISTANT"
        lines.append(f"[{role}]\n{txt}")
    return "\n\n".join(lines)

def _ckey(q: str, intent: str, hist: str) -> str:
    h = hashlib.sha1(hist.encode("utf-8")).hexdigest()[:8] if hist else "nohist"
    return f"{intent}::{q.strip().lower()}::{h}"

def answer_with_router(
    query: str,
    k: int = 4,
    gemini_model: str = "gemini-2.0-flash",
    history: Optional[List[Tuple[str, str]]] = None,
    long_answer: bool = False,
) -> Dict[str, Any]:

    # FAQ trước để phản hồi nhanh
    hit = lookup_faq(query)
    if hit:
        ans = hit["answer"]
        srcs = hit.get("sources", [])
        if srcs:
            ans += "\n\n**Nguồn:** " + "; ".join(srcs)
        set_cached(_ckey(query, "faq", ""), ans)
        return {"intent": "faq", "answer": ans, "sources": srcs}

    intent = route_intent(query)
    hist_text = _history_to_text(history, max_turns=8)
    ckey = _ckey(query, intent, hist_text)

    cached = get_cached(ckey)
    if cached:
        return {"intent": "cache", "answer": cached, "sources": []}

    # Poem mode
    if intent == "poem":
        if not poem_ready():
            msg = "Kho thơ chưa sẵn sàng (cần data/interim/poem/poem.txt, mỗi câu 1 dòng)."
            set_cached(ckey, msg); return {"intent": "poem", "answer": msg, "sources": []}

        spec = parse_poem_request(query)
        if spec and spec[0] == "opening":
            n = max(1, min(int(spec[1]), 400))
            lines = get_opening(n)
            txt = "\n".join(f"{i+1:>4}: {ln}" for i, ln in enumerate(lines))
            ans = f"**{n} câu đầu Truyện Kiều:**\n\n{txt}"
            set_cached(ckey, ans); return {"intent": "poem", "answer": ans, "sources": ["poem"]}

        if spec and spec[0] == "range":
            a, b = int(spec[1]), int(spec[2])
            if a > b: a, b = b, a
            lines = get_range(a, b)
            txt = "\n".join(f"{a+i:>4}: {ln}" for i, ln in enumerate(lines))
            ans = f"**Các câu {a}–{b} trong Truyện Kiều:**\n\n{txt}"
            set_cached(ckey, ans); return {"intent": "poem", "answer": ans, "sources": ["poem"]}

        prompt = _wrap_user_prompt(
            "Bạn giúp người dùng trích thơ theo số câu hoặc khoảng câu (ví dụ: '10 câu đầu', 'câu 241–260'). Nếu họ chưa nêu rõ, hãy hỏi lại rất ngắn.",
            query, hist_text
        )
        ans = generate_answer_gemini(prompt, model=gemini_model)
        set_cached(ckey, ans); return {"intent": "poem", "answer": ans, "sources": []}

    # Small talk
    if intent == "chitchat":
        prompt = _wrap_user_prompt(SMALL_TALK_SYS, query, hist_text)
        ans = generate_answer_gemini(prompt, model=gemini_model)
        set_cached(ckey, ans); return {"intent": intent, "answer": ans, "sources": []}

    # Generic
    if intent == "generic":
        prompt = _wrap_user_prompt(GENERIC_SYS, query, hist_text)
        ans = generate_answer_gemini(prompt, model=gemini_model)
        set_cached(ckey, ans); return {"intent": intent, "answer": ans, "sources": []}

    # Domain → RAG
    pack = answer_question(
        query, k=k, synthesize="single", gen_model=gemini_model,
        force_quote=True, long_answer=long_answer, history_text=hist_text
    )
    ans = pack.get("answer")
    if ans:
        srcs = _sources_join(pack.get("contexts"))
        if srcs:
            ans = f"{ans}\n\n**Nguồn:** {srcs}"
        set_cached(ckey, ans)
        return {"intent": "domain", "answer": ans, "sources": srcs}

    # Fallback
    ans = generate_answer_gemini(pack.get("prompt", query), model=gemini_model)
    set_cached(ckey, ans)
    return {"intent": "domain", "answer": ans, "sources": _sources_join(pack.get("contexts"))}
