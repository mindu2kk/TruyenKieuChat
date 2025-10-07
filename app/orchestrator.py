# app/orchestrator.py
# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple, Optional
from router import route_intent, parse_poem_request
from rag_pipeline import answer_question
from generation import generate_answer_gemini
from faq import lookup_faq
from cache import get_cached, set_cached

# poem tools
from poem_tools import (
    poem_ready, get_opening, get_range, get_line,
    search_lines_by_keywords, search_lines_by_span
)

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

def _history_to_text(history: Optional[List[Tuple[str,str]]], max_turns=8) -> str:
    if not history: return ""
    h = history[-max_turns:]
    lines = []
    for role, txt in h:
        role = "USER" if role=="user" else "ASSISTANT"
        lines.append(f"[{role}]\n{txt}")
    return "\n\n".join(lines)

def _fmt_numbered(lines: List[str], start: int = 1) -> str:
    return "\n".join(f"{start+i:>4}: {ln}" for i, ln in enumerate(lines))

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
        srcs = hit.get("sources", [])
        if srcs: ans += "\n\n**Nguồn:** " + "; ".join(srcs)
        set_cached(qkey, ans); return {"intent":"faq","answer":ans,"sources":srcs}

    # 2) route
    intent = route_intent(query)

    # chitchat
    if intent == "chitchat":
        ans = generate_answer_gemini(_wrap_user_prompt(SMALL_TALK_SYS, query), model=gemini_model)
        set_cached(qkey, ans); return {"intent":intent, "answer":ans, "sources":[]}

    # generic
    if intent == "generic":
        ans = generate_answer_gemini(_wrap_user_prompt(GENERIC_SYS, query), model=gemini_model)
        set_cached(qkey, ans); return {"intent":intent, "answer":ans, "sources":[]}

    # poem mode
    if intent == "poem":
        if not poem_ready():
            msg = "Kho thơ chưa sẵn sàng (cần data/interim/poem/poem.txt, mỗi câu 1 dòng)."
            set_cached(qkey, msg); return {"intent":"poem","answer":msg,"sources":[]}

        spec = parse_poem_request(query.lower())
        if spec and spec[0] == "opening":
            n = max(1, min(int(spec[1]), 400))
            lines = get_opening(n)
            ans = f"**{n} câu đầu Truyện Kiều:**\n\n" + _fmt_numbered(lines, 1)
            set_cached(qkey, ans); return {"intent":"poem","answer":ans,"sources":["poem"]}

        if spec and spec[0] == "range":
            a, b = int(spec[1]), int(spec[2])
            if a > b: a, b = b, a
            lines = get_range(a, b)
            ans = f"**Các câu {a}–{b} trong Truyện Kiều:**\n\n" + _fmt_numbered(lines, a)
            set_cached(qkey, ans); return {"intent":"poem","answer":ans,"sources":["poem"]}

        if spec and spec[0] == "single":
            n = int(spec[1])
            line = get_line(n)
            if line:
                ans = f"**Câu {n} trong Truyện Kiều:**\n\n{n:>4}: {line}"
            else:
                ans = f"Không tìm thấy câu {n}."
            set_cached(qkey, ans); return {"intent":"poem","answer":ans,"sources":["poem"]}

        if spec and spec[0] == "compare":
            a, b = int(spec[1]), int(spec[2])
            la, lb = get_line(a), get_line(b)
            if la and lb:
                ans = (
                    f"**So sánh câu {a} và {b}:**\n\n"
                    f"{a:>4}: {la}\n{b:>4}: {lb}\n\n"
                    "• Vần/nhịp: cả hai giữ nhịp lục bát (6–8) quen thuộc; đối ứng nghĩa qua cặp từ then chốt.\n"
                    "• Hình ảnh / trường từ vựng: đối chiếu những từ khóa lặp/đối/ẩn dụ để nêu sắc thái.\n"
                    "• Ý: nêu điểm giống/khác (bổ sung hay chuyển ý) → rút ra hiệu quả nghệ thuật."
                )
            else:
                ans = "Chưa tra được đủ hai câu để so sánh."
            set_cached(qkey, ans); return {"intent":"poem","answer":ans,"sources":["poem"]}

        # không parse được → hỏi lại ngắn
        prompt = _wrap_user_prompt(
            "Bạn giúp người dùng trích thơ theo số câu hoặc khoảng câu (ví dụ: '10 câu đầu', 'câu 241–260', 'câu 11 là gì'). Nếu họ chưa nêu rõ, hãy hỏi lại rất ngắn.",
            query
        )
        ans = generate_answer_gemini(prompt, model=gemini_model)
        set_cached(qkey, ans); return {"intent":"poem","answer":ans,"sources":[]}

    # 3) RAG domain: luôn cố gắng chèn dẫn thơ minh hoạ
    hist_text = _history_to_text(history, max_turns=8)
    pack = answer_question(
        query,
        k=k,
        synthesize="single",
        gen_model=gemini_model,
        force_quote=True,
        long_answer=long_answer,
        history_text=hist_text,
    )

    # Nếu pack có gợi ý trích thơ (từ quote step), gắn vào trả lời nếu chưa có
    ans = pack.get("answer") or generate_answer_gemini(pack["prompt"], model=gemini_model)
    if pack.get("quotes"):
        qtxt = "\n".join([f"{n:>4}: {ln}" for n, ln in pack["quotes"]])
        ans = f"{ans}\n\n**Trích thơ minh hoạ:**\n{qtxt}"

    if pack.get("contexts"):
        srcs = _sources_from_ctx(pack["contexts"])
        if srcs:
            ans = f"{ans}\n\n**Nguồn:** {srcs}"

    set_cached(qkey, ans)
    return {"intent":"domain","answer":ans,"sources":pack.get("contexts",[])}
