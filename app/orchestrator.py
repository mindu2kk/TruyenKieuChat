# app/orchestrator.py
# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple, Optional

def _norm_key(q: str) -> str:
    return (q or "").strip().lower()

def _history_to_text(history: Optional[List[Tuple[str, str]]], max_turns: int = 6) -> str:
    if not history:
        return ""
    h = history[-max_turns:]
    lines = []
    for role, txt in h:
        role = "USER" if role == "user" else "ASSISTANT"
        lines.append(f"[{role}]\n{txt}")
    return "\n\n".join(lines)

def _generation_failure_response(intent: str, reason: str, *, sources: Optional[List[str]] = None) -> Dict[str, Any]:
    detail = (reason or "").strip()
    message = (
        "🤖 Xin lỗi, hệ thống chưa thể gọi mô hình Gemini để tạo câu trả lời. "
        "Vui lòng kiểm tra GOOGLE_API_KEY và kết nối mạng."
    )
    if detail:
        message += f"\n\nChi tiết kỹ thuật: {detail}"
    return {"intent": intent, "answer": message, "sources": sources or [], "error": detail}

def _safe_generate(
    intent: str,
    prompt: str,
    *,
    sources: Optional[List[str]] = None,
    **gen_kwargs: Any,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Gọi Gemini an toàn:
    - ép kiểu max_tokens -> int (tránh len(int))
    - nếu model trả rỗng -> quy thành lỗi để UI không im lặng
    - bắt mọi exception -> trả payload lỗi thống nhất
    """
    if "max_tokens" in gen_kwargs and gen_kwargs["max_tokens"] is not None:
        try:
            gen_kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        except Exception:
            del gen_kwargs["max_tokens"]

    try:
        from .generation import generate_answer_gemini
        out: str = generate_answer_gemini(prompt, **gen_kwargs)
        if not (out and out.strip()):
            # ❗ Quan trọng: coi output rỗng là lỗi có thông báo rõ ràng
            return None, _generation_failure_response(intent, "Model trả về nội dung rỗng.", sources=sources)
        return out, None
    except Exception as exc:
        return None, _generation_failure_response(intent, str(exc), sources=sources)

def answer_with_router(
    query: str,
    k: int = 5,
    gemini_model: str = "gemini-2.0-flash",
    history: Optional[List[Tuple[str, str]]] = None,
    long_answer: bool = False,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:

    # ⬇️ Lazy import tất cả submodule PHỤ THUỘC — chỉ khi hàm được gọi
    from .router import route_intent, parse_poem_request
    from .rag_pipeline import answer_question
    from .faq import lookup_faq
    from .cache import get_cached, set_cached
    from .poem_tools import poem_ready, get_opening, get_range, get_single, compare_lines
    from .prompt_engineering import (
        DEFAULT_LONG_TOKEN_BUDGET,
        DEFAULT_SHORT_TOKEN_BUDGET,
        build_generic_prompt,
        build_poem_disambiguation_prompt,
        build_smalltalk_prompt,
        build_poem_compare_prompt,
    )
    from .verifier import verify_poem_quotes

    qkey = _norm_key(query)
    short_history = _history_to_text(history, max_turns=4)
    full_history = _history_to_text(history, max_turns=8)

    if max_tokens is None:
        max_tokens = DEFAULT_LONG_TOKEN_BUDGET if long_answer else DEFAULT_SHORT_TOKEN_BUDGET

    # 0) cache
    cached = get_cached(qkey)
    if cached:
        return {"intent": "cache", "answer": cached, "sources": []}

    # 1) FAQ
    hit = lookup_faq(query)
    if hit:
        ans = hit["answer"]
        set_cached(qkey, ans)
        return {"intent": "faq", "answer": ans, "sources": []}

    # 2) route
    intent = route_intent(query)

    if intent == "chitchat":
        prompt = build_smalltalk_prompt(query, history_text=short_history)
        ans, failure = _safe_generate(
            intent, prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens
        )
        if failure: return failure
        set_cached(qkey, ans or "")
        return {"intent": intent, "answer": ans or "", "sources": []}

    if intent == "generic":
        prompt = build_generic_prompt(query, history_text=full_history, depth="expanded" if long_answer else "balanced")
        ans, failure = _safe_generate(
            intent, prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens
        )
        if failure: return failure
        set_cached(qkey, ans or "")
        return {"intent": intent, "answer": ans or "", "sources": []}

    if intent == "poem":
        if not poem_ready():
            msg = "Kho thơ chưa sẵn sàng (cần data/interim/poem/poem.txt, mỗi câu 1 dòng)."
            set_cached(qkey, msg)
            return {"intent": "poem", "answer": msg, "sources": []}

        spec = parse_poem_request(query)
        if spec:
            kind = spec[0]
            if kind == "opening":
                n = max(1, min(int(spec[1]), 1500))
                lines = get_opening(n)
                txt = "\n".join(f"{i + 1:>4}: {ln}" for i, ln in enumerate(lines))
                ans = f"**{n} câu đầu Truyện Kiều:**\n\n{txt}"
                set_cached(qkey, ans)
                return {"intent": "poem", "answer": ans, "sources": []}
            if kind == "range":
                a, b = int(spec[1]), int(spec[2])
                if a > b: a, b = b, a
                lines = get_range(a, b)
                txt = "\n".join(f"{a + i:>4}: {ln}" for i, ln in enumerate(lines))
                ans = f"**Các câu {a}–{b} trong Truyện Kiều:**\n\n{txt}"
                set_cached(qkey, ans)
                return {"intent": "poem", "answer": ans, "sources": []}
            if kind == "single":
                n = int(spec[1])
                ln = get_single(n)
                ans = f"**Câu {n} trong Truyện Kiều:**\n\n{n:>4}: {ln}" if ln else f"Chưa tra được câu {n} (vượt ngoài số dòng hiện có)."
                set_cached(qkey, ans)
                return {"intent": "poem", "answer": ans, "sources": []}
            if kind == "compare":
                a, b = int(spec[1]), int(spec[2])
                line_a, line_b = compare_lines(a, b)
                if not line_a or not line_b:
                    ans = "Không đủ dữ liệu để so sánh hai câu được yêu cầu."
                    set_cached(qkey, ans)
                    return {"intent": "poem", "answer": ans, "sources": []}
                prompt = build_poem_compare_prompt(query, line_a=line_a, line_b=line_b, history_text=short_history)
                ans, failure = _safe_generate(
                    "poem", prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens,
                    sources=[f"câu {line_a.number}", f"câu {line_b.number}"],
                )
                if failure: return failure
                verification = verify_poem_quotes(ans or "")
                set_cached(qkey, ans or "")
                return {"intent": "poem", "answer": ans or "", "sources": [f"câu {line_a.number}", f"câu {line_b.number}"], "verification": verification}

        prompt = build_poem_disambiguation_prompt(query, history_text=short_history)
        ans, failure = _safe_generate("poem", prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens)
        if failure: return failure
        verification = verify_poem_quotes(ans or "")
        set_cached(qkey, ans or "")
        return {"intent": "poem", "answer": ans or "", "sources": [], "verification": verification}

    # 3) Domain → RAG
    pack = answer_question(
        query, k=k, synthesize="single", gen_model=gemini_model,
        force_quote=True, long_answer=long_answer, history_text=full_history, max_tokens=max_tokens,
    )
    if pack.get("generation_error"):
        return _generation_failure_response("domain", str(pack["generation_error"]))

    ans = pack.get("answer")
    if ans:
        verification = verify_poem_quotes(ans or "")
        set_cached(qkey, ans or "")
        return {"intent": "domain", "answer": ans or "", "sources": [], "verification": verification}

    # 4) fallback — dùng prompt đã build
    ans, failure = _safe_generate("domain", pack["prompt"], model=gemini_model, long_answer=long_answer, max_tokens=max_tokens)
    if failure: return failure
    verification = verify_poem_quotes(ans or "")
    set_cached(qkey, ans or "")
    return {"intent": "domain", "answer": ans or "", "sources": [], "verification": verification}
