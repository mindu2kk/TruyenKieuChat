from typing import Dict, Any, List, Tuple, Optional
try:  # pragma: no cover - support both package and script execution
    from .router import route_intent, parse_poem_request
    from .rag_pipeline import answer_question
    from .generation import generate_answer_gemini
    from .faq import lookup_faq
    from .cache import get_cached, set_cached
    from .poem_tools import (
        poem_ready,
        get_opening,
        get_range,
        get_single,
        compare_lines,
    )
    from .prompt_engineering import (
        DEFAULT_LONG_TOKEN_BUDGET,
        DEFAULT_SHORT_TOKEN_BUDGET,
        build_generic_prompt,
        build_poem_disambiguation_prompt,
        build_smalltalk_prompt,
        build_poem_compare_prompt,
    )
    from .verifier import verify_poem_quotes
except ImportError:  # pragma: no cover - script mode
    from router import route_intent, parse_poem_request
    from rag_pipeline import answer_question
    from generation import generate_answer_gemini
    from faq import lookup_faq
    from cache import get_cached, set_cached
    from poem_tools import (
        poem_ready,
        get_opening,
        get_range,
        get_single,
        compare_lines,
    )
    from prompt_engineering import (
        DEFAULT_LONG_TOKEN_BUDGET,
        DEFAULT_SHORT_TOKEN_BUDGET,
        build_generic_prompt,
        build_poem_disambiguation_prompt,
        build_smalltalk_prompt,
        build_poem_compare_prompt,
    )
    from verifier import verify_poem_quotes

def _norm_key(q: str) -> str:
    return (q or "").strip().lower()

def _history_to_text(history: Optional[List[Tuple[str,str]]], max_turns=6) -> str:
    if not history: return ""
    h = history[-max_turns:]
    lines = []
    for role, txt in h:
        role = "USER" if role=="user" else "ASSISTANT"
        lines.append(f"[{role}]\n{txt}")
    return "\n\n".join(lines)

def answer_with_router(
    query: str,
    k: int = 5,
    gemini_model: str = "gemini-2.0-flash",
    history: Optional[List[Tuple[str,str]]] = None,
    long_answer: bool = False,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:

    qkey = _norm_key(query)
    short_history = _history_to_text(history, max_turns=4)
    full_history = _history_to_text(history, max_turns=8)

    if max_tokens is None:
        max_tokens = DEFAULT_LONG_TOKEN_BUDGET if long_answer else DEFAULT_SHORT_TOKEN_BUDGET

    # 0) cache
    cached = get_cached(qkey)
    if cached:
        return {"intent": "cache", "answer": cached, "sources": []}

    # 1) FAQ (không in nguồn)
    hit = lookup_faq(query)
    if hit:
        ans = hit["answer"]
        set_cached(qkey, ans)
        return {"intent": "faq", "answer": ans, "sources": []}

    # 2) route
    intent = route_intent(query)

    if intent == "chitchat":
        prompt = build_smalltalk_prompt(query, history_text=short_history)
        ans = generate_answer_gemini(prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens)
        set_cached(qkey, ans)
        return {"intent": intent, "answer": ans, "sources": []}

    if intent == "generic":
        prompt = build_generic_prompt(query, history_text=full_history, depth="expanded" if long_answer else "balanced")
        ans = generate_answer_gemini(prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens)
        set_cached(qkey, ans)
        return {"intent": intent, "answer": ans, "sources": []}

    # POEM MODE
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
                txt = "\n".join(f"{i+1:>4}: {ln}" for i, ln in enumerate(lines))
                ans = f"**{n} câu đầu Truyện Kiều:**\n\n{txt}"
                set_cached(qkey, ans)
                return {"intent": "poem", "answer": ans, "sources": []}
            if kind == "range":
                a, b = int(spec[1]), int(spec[2])
                if a > b: a, b = b, a
                lines = get_range(a, b)
                txt = "\n".join(f"{a+i:>4}: {ln}" for i, ln in enumerate(lines))
                ans = f"**Các câu {a}–{b} trong Truyện Kiều:**\n\n{txt}"
                set_cached(qkey, ans)
                return {"intent": "poem", "answer": ans, "sources": []}
            if kind == "single":
                n = int(spec[1])
                ln = get_single(n)
                if ln:
                    ans = f"**Câu {n} trong Truyện Kiều:**\n\n{n:>4}: {ln}"
                else:
                    ans = f"Chưa tra được câu {n} (vượt ngoài số dòng hiện có)."
                set_cached(qkey, ans)
                return {"intent": "poem", "answer": ans, "sources": []}
            if kind == "compare":
                a, b = int(spec[1]), int(spec[2])
                line_a, line_b = compare_lines(a, b)
                if not line_a or not line_b:
                    ans = "Không đủ dữ liệu để so sánh hai câu được yêu cầu."
                    set_cached(qkey, ans)
                    return {"intent": "poem", "answer": ans, "sources": []}
                prompt = build_poem_compare_prompt(
                    query,
                    line_a=line_a,
                    line_b=line_b,
                    history_text=short_history,
                )
                ans = generate_answer_gemini(
                    prompt,
                    model=gemini_model,
                    long_answer=long_answer,
                    max_tokens=max_tokens,
                )
                verification = verify_poem_quotes(ans)
                set_cached(qkey, ans)
                sources = [f"câu {line_a.number}", f"câu {line_b.number}"]
                return {
                    "intent": "poem",
                    "answer": ans,
                    "sources": sources,
                    "verification": verification,
                }

        # không parse được — nhờ model hỏi lại ngắn
        prompt = build_poem_disambiguation_prompt(query, history_text=short_history)
        ans = generate_answer_gemini(prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens)
        verification = verify_poem_quotes(ans)
        set_cached(qkey, ans)
        return {"intent": "poem", "answer": ans, "sources": [], "verification": verification}

    # 3) Domain → RAG
    pack = answer_question(
        query,
        k=k,
        synthesize="single",
        gen_model=gemini_model,
        force_quote=True,
        long_answer=long_answer,
        history_text=full_history,
        max_tokens=max_tokens,
    )

    ans = pack.get("answer")
    if ans:
        verification = verify_poem_quotes(ans)
        set_cached(qkey, ans)
        return {"intent": "domain", "answer": ans, "sources": [], "verification": verification}

    # 4) fallback
    ans = generate_answer_gemini(pack["prompt"], model=gemini_model, long_answer=long_answer, max_tokens=max_tokens)
    verification = verify_poem_quotes(ans)
    set_cached(qkey, ans)
    return {"intent": "domain", "answer": ans, "sources": [], "verification": verification}