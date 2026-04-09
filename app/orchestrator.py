# app/orchestrator.py
# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple, Optional
import os

# Bật debug (in kèm một ít metadata khi lỗi) bằng cách đặt biến môi trường: DEBUG_ORCH=1
_DEBUG_ORCH = os.getenv("DEBUG_ORCH", "0") == "1"

# Mặc định ẩn nguồn; đặt TKC_SHOW_SOURCES=1 để bật lại
_SHOW_SOURCES = os.getenv("TKC_SHOW_SOURCES", "0").strip().lower() in {"1", "true", "yes", "on"}

def _maybe_sources(srcs: Optional[List[str]]) -> List[str]:
    if not _SHOW_SOURCES:
        return []
    return list(srcs or [])

import re

_CHAR_NAMES = ["thúy kiều","thuy kieu","thúy vân","thuy van","kim trọng","kim trong",
               "từ hải","tu hai","hoạn thư","hoan thu","mã giám sinh","ma giam sinh",
               "thúc sinh","thuc sinh","bạc bà","bac ba","bạc hạnh","bac hanh"]

def _who_is_character(q: str) -> bool:
    ql = (q or "").lower().strip()
    if not re.search(r"\b(là ai|la ai)\b", ql):
        return False
    return any(name in ql for name in _CHAR_NAMES)


# ==== Heuristics cho close-reading & poem-only ====
_TRICH_DAN_TRIGGER = ["trích", "câu thơ", "nguyên văn", "dẫn", "lục bát", "nhịp", "vần", "điệp", "đối", "Lầu Ngưng Bích", "Đoạn trường"]
_CLOSE_READING_TRIGGER = ["trữ tình ngoại đề", "điểm nhìn", "ẩn dụ", "nhịp điệu", "mapping", "bản đồ ý niệm", "close reading"]

def _needs_poem_only(q: str) -> bool:
    ql = (q or "").lower()
    return any(t.lower() in ql for t in _TRICH_DAN_TRIGGER)

def _is_close_reading(q: str) -> bool:
    ql = (q or "").lower()
    return any(t.lower() in ql for t in _CLOSE_READING_TRIGGER)

def _norm_key(q: str) -> str:
    return (q or "").strip().lower()

def _make_cache_key(q: str, *, long_answer: bool, intent: str) -> str:
    return f"{_norm_key(q)}|la={int(bool(long_answer))}|intent={intent}"

def _history_to_text(history: Optional[List[Tuple[str, str]]], max_turns: int = 6) -> str:
    if not history:
        return ""
    h = history[-max_turns:]
    lines = []
    for role, txt in h:
        role = "USER" if role == "user" else "ASSISTANT"
        lines.append(f"[{role}]\n{txt}")
    return "\n\n".join(lines)

def _generation_failure_response(
    intent: str,
    reason: str,
    *,
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    detail = (reason or "").strip()
    message = (
        "🤖 Xin lỗi, hệ thống chưa thể gọi mô hình để tạo câu trả lời. "
        "Vui lòng kiểm tra API key và kết nối mạng."
    )
    if detail:
        message += f"\n\nChi tiết kỹ thuật: {detail}"
    # nguồn luôn rỗng nếu _SHOW_SOURCES = False
    return {"intent": intent, "answer": message, "sources": _maybe_sources(sources), "error": detail}

def _safe_generate(
    intent: str,
    prompt: str,
    *,
    sources: Optional[List[str]] = None,
    **gen_kwargs: Any,
):
    # ép kiểu max_tokens
    if "max_tokens" in gen_kwargs and gen_kwargs["max_tokens"] is not None:
        try:
            gen_kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        except Exception:
            del gen_kwargs["max_tokens"]

    def _dbg_meta(p: Any) -> Dict[str, Any]:
        try:
            plen = len(p)
        except Exception:
            plen = 0
        try:
            head = (p if isinstance(p, str) else str(p))[:400]
        except Exception:
            head = ""
        return {
            "model": gen_kwargs.get("model"),
            "max_tokens": gen_kwargs.get("max_tokens"),
            "prompt_type": type(p).__name__,
            "prompt_chars": plen,
            "prompt_head": head,
        }

    try:
        from .generation import generate_answer_gemini
        if not isinstance(prompt, str):
            prompt = str(prompt)
        out: str = generate_answer_gemini(prompt, **gen_kwargs)
        if not (out and out.strip()):
            failure = _generation_failure_response(intent, "Model trả về nội dung rỗng.", sources=sources)
            if _DEBUG_ORCH:
                failure["debug"] = _dbg_meta(prompt)
            return None, failure
        return out, None
    except Exception as exc:
        failure = _generation_failure_response(intent, str(exc), sources=sources)
        if _DEBUG_ORCH:
            failure["debug"] = _dbg_meta(prompt)
        return None, failure

def answer_with_router(
    query: str,
    k: int = 5,
    gemini_model: str = "gemini-2.0-flash",
    history: Optional[List[Tuple[str, str]]] = None,
    long_answer: bool = False,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Hàm điều phối chính — được UI gọi.
    """
    # Lazy import
    from .router import route_intent, parse_poem_request, get_chitchat_response
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

    short_history = _history_to_text(history, max_turns=4)
    full_history = _history_to_text(history, max_turns=8)

    if max_tokens is None:
        max_tokens = DEFAULT_LONG_TOKEN_BUDGET if long_answer else DEFAULT_SHORT_TOKEN_BUDGET

    # 1) FAQ
    hit = lookup_faq(query)
    if hit:
        ans = hit["answer"]
        intent = "faq"
        qkey = _make_cache_key(query, long_answer=long_answer, intent=intent)
        set_cached(qkey, ans)
        return {"intent": intent, "answer": ans, "sources": _maybe_sources([])}

    # 2) Route
    intent = route_intent(query)
    qkey = _make_cache_key(query, long_answer=long_answer, intent=intent)

    # 0) Cache sau khi biết intent
    cached = get_cached(qkey)
    if cached:
        return {"intent": "cache", "answer": cached, "sources": _maybe_sources([])}

    # ---- Small talk
    if intent == "chitchat":
        # Trả lời cứng nếu là câu chào đơn giản — không cần gọi LLM
        quick = get_chitchat_response(query)
        if quick:
            set_cached(qkey, quick)
            return {"intent": intent, "answer": quick, "sources": _maybe_sources([])}
        prompt = build_smalltalk_prompt(query, history_text=short_history)
        ans, failure = _safe_generate(
            intent, prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens
        )
        if failure:
            return failure
        set_cached(qkey, ans or "")
        return {"intent": intent, "answer": ans or "", "sources": _maybe_sources([])}

    # ---- Generic factual
    if intent == "generic":
        prompt = build_generic_prompt(
            query,
            history_text=full_history,
            depth="expanded" if long_answer else "balanced",
        )
        ans, failure = _safe_generate(
            intent, prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens
        )
        if failure:
            return failure
        set_cached(qkey, ans or "")
        return {"intent": intent, "answer": ans or "", "sources": _maybe_sources([])}

    # ---- Poem mode
    if intent == "poem":
        if not poem_ready():
            msg = "Kho thơ chưa sẵn sàng (cần data/interim/poem/poem.txt, mỗi câu 1 dòng)."
            set_cached(qkey, msg)
            return {"intent": "poem", "answer": msg, "sources": _maybe_sources([])}

        spec = parse_poem_request(query)
        if spec:
            kind = spec[0]
            if kind == "opening":
                n = max(1, min(int(spec[1]), 1500))
                lines = get_opening(n)
                txt = "\n".join(f"{i + 1:>4}: {ln}" for i, ln in enumerate(lines))
                ans = f"**{n} câu đầu Truyện Kiều:**\n\n{txt}"
                set_cached(qkey, ans)
                return {"intent": "poem", "answer": ans, "sources": _maybe_sources([])}

            if kind == "range":
                a, b = int(spec[1]), int(spec[2])
                if a > b:
                    a, b = b, a
                lines = get_range(a, b)
                txt = "\n".join(f"{a + i:>4}: {ln}" for i, ln in enumerate(lines))
                ans = f"**Các câu {a}–{b} trong Truyện Kiều:**\n\n{txt}"
                set_cached(qkey, ans)
                return {"intent": "poem", "answer": ans, "sources": _maybe_sources([])}

            if kind == "single":
                n = int(spec[1])
                ln = get_single(n)
                if ln:
                    ans = f"**Câu {n} trong Truyện Kiều:**\n\n{n:>4}: {ln}"
                else:
                    ans = f"Chưa tra được câu {n} (vượt ngoài số dòng hiện có)."
                set_cached(qkey, ans)
                return {"intent": "poem", "answer": ans, "sources": _maybe_sources([])}

            if kind == "compare":
                a, b = int(spec[1]), int(spec[2])
                line_a, line_b = compare_lines(a, b)
                if not line_a or not line_b:
                    ans = "Không đủ dữ liệu để so sánh hai câu được yêu cầu."
                    set_cached(qkey, ans)
                    return {"intent": "poem", "answer": ans, "sources": _maybe_sources([])}
                prompt = build_poem_compare_prompt(
                    query,
                    line_a=line_a,
                    line_b=line_b,
                    history_text=short_history,
                )
                ans, failure = _safe_generate(
                    "poem",
                    prompt,
                    model=gemini_model,
                    long_answer=long_answer,
                    max_tokens=max_tokens,
                    sources=[f"câu {line_a.number}", f"câu {line_b.number}"],
                )
                if failure:
                    return failure
                verification = verify_poem_quotes(ans or "")
                set_cached(qkey, ans or "")
                # nguồn luôn rỗng/ẩn
                return {
                    "intent": "poem",
                    "answer": ans or "",
                    "sources": _maybe_sources([f"câu {line_a.number}", f"câu {line_b.number}"]),
                    "verification": verification,
                }

        # Không parse được — nhờ model hỏi lại ngắn
        prompt = build_poem_disambiguation_prompt(query, history_text=short_history)
        ans, failure = _safe_generate(
            "poem", prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens
        )
        if failure:
            return failure
        verification = verify_poem_quotes(ans or "")
        set_cached(qkey, ans or "")
        return {"intent": "poem", "answer": ans or "", "sources": _maybe_sources([]), "verification": verification}

    # ---- Domain → RAG
    poem_only = _needs_poem_only(query)
    close_reading = _is_close_reading(query)
    is_char_who  = _who_is_character(query)


    pack = answer_question(
        query,
        k=k,
        synthesize="single",
        gen_model=gemini_model,
        force_quote=not is_char_who,
        
        long_answer=long_answer,
        history_text=full_history,
        max_tokens=max_tokens,
        # Hints cho RAG pipeline
        prefer_poem_source=poem_only,
        top_evidence=6,
        essay_mode=("hsg" if close_reading else None),
    )

    if pack.get("generation_error"):
        return _generation_failure_response("domain", str(pack["generation_error"]))

    ans = pack.get("answer")
    sources = pack.get("sources", [])
    evidence = pack.get("evidence", [])
    verification = verify_poem_quotes(ans or "") if ans else None

    bad_count = 0
    if isinstance(verification, dict):
        bad_count = len(verification.get("invalid_quotes", [])) + len(verification.get("non_exact", []))

    if ans:
        if poem_only and bad_count >= 2:
            ans += (
                "\n\n**Lưu ý:** Phát hiện vài trích dẫn chưa khớp nguyên văn. "
                "Bạn có thể yêu cầu: `trích câu n–m` để xem bản gốc."
            )
        set_cached(qkey, ans or "")
        return {
            "intent": "domain",
            "answer": ans or "",
            "sources": _maybe_sources(sources),  # sẽ là [] nếu không bật TKC_SHOW_SOURCES
            "verification": verification,
            "evidence": evidence,
        }
        
    if not ans and is_char_who:
        from .prompt_engineering import build_generic_prompt
        hint = "Giải thích ngắn gọn nhân vật trong Truyện Kiều (kiến thức phổ thông, không cần trích dẫn)."
        prompt = build_generic_prompt(f"{query}\n\n{hint}", history_text=full_history, depth="balanced")
        ans2, failure2 = _safe_generate("domain", prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens)
        if not failure2 and ans2:
            set_cached(qkey, ans2 or "")
            return {
                "intent": "domain",
                "answer": ans2 or "",
                "sources": _maybe_sources([]),  # vẫn ẩn nguồn như trước
                "verification": verify_poem_quotes(ans2 or ""),
            }
    

    # Fallback — dùng prompt đã build (nếu có)
    p = pack.get("prompt", "")
    if not isinstance(p, str):
        p = str(p)
    ans, failure = _safe_generate(
        "domain",
        p,
        model=gemini_model,
        long_answer=long_answer,
        max_tokens=max_tokens,
    )
    if failure:
        return failure
    verification = verify_poem_quotes(ans or "")
    set_cached(qkey, ans or "")
    return {
        "intent": "domain",
        "answer": ans or "",
        "sources": _maybe_sources(pack.get("sources", [])),
        "verification": verification,
    }
