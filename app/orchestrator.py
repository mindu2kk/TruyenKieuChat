# app/orchestrator.py
# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple, Optional
import os
from dataclasses import replace

# Bật debug (in kèm một ít metadata khi lỗi) bằng cách đặt biến môi trường: DEBUG_ORCH=1
_DEBUG_ORCH = os.getenv("DEBUG_ORCH", "0") == "1"

# Mặc định ẩn nguồn; đặt TKC_SHOW_SOURCES=1 để bật lại
_SHOW_SOURCES = os.getenv("TKC_SHOW_SOURCES", "0").strip().lower() in {"1", "true", "yes", "on"}


def _maybe_sources(srcs: Optional[List[str]]) -> List[str]:
    if not _SHOW_SOURCES:
        return []
    return list(srcs or [])


import re

_CHAR_NAMES = [
    "thúy kiều",
    "thuy kieu",
    "thúy vân",
    "thuy van",
    "kim trọng",
    "kim trong",
    "từ hải",
    "tu hai",
    "hoạn thư",
    "hoan thu",
    "mã giám sinh",
    "ma giam sinh",
    "thúc sinh",
    "thuc sinh",
    "bạc bà",
    "bac ba",
    "bạc hạnh",
    "bac hanh",
]


def _who_is_character(q: str) -> bool:
    ql = (q or "").lower().strip()
    if not re.search(r"\b(là ai|la ai)\b", ql):
        return False
    return any(name in ql for name in _CHAR_NAMES)


# ==== Heuristics cho close-reading & poem-only ====
_TRICH_DAN_TRIGGER = [
    "trích",
    "câu thơ",
    "nguyên văn",
    "dẫn",
    "lục bát",
    "nhịp",
    "vần",
    "điệp",
    "đối",
    "Lầu Ngưng Bích",
    "Đoạn trường",
]
_CLOSE_READING_TRIGGER = [
    "trữ tình ngoại đề",
    "điểm nhìn",
    "ẩn dụ",
    "nhịp điệu",
    "mapping",
    "bản đồ ý niệm",
    "close reading",
]


def _needs_poem_only(q: str) -> bool:
    from .router import requests_poem_evidence

    return requests_poem_evidence(q)


def _is_close_reading(q: str) -> bool:
    ql = (q or "").lower()
    return any(t.lower() in ql for t in _CLOSE_READING_TRIGGER)


def _norm_key(q: str) -> str:
    return (q or "").strip().lower()


def _make_cache_key(q: str, *, long_answer: bool, intent: str, max_tokens: Optional[int] = None) -> str:
    return f"{_norm_key(q)}|la={int(bool(long_answer))}|tokens={max_tokens or 0}|intent={intent}"


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
        "🤖 Xin lỗi, hệ thống chưa thể gọi mô hình để tạo câu trả lời. " "Vui lòng kiểm tra API key và kết nối mạng."
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
    gemini_model: Optional[str] = None,
    history: Optional[List[Tuple[str, str]]] = None,
    long_answer: bool = False,
    max_tokens: Optional[int] = None,
    response_length: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Hàm điều phối chính — được UI gọi.
    """
    # Keep the verified FAQ path genuinely cheap: do not import the RAG stack,
    # reranker, Gemini SDK or poem corpus until the query actually needs them.
    from .faq import lookup_faq
    from .cache import get_cached, set_cached
    from .router import RouteDecision, route_intent, route_query
    from .answer_harness import deterministic_quality, route_metadata
    from .token_budget import plan_token_budget

    decision = route_query(query)
    intent = route_intent(query)
    if intent != decision.intent:
        decision = replace(decision, intent=intent, flow=intent, reason="intent-override")
    budget_plan = plan_token_budget(
        query,
        decision,
        long_answer=long_answer,
        requested_max_tokens=max_tokens,
        response_length=response_length,
    )
    max_tokens = budget_plan.max_output_tokens
    long_answer = long_answer or budget_plan.long_form

    def _route_meta(quality):
        return route_metadata(decision, quality, budget_plan.as_dict())

    # 1) FAQ
    hit = lookup_faq(query)
    if hit:
        ans = hit["answer"]
        intent = "faq"
        qkey = _make_cache_key(query, long_answer=long_answer, intent=intent, max_tokens=max_tokens)
        set_cached(qkey, ans)
        decision = RouteDecision(intent, "verified-faq", 1.0, "faq-match")
        budget_plan = plan_token_budget(query, decision)
        return {
            "intent": intent,
            "answer": ans,
            "sources": _maybe_sources([]),
            "harness": _route_meta(deterministic_quality()),
        }

    from .router import get_chitchat_response, parse_poem_request
    from .rag_pipeline import answer_question
    from .poem_tools import poem_ready, get_opening, get_range, get_single, compare_lines
    from .prompt_engineering import (
        build_generic_prompt,
        build_grounded_poem_explanation_prompt,
        build_poem_disambiguation_prompt,
        build_smalltalk_prompt,
        build_poem_compare_prompt,
    )
    from .answer_harness import (
        curated_poem_explanation,
        grounded_quality,
        is_refusal_answer,
        out_of_scope_answer,
        verified_core_answer,
        verify_generated_answer,
    )

    short_history = _history_to_text(history, max_turns=4)
    full_history = _history_to_text(history, max_turns=8)

    gemini_model = (gemini_model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()

    # 2) Route was resolved before loading the heavy RAG stack so the token
    # planner can select a budget from the actual flow.
    qkey = _make_cache_key(query, long_answer=long_answer, intent=intent, max_tokens=max_tokens)

    def _verify_generated(candidate: str, *, require_exact_quotes: bool, has_evidence: bool):
        return verify_generated_answer(
            candidate,
            require_exact_quotes=require_exact_quotes,
            has_evidence=has_evidence,
            query=query,
        )

    def _cache_verified(answer: str, quality) -> None:
        cacheable_statuses = {
            "cached",
            "generated",
            "grounded",
            "verified",
            "verified-poem-analysis",
            "verified-poem-text",
        }
        if quality.status in cacheable_statuses:
            set_cached(qkey, answer)

    def _finish_exact_poem_answer(base_answer: str, poem_text: str):
        quality = deterministic_quality("verified-poem-text")
        if decision.flow != "grounded-poem-analysis":
            return base_answer, quality

        curated = curated_poem_explanation(poem_text)
        if curated:
            return (
                f"{base_answer}\n\n**Giải thích ngắn:**\n\n{curated}",
                deterministic_quality("verified-poem-analysis"),
            )

        prompt = build_grounded_poem_explanation_prompt(
            query,
            poem_text=poem_text,
            history_text=short_history,
        )
        explanation, failure = _safe_generate(
            "poem",
            prompt,
            model=gemini_model,
            long_answer=False,
            max_tokens=max_tokens,
        )
        if failure or not explanation:
            return base_answer, quality
        checked, _verification, quality = _verify_generated(
            explanation,
            require_exact_quotes=False,
            has_evidence=True,
        )
        return f"{base_answer}\n\n**Giải thích ngắn:**\n\n{checked}", quality

    if intent == "core_fact":
        verified = verified_core_answer(query)
        if verified:
            set_cached(qkey, verified)
            return {
                "intent": intent,
                "answer": verified,
                "sources": _maybe_sources([]),
                "harness": _route_meta(deterministic_quality()),
            }

    if intent == "out_of_scope":
        answer = out_of_scope_answer()
        set_cached(qkey, answer)
        return {
            "intent": intent,
            "answer": answer,
            "sources": _maybe_sources([]),
            "harness": _route_meta(deterministic_quality("safe-refusal")),
        }

    # 0) Cache sau khi biết intent
    cached = get_cached(qkey)
    if cached:
        return {
            "intent": "cache",
            "answer": cached,
            "sources": _maybe_sources([]),
            "harness": _route_meta(deterministic_quality("cached")),
        }

    # ---- Small talk
    if intent == "chitchat":
        # Trả lời cứng nếu là câu chào đơn giản — không cần gọi LLM
        quick = get_chitchat_response(query)
        if quick:
            set_cached(qkey, quick)
            return {
                "intent": intent,
                "answer": quick,
                "sources": _maybe_sources([]),
                "harness": _route_meta(deterministic_quality()),
            }
        prompt = build_smalltalk_prompt(query, history_text=short_history)
        ans, failure = _safe_generate(
            intent, prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens
        )
        if failure:
            return failure
        set_cached(qkey, ans or "")
        return {
            "intent": intent,
            "answer": ans or "",
            "sources": _maybe_sources([]),
            "harness": _route_meta(deterministic_quality("generated")),
        }

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
        return {
            "intent": intent,
            "answer": ans or "",
            "sources": _maybe_sources([]),
            "harness": _route_meta(deterministic_quality("generated")),
        }

    # ---- Poem mode
    if intent == "poem":
        if not poem_ready():
            msg = "Kho thơ chưa sẵn sàng (cần data/interim/poem/poem.txt, mỗi câu 1 dòng)."
            set_cached(qkey, msg)
            return {
                "intent": "poem",
                "answer": msg,
                "sources": _maybe_sources([]),
                "harness": _route_meta(grounded_quality(has_evidence=False)),
            }

        spec = parse_poem_request(query)
        if spec:
            kind = spec[0]
            if kind == "opening":
                n = max(1, min(int(spec[1]), 1500))
                lines = get_opening(n)
                txt = "\n".join(f"{i + 1:>4}: {ln}" for i, ln in enumerate(lines))
                ans, quality = _finish_exact_poem_answer(
                    f"**{n} câu đầu Truyện Kiều:**\n\n{txt}",
                    "\n".join(lines),
                )
                _cache_verified(ans, quality)
                return {
                    "intent": "poem",
                    "answer": ans,
                    "sources": _maybe_sources([]),
                    "harness": _route_meta(quality),
                }

            if kind == "range":
                a, b = int(spec[1]), int(spec[2])
                if a > b:
                    a, b = b, a
                lines = get_range(a, b)
                expected_count = b - a + 1
                if len(lines) != expected_count:
                    ans = (
                        f"Chưa tra đủ các câu {a}–{b} trong bản thơ hiện có "
                        f"(tìm thấy {len(lines)}/{expected_count} câu)."
                    )
                    quality = deterministic_quality("not-found")
                    return {
                        "intent": "poem",
                        "answer": ans,
                        "sources": _maybe_sources([]),
                        "harness": _route_meta(quality),
                    }
                txt = "\n".join(f"{a + i:>4}: {ln}" for i, ln in enumerate(lines))
                ans, quality = _finish_exact_poem_answer(
                    f"**Các câu {a}–{b} trong Truyện Kiều:**\n\n{txt}",
                    "\n".join(lines),
                )
                _cache_verified(ans, quality)
                return {
                    "intent": "poem",
                    "answer": ans,
                    "sources": _maybe_sources([]),
                    "harness": _route_meta(quality),
                }

            if kind == "single":
                n = int(spec[1])
                ln = get_single(n)
                if ln:
                    ans, quality = _finish_exact_poem_answer(
                        f"**Câu {n} trong Truyện Kiều:**\n\n{n:>4}: {ln}",
                        ln,
                    )
                else:
                    ans = f"Chưa tra được câu {n} (vượt ngoài số dòng hiện có)."
                    quality = deterministic_quality("not-found")
                _cache_verified(ans, quality)
                return {
                    "intent": "poem",
                    "answer": ans,
                    "sources": _maybe_sources([]),
                    "harness": _route_meta(quality),
                }

            if kind == "compare":
                a, b = int(spec[1]), int(spec[2])
                line_a, line_b = compare_lines(a, b)
                if not line_a or not line_b:
                    ans = "Không đủ dữ liệu để so sánh hai câu được yêu cầu."
                    set_cached(qkey, ans)
                    return {
                        "intent": "poem",
                        "answer": ans,
                        "sources": _maybe_sources([]),
                        "harness": _route_meta(grounded_quality(has_evidence=False)),
                    }
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
                checked, verification, quality = _verify_generated(
                    ans or "", require_exact_quotes=True, has_evidence=True
                )
                _cache_verified(checked, quality)
                # nguồn luôn rỗng/ẩn
                return {
                    "intent": "poem",
                    "answer": checked,
                    "sources": _maybe_sources([f"câu {line_a.number}", f"câu {line_b.number}"]),
                    "verification": verification,
                    "harness": _route_meta(quality),
                }

        # Không parse được — nhờ model hỏi lại ngắn
        prompt = build_poem_disambiguation_prompt(query, history_text=short_history)
        ans, failure = _safe_generate(
            "poem", prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens
        )
        if failure:
            return failure
        set_cached(qkey, ans or "")
        return {
            "intent": "poem",
            "answer": ans or "",
            "sources": _maybe_sources([]),
            "harness": _route_meta(deterministic_quality("clarification")),
        }

    # ---- Domain → RAG
    poem_only = decision.requires_poem_evidence or _needs_poem_only(query)
    close_reading = _is_close_reading(query)
    is_char_who = _who_is_character(query)

    pack = answer_question(
        query,
        k=k,
        synthesize="single",
        gen_model=gemini_model,
        # Do not force a quotation for every factual question. It made normal
        # answers verbose and increased the chance of a fabricated verse.
        force_quote=decision.requires_exact_quotes,
        long_answer=long_answer,
        history_text=full_history,
        max_tokens=max_tokens,
        # Hints cho RAG pipeline
        prefer_poem_source=poem_only,
        top_evidence=6,
        essay_mode=("hsg" if close_reading and long_answer else None),
    )

    if pack.get("generation_error"):
        return _generation_failure_response("domain", str(pack["generation_error"]))

    ans = pack.get("answer")
    sources = pack.get("sources", [])
    evidence = pack.get("evidence", [])

    # Evidence was retrieved, so a generic refusal is often a generation
    # failure rather than a real absence of information. Retry once with a
    # direct evidence-first contract, but never bypass exact-poem safeguards.
    if ans and evidence and not decision.requires_exact_quotes and is_refusal_answer(str(ans)):
        retry_prompt = str(pack.get("prompt") or "").strip()
        if retry_prompt:
            retry_prompt += (
                "\n\n[KHÔI PHỤC CÂU TRẢ LỜI CÓ EVIDENCE]\n"
                "Các đoạn EVIDENCE đã được tìm thấy. Hãy đọc lại và trả lời trực tiếp câu hỏi bằng dữ kiện có trong đó. "
                "Không nói về quá trình tìm kiếm, corpus hay việc xác minh. Không trích thơ nếu câu hỏi không yêu cầu."
            )
            retried, retry_failure = _safe_generate(
                intent,
                retry_prompt,
                model=gemini_model,
                long_answer=long_answer,
                max_tokens=max_tokens,
            )
            if not retry_failure and retried and not is_refusal_answer(retried):
                ans = retried

    if ans:
        checked, verification, quality = _verify_generated(
            ans,
            require_exact_quotes=decision.requires_exact_quotes,
            has_evidence=bool(evidence),
        )
        if quality.status == "incomplete":
            repair_prompt = str(pack.get("prompt") or "").strip()
            if repair_prompt:
                repair_prompt += (
                    "\n\n[KIỂM TRA HOÀN CHỈNH]\n"
                    "Phản hồi trước đã kết thúc giữa chừng. Viết lại từ đầu, hoàn thành mọi ý được yêu cầu; "
                    "không để tiêu đề, dấu hai chấm hoặc số thứ tự đứng một mình ở cuối."
                )
                repaired, repair_failure = _safe_generate(
                    intent,
                    repair_prompt,
                    model=gemini_model,
                    long_answer=long_answer,
                    max_tokens=max(int(max_tokens), 640),
                )
                if not repair_failure and repaired:
                    repaired_checked, repaired_verification, repaired_quality = _verify_generated(
                        repaired,
                        require_exact_quotes=decision.requires_exact_quotes,
                        has_evidence=bool(evidence),
                    )
                    if repaired_quality.status != "incomplete":
                        checked, verification, quality = (
                            repaired_checked,
                            repaired_verification,
                            repaired_quality,
                        )
        _cache_verified(checked, quality)
        return {
            "intent": intent,
            "answer": checked,
            "sources": _maybe_sources(sources),  # sẽ là [] nếu không bật TKC_SHOW_SOURCES
            "verification": verification,
            "evidence": evidence,
            "harness": _route_meta(quality),
        }

    if not ans and is_char_who:
        from .prompt_engineering import build_generic_prompt

        hint = "Giải thích ngắn gọn nhân vật trong Truyện Kiều (kiến thức phổ thông, không cần trích dẫn)."
        prompt = build_generic_prompt(f"{query}\n\n{hint}", history_text=full_history, depth="balanced")
        ans2, failure2 = _safe_generate(
            "domain", prompt, model=gemini_model, long_answer=long_answer, max_tokens=max_tokens
        )
        if not failure2 and ans2:
            checked, verification, quality = _verify_generated(
                ans2,
                require_exact_quotes=False,
                has_evidence=False,
            )
            _cache_verified(checked, quality)
            return {
                "intent": intent,
                "answer": checked,
                "sources": _maybe_sources([]),  # vẫn ẩn nguồn như trước
                "verification": verification,
                "harness": _route_meta(quality),
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
    checked, verification, quality = _verify_generated(
        ans or "",
        require_exact_quotes=decision.requires_exact_quotes,
        has_evidence=False,
    )
    _cache_verified(checked, quality)
    return {
        "intent": intent,
        "answer": checked,
        "sources": _maybe_sources(pack.get("sources", [])),
        "verification": verification,
        "harness": _route_meta(quality),
    }
