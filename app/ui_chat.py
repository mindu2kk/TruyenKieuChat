# -*- coding: utf-8 -*-
"""Streamlit chat UI for Kiều Bot."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, MutableMapping, Tuple
import sys
from pathlib import Path
import traceback
import streamlit as st

# === ÉP đường dẫn gốc vào sys.path và import theo package "app" ===
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestrator import answer_with_router  # type: ignore
from app.generation import is_gemini_configured  # type: ignore
from app.poem_tools import poem_ready  # type: ignore

ChatHistory = List[MutableMapping[str, Any]]
RawHistory = List[Tuple[str, str]]


def _init_state() -> None:
    if "chat" not in st.session_state:
        st.session_state.chat = []  # type: ignore[assignment]


def _normalize_history(chat: Iterable[Any]) -> ChatHistory:
    normalized: ChatHistory = []
    for item in chat:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            role, content = item
            normalized.append({"role": role, "content": content})
    return normalized


def _history_for_model(chat: ChatHistory, limit: int = 12) -> RawHistory:
    window = chat[-limit:]
    return [(entry.get("role", "user"), entry.get("content", "")) for entry in window]


def _render_sources(sources: Iterable[str]) -> None:
    chips = [src for src in sources if src]
    if not chips:
        return
    st.caption("📎 Nguồn tham chiếu: " + ", ".join(chips))


def _render_verification(payload: Dict[str, Any]) -> None:
    if not payload:
        return
    quotes: List[Dict[str, Any]] = payload.get("quotes") or []
    accepted: List[Dict[str, Any]] = payload.get("accepted") or []
    coverage = payload.get("coverage")

    accepted_key = {(item.get("quote"), item.get("matched_line")) for item in accepted}

    summary_parts = []
    if coverage is not None:
        summary_parts.append(f"coverage ~{coverage * 100:.0f}%")
    if accepted:
        summary_parts.append(f"{len(accepted)}/{len(quotes) or 1} trích dẫn khớp")
    elif quotes:
        summary_parts.append("chưa khớp trích dẫn nào")

    if summary_parts:
        st.caption("🔍 Kiểm chứng trích dẫn: " + " · ".join(summary_parts))

    if not quotes:
        return

    with st.expander("Chi tiết kiểm chứng", expanded=False):
        for item in quotes:
            quote = item.get("quote", "")
            matched = item.get("matched_text") or "(không trùng khớp)"
            number = item.get("matched_line")
            score = item.get("score", 0.0)
            key = (item.get("quote"), item.get("matched_line"))
            icon = "✅" if key in accepted_key else "⚠️"
            header = f"{icon} "
            if number:
                header += f"Câu {number}"
            else:
                header += "Không rõ câu"
            header += f" · score={score:.1f}"
            st.markdown(f"**{header}**")
            st.markdown(f"• Trích: `{quote}`")
            st.markdown(f"• Khớp: `{matched}`")


def _render_meta(meta: Dict[str, Any], *, debug: bool) -> None:
    if not meta:
        return

    error_detail = meta.get("error")
    if error_detail:
        st.warning("⚠️ Không thể gọi Gemini – kiểm tra GOOGLE_API_KEY.", icon="⚠️")

    if not debug:
        return

    intent = meta.get("intent")
    elapsed_ms = meta.get("elapsed_ms")
    if intent or elapsed_ms is not None:
        parts = []
        if intent:
            parts.append(f"🧭 Intent: `{intent}`")
        if elapsed_ms is not None:
            parts.append(f"⏱️ {elapsed_ms:.0f} ms")
        if parts:
            st.caption(" · ".join(parts))

    _render_sources(meta.get("sources") or [])
    verification = meta.get("verification")
    if isinstance(verification, dict):
        _render_verification(verification)


def _clear_chat() -> None:
    st.session_state.chat = []  # type: ignore[assignment]


st.set_page_config(page_title="Kiều Bot", page_icon="📚", layout="centered")

# ===== header =====
st.markdown(
    """
    <div style="text-align:center;margin-top:-20px">
      <h1 style="margin-bottom:4px">📚 Kiều Bot</h1>
      <p style="color:#666;margin-top:0">Tra thơ chuẩn, phân tích mượt, không rườm rà nguồn.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Thiết lập")
    k = st.slider("Top-k ngữ cảnh", 3, 8, 5)
    model = st.selectbox("Gemini model", ["gemini-2.0-flash"], index=0)
    long_ans = st.toggle("Văn phong luận văn (dài hơn)", value=True)
    max_tok = st.slider("Giới hạn độ dài trả lời (tokens)", 256, 8096, 1024, step=128)
    debug_meta = st.toggle("Hiển thị intent & kiểm chứng", value=True)
    if st.button("🧹 Xóa hội thoại", use_container_width=True):
        _clear_chat()
        st.rerun()

if not is_gemini_configured():
    st.info(
        "🔑 Chưa thấy GOOGLE_API_KEY. Một số câu trả lời sẽ lỗi cho đến khi bạn cấu hình khóa Gemini.",
        icon="ℹ️",
    )
if not poem_ready():
    st.info(
        "📜 Kho thơ chưa sẵn sàng (thiếu data/interim/poem). Các câu hỏi về thơ sẽ dùng dữ liệu tối giản.",
        icon="ℹ️",
    )

_init_state()
st.session_state.chat = _normalize_history(st.session_state.chat)  # type: ignore[attr-defined]
chat: ChatHistory = st.session_state.chat  # type: ignore[assignment]

# history view
for item in chat:
    role = item.get("role", "assistant")
    content = item.get("content", "")
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            _render_meta(item.get("meta") or {}, debug=debug_meta)

# input
user_msg = st.chat_input("Hỏi về Truyện Kiều…")
if user_msg:
    chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    history = _history_for_model(chat, limit=12)

    with st.chat_message("assistant"):
        t0 = time.time()
        try:
            ret = answer_with_router(
                user_msg,
                k=k,
                gemini_model=model,
                history=history,
                long_answer=long_ans,
                max_tokens=max_tok,
            )
        except Exception as exc:
            st.error(f"Lỗi khi trả lời: {exc}")
            st.code(traceback.format_exc())
            ret = {"answer": "Xin lỗi, có lỗi kỹ thuật khi xử lý câu hỏi."}

        elapsed_ms = (time.time() - t0) * 1000
        answer_text = ret.get("answer", "(không có câu trả lời)")
        st.markdown(answer_text)

        meta = {
            "intent": ret.get("intent"),
            "sources": ret.get("sources") or [],
            "verification": ret.get("verification"),
            "elapsed_ms": elapsed_ms,
            "error": ret.get("error"),
        }
        _render_meta(meta, debug=debug_meta)

    chat.append({"role": "assistant", "content": answer_text, "meta": meta})
