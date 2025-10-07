# app/ui_chat.py
# -*- coding: utf-8 -*-
"""
Kiều Bot – Chat UI (RAG + định tuyến + nhớ ngữ cảnh ngắn hạn an toàn)
- Không dùng value= cho st.chat_input (tránh TypeError trên Streamlit Cloud).
- Sidebar: chỉnh k, chọn model, bật/tắt trả lời dài, nút xoá hội thoại.
- Hiển thị lịch sử dạng bong bóng, có đồng hồ latency, và nguồn tham khảo (expander).
- Gợi ý nhanh bằng nút (không cần prefill chat_input).
- Tự động fallback nếu orchestrator.answer_with_router chưa hỗ trợ history=...
"""

from __future__ import annotations
import os
import time
import streamlit as st

# ====== App setup ======
st.set_page_config(page_title="Kiều Bot", page_icon="📚", layout="centered")

# ====== Import orchestrator an toàn ======
try:
    from orchestrator import answer_with_router  # điều phối theo intent
except Exception as e:
    st.error(f"Không import được orchestrator: {e}")
    st.stop()

# ====== Helpers ======
def ensure_state():
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list[(role, text, meta)]
    if "cfg" not in st.session_state:
        st.session_state.cfg = {
            "k": 4,
            "model": "gemini-2.0-flash",
            "long_answer": False,
        }

def push_msg(role: str, text: str, meta: dict | None = None):
    st.session_state.chat.append((role, text, meta or {}))

def render_message(role: str, text: str):
    avatar = "🧑‍💻" if role == "user" else "📚"
    with st.chat_message(role, avatar=avatar):
        st.markdown(text)

def render_sources(sources):
    if not sources:
        return
    # hỗ trợ cả string lẫn list
    if isinstance(sources, str):
        src_list = [s.strip() for s in sources.split(";") if s.strip()]
    elif isinstance(sources, list):
        # nếu list là contexts (dict), ta gắng trích meta.source
        if sources and isinstance(sources[0], dict):
            tmp = []
            for c in sources:
                src = (c.get("meta") or {}).get("source")
                if src and src not in tmp:
                    tmp.append(src)
            src_list = tmp
        else:
            src_list = [str(s) for s in sources]
    else:
        src_list = []
    if not src_list:
        return

    with st.expander("Nguồn / tham chiếu"):
        for s in src_list:
            st.markdown(f"- `{s}`")

def render_header():
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 0.6rem;">
            <h1 style="margin:0;">📚 Kiều Bot</h1>
            <p style="color:#666; margin:0.25rem 0 0;">
                Hỏi–đáp về <em>Truyện Kiều</em> (RAG + trích thơ).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_suggestions():
    st.caption("Gợi ý nhanh:")
    cols = st.columns(3)
    examples = [
        "Cho tôi 10 câu đầu Truyện Kiều",
        "So sánh Thúy Vân và Thúy Kiều",
        "Ý nghĩa câu “Chữ tâm kia mới bằng ba chữ tài”",
    ]
    fired = None
    for i, (c, s) in enumerate(zip(cols, examples)):
        if c.button(s, key=f"suggest_{i}"):
            fired = s
    return fired

# ====== Main UI ======
ensure_state()

with st.sidebar:
    st.header("Thiết lập")
    st.session_state.cfg["k"] = st.slider("Top-k ngữ cảnh", 3, 6, st.session_state.cfg["k"])
    st.session_state.cfg["model"] = st.selectbox(
        "Gemini model",
        ["gemini-2.0-flash", "gemini-2.0-flash-lite"],
        index=0 if st.session_state.cfg["model"] == "gemini-2.0-flash" else 1,
    )
    st.session_state.cfg["long_answer"] = st.toggle("Trả lời dài hơn (nhập vai luận văn)", value=st.session_state.cfg["long_answer"])
    if st.button("🗑️ Xoá hội thoại"):
        st.session_state.chat = []
        st.rerun()

    # cảnh báo thiếu API key cho Gemini
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("Thiếu GOOGLE_API_KEY trong môi trường. Hãy bổ sung để sinh câu trả lời.", icon="⚠️")

render_header()

# Hiển thị lịch sử hội thoại
for role, text, _meta in st.session_state.chat:
    render_message(role, text)

# Gợi ý nhanh
preset = render_suggestions()

# Ô nhập (KHÔNG dùng value= ...)
user_msg = st.chat_input("Hỏi về Truyện Kiều…", key="chat_input")

# Nếu bấm gợi ý nhanh thì ưu tiên dùng gợi ý
if preset and not user_msg:
    user_msg = preset

if user_msg:
    # Hiển thị tin người dùng
    push_msg("user", user_msg)
    render_message("user", user_msg)

    # Lấy history ngắn hạn (cuối 12 message ~ 6 lượt)
    history_pairs = [(r, t) for (r, t, _m) in st.session_state.chat[-12:]]

    # Gọi orchestrator
    with st.chat_message("assistant", avatar="📚"):
        t0 = time.time()
        k = st.session_state.cfg["k"]
        model = st.session_state.cfg["model"]
        long_ans = st.session_state.cfg["long_answer"]

        # Ưu tiên gọi với history; nếu hàm chưa hỗ trợ -> fallback
        try:
            ret = answer_with_router(
                user_msg,
                k=k,
                gemini_model=model,
                long_answer=long_ans,
                history=history_pairs,  # có thể không được hỗ trợ ở phiên bản cũ
            )
        except TypeError:
            # orchestrator cũ chưa có tham số 'history'
            ret = answer_with_router(
                user_msg,
                k=k,
                gemini_model=model,
                long_answer=long_ans,
            )
        except Exception as e:
            st.error(f"Lỗi gọi orchestrator: {e}")
            ret = {"answer": "Xin lỗi, mình gặp sự cố khi xử lý yêu cầu.", "sources": []}

        elapsed = (time.time() - t0) * 1000.0

        ans_text = ret.get("answer") or "Xin lỗi, mình chưa có câu trả lời phù hợp."
        render_message("assistant", ans_text)

        # Nguồn / contexts
        render_sources(ret.get("sources"))

        # Thông tin phụ (latency + intent)
        intent = ret.get("intent", "?")
        st.caption(f"⏱️ {elapsed:.0f} ms • intent: `{intent}`")

    # Lưu trả lời
    push_msg("assistant", ans_text, {"intent": ret.get("intent"), "sources": ret.get("sources")})
