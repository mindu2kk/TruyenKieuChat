# -*- coding: utf-8 -*-
"""
Chat UI (Streamlit) cho Kiều Bot
- Sidebar cấu hình
- Chat + nhớ ngắn hạn
- Hiển thị nguồn & thời gian phản hồi
"""
import time
import streamlit as st

try:
    from app.orchestrator import answer_with_router
except Exception:
    from orchestrator import answer_with_router  # type: ignore

st.set_page_config(page_title="Kiều Bot", page_icon="📚", layout="centered")

st.markdown(
    """
    <style>
    .smallcaps { font-variant: small-caps; letter-spacing: .02rem; }
    .meta { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='smallcaps'>📚 Kiều Bot</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Thiết lập")
    k = st.slider("Top-k ngữ cảnh", 3, 8, 5)
    model = st.selectbox("Gemini model", ["gemini-2.0-flash"], index=0)
    long_ans = st.toggle("Trả lời dài (nghị luận hơn)", value=True)
    max_tok = st.slider("Giới hạn tokens đầu ra", 512, 6144, 3072, step=256)

if "chat" not in st.session_state:
    st.session_state.chat = []  # list[(role, text)]

# render lịch sử
for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

# ô nhập
user_msg = st.chat_input("Hỏi về Truyện Kiều…")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    # lấy history ngắn hạn
    history = st.session_state.chat[-12:]

    with st.chat_message("assistant"):
        t0 = time.time()
        ret = answer_with_router(
        user_msg,
        k=k,
        gemini_model=model,
        history=history,
        long_answer=long_ans,
        max_tokens=max_tok,     # <— thêm dòng này
    )
        ans = (ret or {}).get("answer", "Không có phản hồi.")
        st.markdown(ans)
        st.caption(f"⏱ {(time.time()-t0)*1000:.0f} ms")
    st.session_state.chat.append(("assistant", ans))
