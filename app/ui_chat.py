# -*- coding: utf-8 -*-
import time
import streamlit as st
from orchestrator import answer_with_router

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
    model = st.selectbox("Gemini model", ["gemini-2.0-flash", "gemini-2.0-flash-lite"], index=0)
    long_ans = st.toggle("Văn phong luận văn (dài hơn)", value=True)
    max_tok = st.slider("Giới hạn độ dài trả lời (tokens)", 256, 8096, 1024, step=128)

# ===== memory =====
if "chat" not in st.session_state:
    st.session_state.chat = []  # [(role, text)]

# history view
for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

# input
user_msg = st.chat_input("Hỏi về Truyện Kiều…")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    history = st.session_state.chat[-12:]  # ngắn hạn

    with st.chat_message("assistant"):
        t0 = time.time()
        ret = answer_with_router(
            user_msg,
            k=k,
            gemini_model=model,
            history=history,
            long_answer=long_ans,
            max_tokens=max_tok,
        )
        st.markdown(ret["answer"])
        st.caption(f"⏱️ {(time.time() - t0) * 1000:.0f} ms")

    st.session_state.chat.append(("assistant", ret["answer"]))
