# -*- coding: utf-8 -*-
"""
Chat UI cho Kiều Bot (RAG + định tuyến + nhớ ngữ cảnh ngắn hạn)
- Poem mode: trích NGUYÊN VĂN theo số câu/khoảng.
- Domain: RAG + Gemini, có điều khiển max_tokens & long_answer.
- Ẩn nguồn để nhường token cho phần trả lời (theo yêu cầu).
"""

import time
import streamlit as st
from orchestrator import answer_with_router  # điều phối trả lời theo intent

# ====== Cấu hình trang ======
st.set_page_config(page_title="Kiều Bot", page_icon="📚", layout="centered")

st.markdown(
    "<h1 style='text-align:center;margin-top:0'>📚 Kiều Bot</h1>",
    unsafe_allow_html=True
)

# ====== Sidebar: cấu hình ======
with st.sidebar:
    st.header("Thiết lập")
    k = st.slider("Top-k ngữ cảnh (RAG)", 3, 10, 5)
    model = st.selectbox("Gemini model", ["gemini-2.0-flash", "gemini-2.0-flash-lite"], index=0)
    long_ans = st.toggle("Trả lời theo văn phong luận", value=False)
    max_tok = st.slider("Giới hạn token đầu ra", 256, 8192, 2048, step=256)

# ====== Bộ nhớ hội thoại ======
if "chat" not in st.session_state:
    st.session_state.chat = []  # list[(role, text)]

# Hiển thị lịch sử hội thoại
for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

# ====== Ô nhập người dùng ======
user_msg = st.chat_input("Hỏi về Truyện Kiều…", key="chat_input")

if user_msg:
    # hiển thị tin người dùng
    st.session_state.chat.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    # lấy history ngắn hạn (cuối 12 lượt: 6 cặp user/assistant)
    history = st.session_state.chat[-12:]

    # gọi router để trả lời đúng nhánh
    with st.chat_message("assistant"):
        t0 = time.time()
        ret = answer_with_router(
            user_msg,
            k=k,
            gemini_model=model,
            long_answer=long_ans,
            history=history,
            max_tokens=max_tok,   # <— truyền max_tokens xuống
        )
        st.markdown(ret["answer"])
        st.caption(f"⏱️ {(time.time() - t0) * 1000:.0f} ms")

    # lưu câu trả lời vào lịch sử
    st.session_state.chat.append(("assistant", ret["answer"]))
