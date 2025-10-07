# app/ui_chat.py
# -*- coding: utf-8 -*-
"""
UI chat “đẹp hơn” cho Kiều Bot
- Bong bóng chat + chip nguồn
- Gợi ý prompt khi chưa có hội thoại
- Nút xóa lịch sử + tải transcript
- Tự động truyền history nếu orchestrator hỗ trợ; fallback nếu không
"""

import time, json, io
from datetime import datetime
from pathlib import Path
import streamlit as st

# ====== nhập orchestrator ======
from orchestrator import answer_with_router  # hàm điều phối

# ====== cấu hình trang & CSS ======
st.set_page_config(page_title="Kiều Bot", page_icon="📚", layout="centered")

CUSTOM_CSS = """
<style>
/* nền nhẹ */
.stApp { background: #0b1020; }
.block-container { max-width: 860px; padding-top: 1.5rem; }

/* tiêu đề */
.kieu-title {
  font-size: 28px; font-weight: 700; color: #e8ecff;
  display:flex; gap:.6rem; align-items:center;
}

/* card trạng thái nhỏ */
.status-bar {
  display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.3rem; margin-bottom:.8rem;
  opacity:.9;
}
.badge {
  background: #1a2140; color:#cfd6ff; border:1px solid #2a3366;
  padding: .22rem .5rem; border-radius: 999px; font-size: 12px;
}

/* khung chat */
.chat-bubble {
  padding: .8rem 1rem; border-radius: 14px; line-height: 1.55;
  border: 1px solid rgba(255,255,255,.08);
  box-shadow: 0 4px 14px rgba(0,0,0,.25);
}
.user {
  background: linear-gradient(180deg,#17224a,#121a38);
  color: #e6ebff;
}
.assistant {
  background: #0f1733;
  color: #ecf1ff;
}
.meta-line { font-size: 12px; color: #9fb0ff; margin-top: .5rem; }

/* chip nguồn */
.src-chips { display:flex; gap:.4rem; flex-wrap: wrap; margin-top:.5rem; }
.src-chip {
  font-size: 11px; padding: .18rem .5rem; border-radius: 999px;
  border: 1px dashed #3a4aa0; color:#d8deff; background: rgba(48,66,160,.18);
}

/* hộp gợi ý */
.hints {
  display:grid; grid-template-columns: repeat(2, minmax(0,1fr));
  gap:.5rem; margin-top:.6rem;
}
.hint {
  border:1px solid #304090; color:#dbe2ff; background: rgba(48,64,144,.18);
  border-radius:12px; padding:.6rem .7rem; cursor:pointer; user-select:none;
}
.hint:hover { background: rgba(48,64,144,.30); }

/* thanh divider tinh tế */
hr.soft { border:none; height:1px; background: linear-gradient(90deg, transparent, #2f3a66, transparent); margin:1.1rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ====== header ======
st.markdown(f"""
<div class="kieu-title">📚 Kiều Bot <span style="font-size:16px;font-weight:400;opacity:.8">— Trợ lý Truyện Kiều</span></div>
<div class="status-bar">
  <span class="badge">RAG</span>
  <span class="badge">Router</span>
  <span class="badge">Poem mode</span>
  <span class="badge">Short-term Memory</span>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ====== sidebar ======
with st.sidebar:
    st.header("⚙️ Thiết lập")
    k = st.slider("Top-k ngữ cảnh", 3, 8, 4)
    model = st.selectbox("Model", ["gemini-2.0-flash", "gemini-2.0-flash-lite"], index=0)
    long_ans = st.toggle("Trả lời theo văn nghị luận (dài hơn)", value=False)
    st.caption("Kho tri thức: chỉ dữ liệu bạn đã nạp.")
    st.markdown("---")

    # tác vụ
    colA, colB = st.columns(2)
    with colA:
        if st.button("🧹 Xóa hội thoại", use_container_width=True):
            st.session_state.chat = []
            st.toast("Đã xóa lịch sử.")
            st.rerun()
    with colB:
        # tải transcript
        def _export_chat() -> bytes:
            lines = []
            for role, text, meta in st.session_state.get("chat", []):
                lines.append(f"{role.upper()}:\n{text}\n")
            return "\n".join(lines).encode("utf-8")
        st.download_button("⬇️ Tải transcript", data=_export_chat(),
                           file_name=f"kieu_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                           mime="text/plain", use_container_width=True)
    st.markdown("---")
    st.caption("Mẹo: Dùng câu như “trích 30 câu đầu”, “câu 241–260”, hoặc câu hỏi phân tích (“giải thích tả cảnh ngụ tình…”)")

# ====== state ======
if "chat" not in st.session_state:
    # mỗi item: (role, text, meta_dict)
    st.session_state.chat = []

# ====== helpers ======
def _get_source_chips(ret) -> list[str]:
    """Lấy danh sách nguồn hiển thị đẹp."""
    chips = []
    srcs = ret.get("sources") or []
    # orchestrator của bạn có 2 kiểu: list ctx dicts hoặc list string
    seen = set()
    for s in srcs:
        if isinstance(s, dict):
            src = (s.get("meta") or {}).get("source")
        else:
            src = str(s)
        if src and src not in seen:
            seen.add(src)
            chips.append(src)
    # fallback: cố đọc từ text “**Nguồn:** …”
    if not chips and isinstance(ret.get("answer"), str) and "**Nguồn:**" in ret["answer"]:
        tail = ret["answer"].split("**Nguồn:**", 1)[-1].strip()
        for token in [t.strip() for t in tail.split(";")]:
            if token and token not in seen:
                seen.add(token); chips.append(token)
    return chips

def _render_message(role: str, text: str, chips: list[str] | None = None):
    css_class = "user" if role == "user" else "assistant"
    with st.chat_message(role, avatar="🧑‍💬" if role=="user" else "🤖"):
        st.markdown(f"<div class='chat-bubble {css_class}'>"+text+"</div>", unsafe_allow_html=True)
        if chips:
            st.markdown(
                "<div class='src-chips'>" + "".join([f"<span class='src-chip'>{st.session_state.get('src_prefix','')}</span>".replace(
                    st.session_state.get('src_prefix',''), ch) for ch in chips]) + "</div>",
                unsafe_allow_html=True
            )

def _call_router(query: str, *, k: int, model: str, long_ans: bool, history):
    """Gọi answer_with_router; nếu signature cũ, tự fallback không truyền history."""
    try:
        return answer_with_router(query, k=k, gemini_model=model, long_answer=long_ans, history=history)
    except TypeError:
        # orchestrator cũ chưa nhận history → gọi không có history
        return answer_with_router(query, k=k, gemini_model=model, long_answer=long_ans)

# ====== hiển thị lịch sử ======
for role, text, meta in st.session_state.chat:
    _render_message(role, text, chips=(meta.get("chips") if meta else None))

# ====== gợi ý khi trống ======
if not st.session_state.chat:
    st.info("Gợi ý nhanh (bấm để chèn):")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Trích 20 câu đầu"):
            st.session_state.pending = "Cho tôi 20 câu đầu Truyện Kiều"
            st.rerun()
        if st.button("Giải thích tả cảnh ngụ tình trong Cảnh ngày xuân"):
            st.session_state.pending = "Giải thích tả cảnh ngụ tình trong Cảnh ngày xuân"
            st.rerun()
    with c2:
        if st.button("So sánh vẻ đẹp Thúy Vân – Thúy Kiều"):
            st.session_state.pending = "So sánh vẻ đẹp Thúy Vân và Thúy Kiều trong đoạn Chị em Thúy Kiều"
            st.rerun()
        if st.button("Ý nghĩa 'Chữ tâm kia mới bằng ba chữ tài'"):
            st.session_state.pending = "Ý nghĩa câu 'Chữ tâm kia mới bằng ba chữ tài'"
            st.rerun()

# ====== input ======
default_prefill = st.session_state.pop("pending", None)
user_msg = st.chat_input("Hỏi về Truyện Kiều…", key="chat_input", value=default_prefill or "")

if user_msg:
    # hiển thị người dùng
    st.session_state.chat.append(("user", user_msg, {}))
    _render_message("user", user_msg)

    # gom history ngắn hạn: chỉ text
    short_hist = [(r, t) for (r, t, _) in st.session_state.chat[-12:]]
    # gọi router
    with st.chat_message("assistant", avatar="🤖"):
        t0 = time.time()
        # typing spinner
        with st.spinner("Đang suy nghĩ…"):
            ret = _call_router(user_msg, k=k, model=model, long_ans=long_ans, history=short_hist)
        answer = ret.get("answer", "Xin lỗi, mình chưa trả lời được.")
        # hiển thị
        chips = _get_source_chips(ret)
        st.markdown(f"<div class='chat-bubble assistant'>{answer}</div>", unsafe_allow_html=True)
        if chips:
            st.markdown("<div class='src-chips'>" + "".join([f"<span class='src-chip'>{ch}</span>" for ch in chips]) + "</div>", unsafe_allow_html=True)
        st.caption(f"⏱️ {(time.time() - t0)*1000:.0f} ms")

    # lưu
    st.session_state.chat.append(("assistant", answer, {"chips": chips}))
