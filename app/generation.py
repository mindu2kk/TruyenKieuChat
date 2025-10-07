# app/generation.py
# -*- coding: utf-8 -*-
"""
Gateway gọi Gemini với cấu hình linh hoạt:
- Tăng max_output_tokens để trả lời dài.
- Prompt sạch, kiểm soát nhiệt độ.
"""

from __future__ import annotations
import os, textwrap
from dotenv import load_dotenv

load_dotenv()
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Mặc định ổn cho luận văn ngắn: 0.6 / 0.9
DEFAULT_GENCFG = dict(
    temperature=0.6,
    top_p=0.9,
    top_k=40,
    candidate_count=1,
    max_output_tokens=2048,  # bạn có thể đẩy lên 4096/6144, model sẽ tự giới hạn nếu vượt
)

def generate_answer_gemini(
    prompt: str,
    model: str = "gemini-2.0-flash",
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    long_answer: bool = False,
) -> str:
    if not GOOGLE_API_KEY:
        return "⚠️ Chưa cấu hình GOOGLE_API_KEY."

    gen_cfg = DEFAULT_GENCFG.copy()
    if max_output_tokens: gen_cfg["max_output_tokens"] = int(max_output_tokens)
    if temperature is not None: gen_cfg["temperature"] = float(temperature)
    if top_p is not None: gen_cfg["top_p"] = float(top_p)

    # Nếu bật long_answer, nới rộng thêm 1 chút
    if long_answer and gen_cfg["max_output_tokens"] < 3072:
        gen_cfg["max_output_tokens"] = 3072

    try:
        gm = genai.GenerativeModel(model_name=model, generation_config=gen_cfg)
        res = gm.generate_content(prompt)
        # SDK mới: .text luôn gom chuỗi tiện dụng
        txt = getattr(res, "text", None)
        if txt: 
            return txt.strip()
        # Fallback (các version cũ)
        if hasattr(res, "candidates") and res.candidates:
            parts = []
            for c in res.candidates:
                ct = getattr(getattr(c, "content", None), "parts", [])
                for p in ct or []:
                    s = getattr(p, "text", "")
                    if s: parts.append(s)
            if parts:
                return "\n".join(parts).strip()
        return "⚠️ Không nhận được nội dung từ mô hình."
    except Exception as e:
        return f"⚠️ Lỗi gọi mô hình: {e}"
