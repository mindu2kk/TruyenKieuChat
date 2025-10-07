# -*- coding: utf-8 -*-
"""
Wrapper gọi Gemini. Cần env: GOOGLE_API_KEY
"""
from __future__ import annotations
import os
import google.generativeai as genai

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

def generate_answer_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str:
    if not API_KEY:
        # Fallback an toàn khi chạy local không có key
        return "Chưa cấu hình GOOGLE_API_KEY nên không thể sinh câu trả lời bằng Gemini. Vui lòng đặt biến môi trường GOOGLE_API_KEY."
    gm = genai.GenerativeModel(model)
    try:
        res = gm.generate_content(prompt)
        return (res.text or "").strip() if hasattr(res, "text") else ""
    except Exception as e:
        return f"Đã lỗi khi gọi Gemini: {e}"
