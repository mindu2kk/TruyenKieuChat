# -*- coding: utf-8 -*-
import os
import google.generativeai as genai

def _setup():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY không tồn tại trong môi trường.")
    genai.configure(api_key=api_key)

def generate_answer_gemini(
    prompt: str,
    model: str = "gemini-2.0-flash",
    long_answer: bool = False,
    max_tokens: int | None = None,
) -> str:
    _setup()
    generation_config = {}
    if max_tokens:
        # Theo SDK Gemini: khóa là max_output_tokens
        generation_config["max_output_tokens"] = int(max_tokens)

    # Tăng tính mạch lạc khi long_answer=True
    if long_answer:
        prompt = f"""{prompt}

[PHONG CÁCH]
- Viết mạch lạc, có mở–thân–kết.
- Ưu tiên lập luận chặt chẽ, có dẫn chứng ngắn gọn (nếu NGỮ CẢNH cung cấp).
"""

    gm = genai.GenerativeModel(model_name=model, generation_config=generation_config)
    res = gm.generate_content(prompt)

    # Trích text an toàn theo nhiều phiên bản SDK
    try:
        return res.text
    except Exception:
        try:
            return "".join(
                (part.text or "") for part in res.candidates[0].content.parts
                if hasattr(part, "text")
            ).strip()
        except Exception:
            return str(res)
