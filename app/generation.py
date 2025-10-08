# -*- coding: utf-8 -*-
import os, re
from typing import Any, Dict

import google.generativeai as genai

from prompt_engineering import DEFAULT_LONG_TOKEN_BUDGET, DEFAULT_SHORT_TOKEN_BUDGET

def _setup():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY không tồn tại trong môi trường.")
    genai.configure(api_key=api_key)

def _postprocess(ans: str) -> str:
    if not ans: return ans
    # cắt mọi dòng kiểu "Nguồn:" / "Source:" nếu model tự đẻ ra
    lines = []
    for ln in ans.splitlines():
        if re.match(r"^\s*(Nguồn|Source)\s*:.*$", ln, flags=re.I):
            continue
        lines.append(ln)
    txt = "\n".join(lines).strip()
    # chống trùng lặp đoạn
    txt = re.sub(r"(?:\n\n)+", "\n\n", txt)
    return txt

def _resolve_generation_config(long_answer: bool, max_tokens: int | None) -> Dict[str, Any]:
    resolved_max = max_tokens
    if resolved_max is None:
        resolved_max = DEFAULT_LONG_TOKEN_BUDGET if long_answer else DEFAULT_SHORT_TOKEN_BUDGET

    return {
        "temperature": 0.6 if long_answer else 0.55,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": int(resolved_max),
    }


def generate_answer_gemini(
    prompt: str,
    model: str = "gemini-2.0-flash",
    long_answer: bool = False,
    max_tokens: int | None = None,
) -> str:
    _setup()

    generation_config = _resolve_generation_config(long_answer, max_tokens)

    if long_answer:
        prompt = f"""{prompt}

[PHONG CÁCH]
- Văn phong nghị luận mạch lạc (mở–thân–kết).
- Luận điểm → dẫn chứng (trích 1–2 câu thơ khi phù hợp) → phân tích → tiểu kết.
- Diễn đạt mềm mại, tránh liệt kê máy móc; ưu tiên sự sáng rõ và cô đọng.
"""

    gm = genai.GenerativeModel(model_name=model, generation_config=generation_config)
    res = gm.generate_content(
        prompt,
        safety_settings={
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_LOW_AND_ABOVE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_LOW_AND_ABOVE",
            "HARM_CATEGORY_SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
        },
    )

    try:
        out = res.text
    except Exception:
        try:
            out = "".join(
                (part.text or "") for part in res.candidates[0].content.parts
                if hasattr(part, "text")
            ).strip()
        except Exception:
            out = str(res)
    return _postprocess(out)