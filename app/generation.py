# -*- coding: utf-8 -*-
import os
import re
from typing import Any, Dict
import google.generativeai as genai

class GenerationError(RuntimeError):
    pass

def is_gemini_configured() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))

try:
    from app.prompt_engineering import DEFAULT_LONG_TOKEN_BUDGET, DEFAULT_SHORT_TOKEN_BUDGET  # type: ignore
except Exception:
    # fallback an toàn nếu thiếu file
    DEFAULT_LONG_TOKEN_BUDGET = 2048
    DEFAULT_SHORT_TOKEN_BUDGET = 768

def _setup() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise GenerationError("Chưa thiết lập GOOGLE_API_KEY nên không thể gọi Gemini.")
    genai.configure(api_key=api_key)

def _postprocess(ans: str) -> str:
    if not ans:
        return ans
    lines = []
    for ln in ans.splitlines():
        if re.match(r"^\s*(Nguồn|Source)\s*:.*$", ln, flags=re.I):
            continue
        lines.append(ln)
    txt = "\n".join(lines).strip()
    txt = re.sub(r"(?:\n\n)+", "\n\n", txt)
    return txt

def _resolve_generation_config(long_answer: bool, max_tokens: int | None) -> Dict[str, Any]:
    resolved_max = max_tokens or (DEFAULT_LONG_TOKEN_BUDGET if long_answer else DEFAULT_SHORT_TOKEN_BUDGET)
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
    res = gm.generate_content(prompt)
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
