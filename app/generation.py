# -*- coding: utf-8 -*-
import os
import re
from typing import Any, Dict
import google.generativeai as genai

class GenerationError(RuntimeError):
    """Raised when the Gemini client cannot be used."""

def is_gemini_configured() -> bool:
    """Return True when a GOOGLE_API_KEY is available."""
    return bool(os.getenv("GOOGLE_API_KEY"))

# Optional tuning constants (safe defaults if module not present)
try:
    from app.prompt_engineering import (
        DEFAULT_LONG_TOKEN_BUDGET,
        DEFAULT_SHORT_TOKEN_BUDGET,
    )  # type: ignore
except Exception:
    DEFAULT_LONG_TOKEN_BUDGET = 2048
    DEFAULT_SHORT_TOKEN_BUDGET = 768

def _setup() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise GenerationError("Chưa thiết lập GOOGLE_API_KEY nên không thể gọi Gemini.")
    try:
        genai.configure(api_key=api_key)
    except Exception as exc:
        raise GenerationError(f"Không cấu hình được Gemini client ({exc}).") from exc

def _postprocess(ans: str) -> str:
    if not ans:
        return ans
    # Bỏ mọi dòng model tự thêm "Nguồn:" / "Source:"
    lines = []
    for ln in ans.splitlines():
        if re.match(r"^\s*(Nguồn|Source)\s*:.*$", ln, flags=re.I):
            continue
        lines.append(ln)
    txt = "\n".join(lines).strip()
    # Gọn khoảng trắng
    txt = re.sub(r"(?:\n\n)+", "\n\n", txt)
    return txt

def _resolve_generation_config(long_answer: bool, max_tokens: int | None) -> Dict[str, Any]:
    resolved_max = max_tokens if max_tokens is not None else (
        DEFAULT_LONG_TOKEN_BUDGET if long_answer else DEFAULT_SHORT_TOKEN_BUDGET
    )
    # Tránh truyền giá trị “None” vào SDK
    cfg: Dict[str, Any] = {
        "temperature": 0.6 if long_answer else 0.55,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": int(resolved_max),
        "candidate_count": 1,
    }
    return cfg

def generate_answer_gemini(
    prompt: str,
    model: str = "gemini-2.0-flash",
    long_answer: bool = False,
    max_tokens: int | None = None,
) -> str:
    """
    Gọi Gemini an toàn (không set generation_config ở constructor để tránh bug len(int)).
    """
    _setup()
    generation_config = _resolve_generation_config(long_answer, max_tokens)

    if long_answer:
        prompt = f"""{prompt}

[PHONG CÁCH]
- Văn phong nghị luận mạch lạc (mở–thân–kết).
- Luận điểm → dẫn chứng (trích 1–2 câu thơ khi phù hợp) → phân tích → tiểu kết.
- Diễn đạt mềm mại, tránh liệt kê máy móc; ưu tiên sự sáng rõ và cô đọng.
"""

    gm = genai.GenerativeModel(model_name=model)
    try:
        res = gm.generate_content(
            prompt,
            generation_config=generation_config,  # <— chuyển config sang đây
            safety_settings={
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_LOW_AND_ABOVE",
                "HARM_CATEGORY_HARASSMENT": "BLOCK_LOW_AND_ABOVE",
                "HARM_CATEGORY_SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
            },
        )
    except Exception as exc:
        raise GenerationError(f"Gọi Gemini thất bại ({exc}).") from exc

    # Lấy text robust
    out = ""
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
