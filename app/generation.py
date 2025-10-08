# -*- coding: utf-8 -*-
import os
import re
from typing import Any, Dict, Optional

import google.generativeai as genai


class GenerationError(RuntimeError):
    """Raised when the Gemini client cannot be used."""


def is_gemini_configured() -> bool:
    """Return True when a GOOGLE_API_KEY is available."""
    return bool(os.getenv("GOOGLE_API_KEY"))


try:  # pragma: no cover - support package/script usage
    from .prompt_engineering import DEFAULT_LONG_TOKEN_BUDGET, DEFAULT_SHORT_TOKEN_BUDGET
except ImportError:  # pragma: no cover
    from prompt_engineering import DEFAULT_LONG_TOKEN_BUDGET, DEFAULT_SHORT_TOKEN_BUDGET  # type: ignore


def _setup() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise GenerationError("Chưa thiết lập GOOGLE_API_KEY nên không thể gọi Gemini.")
    try:
        genai.configure(api_key=api_key)
    except Exception as exc:  # pragma: no cover - network/runtime error guard
        raise GenerationError(f"Không cấu hình được Gemini client ({exc}).") from exc


def _postprocess(ans: str) -> str:
    if not ans:
        return ans
    # cắt mọi dòng kiểu "Nguồn:" / "Source:" nếu model tự sinh
    lines = []
    for ln in ans.splitlines():
        if re.match(r"^\s*(Nguồn|Source)\s*:.*$", ln, flags=re.I):
            continue
        lines.append(ln)
    txt = "\n".join(lines).strip()
    # chống lặp khối trắng
    txt = re.sub(r"(?:\n\n)+", "\n\n", txt)
    return txt


def _resolve_generation_config(long_answer: bool, max_tokens: Optional[int]) -> Dict[str, Any]:
    resolved_max = max_tokens
    if resolved_max is None:
        resolved_max = DEFAULT_LONG_TOKEN_BUDGET if long_answer else DEFAULT_SHORT_TOKEN_BUDGET
    try:
        resolved_max = int(resolved_max)
    except Exception as e:
        raise GenerationError(f"max_tokens không hợp lệ: {resolved_max!r} ({type(resolved_max).__name__})") from e

    return {
        "temperature": 0.6 if long_answer else 0.55,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": resolved_max,
    }


def generate_answer_gemini(
    prompt: str,
    model: str = "gemini-2.0-flash",
    long_answer: bool = False,
    max_tokens: Optional[int] = None,
) -> str:
    try:
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

        try:
            res = gm.generate_content(
                prompt,
                safety_settings={
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_LOW_AND_ABOVE",
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_LOW_AND_ABOVE",
                    "HARM_CATEGORY_SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
                },
            )
        except TypeError as exc:
            # 👉 thường gặp khi có biến sai kiểu ở phía trên (UI truyền nhầm), hoặc SDK đòi kiểu khác
            raise GenerationError(
                f"TypeError từ Gemini SDK: {exc}. "
                f"debug types: prompt={type(prompt).__name__}, model={type(model).__name__}, "
                f"long_answer={type(long_answer).__name__}, max_tokens={type(max_tokens).__name__}"
            ) from exc
        except Exception as exc:
            raise GenerationError(f"Gọi Gemini thất bại ({exc}).") from exc
        ...
    except Exception as exc:
        if isinstance(exc, GenerationError): raise
        raise GenerationError(str(exc)) from exc