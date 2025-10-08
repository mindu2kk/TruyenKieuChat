# app/generation.py
# -*- coding: utf-8 -*-
import os
import re
from typing import Any, Dict, Optional

import google.generativeai as genai


class GenerationError(RuntimeError):
    """Raised when the Gemini client cannot be used."""


def is_gemini_configured() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


try:  # cho cả dạng package và chạy script
    from .prompt_engineering import (
        DEFAULT_LONG_TOKEN_BUDGET,
        DEFAULT_SHORT_TOKEN_BUDGET,
    )
except ImportError:
    from prompt_engineering import (  # type: ignore
        DEFAULT_LONG_TOKEN_BUDGET,
        DEFAULT_SHORT_TOKEN_BUDGET,
    )


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
    # bỏ các dòng “Nguồn: … / Source: …”
    lines = []
    for ln in ans.splitlines():
        if re.match(r"^\s*(Nguồn|Source)\s*:.*$", ln, flags=re.I):
            continue
        lines.append(ln)
    txt = "\n".join(lines).strip()
    txt = re.sub(r"(?:\n\n)+", "\n\n", txt)
    return txt


def _resolve_generation_config(long_answer: bool, max_tokens: Optional[int]) -> Dict[str, Any]:
    resolved_max = max_tokens
    if resolved_max is None:
        resolved_max = DEFAULT_LONG_TOKEN_BUDGET if long_answer else DEFAULT_SHORT_TOKEN_BUDGET

    # BẮT BUỘC đúng kiểu dữ liệu (float/int)
    return {
        "temperature": float(0.6 if long_answer else 0.55),
        "top_p": float(0.9),
        "top_k": int(40),
        "max_output_tokens": int(resolved_max),
    }


def _call_gemini_safe(model: str, prompt: str, generation_config: Optional[Dict[str, Any]]):
    """
    Gọi Gemini an toàn trên nhiều version SDK:
    - Thử truyền generation_config vào generate_content (cách khuyến nghị)
    - Nếu dính TypeError (len(int)), thử cực giản chỉ max_output_tokens
    - Nếu vẫn lỗi, gọi mặc định không config
    """
    gm = genai.GenerativeModel(model_name=model)

    # Cách 1: full config
    try:
        return gm.generate_content(
            prompt,
            generation_config=generation_config,
        )
    except TypeError as exc:
        # Thường gặp “object of type 'int' has no len()” khi SDK đổi API nội bộ
        last_err = exc
    except Exception as exc:
        raise GenerationError(f"Gọi Gemini thất bại ({exc}).") from exc

    # Cách 2: chỉ giữ max_output_tokens
    try:
        minimal_cfg = None
        if generation_config and "max_output_tokens" in generation_config:
            minimal_cfg = {"max_output_tokens": int(generation_config["max_output_tokens"])}
        return gm.generate_content(
            prompt,
            generation_config=minimal_cfg,
        )
    except TypeError:
        pass
    except Exception as exc:
        raise GenerationError(f"Gọi Gemini thất bại ({exc}).") from exc

    # Cách 3: không truyền config (mặc định SDK)
    try:
        return gm.generate_content(prompt)
    except Exception as exc:
        raise GenerationError(f"Gọi Gemini thất bại ({exc}).") from exc


def generate_answer_gemini(
    prompt: str,
    model: str = "gemini-2.0-flash",
    long_answer: bool = False,
    max_tokens: Optional[int] = None,
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

    res = _call_gemini_safe(model, prompt, generation_config)

    # Trích text an toàn
    out = ""
    try:
        out = res.text  # type: ignore[attr-defined]
    except Exception:
        try:
            parts = []
            cand = getattr(res, "candidates", None)
            if cand and len(cand) > 0:
                content = getattr(cand[0], "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        if hasattr(part, "text") and part.text:
                            parts.append(part.text)
            out = "".join(parts).strip()
        except Exception:
            out = str(res)

    return _postprocess(out or "")
