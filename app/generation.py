# -*- coding: utf-8 -*-
import os, re
import google.generativeai as genai

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

def generate_answer_gemini(
    prompt: str,
    model: str = "gemini-2.0-flash",
    long_answer: bool = False,
    max_tokens: int | None = None,
) -> str:
    _setup()
    generation_config = {}
    if max_tokens:
        generation_config["max_output_tokens"] = int(max_tokens)

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
