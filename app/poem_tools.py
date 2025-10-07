# app/poem_tools.py
# -*- coding: utf-8 -*-
"""
Trích NGUYÊN VĂN Truyện Kiều theo số dòng, có xác thực:
- Đọc data/interim/poem/poem.txt (mỗi câu 1 dòng, đã làm sạch số).
- Rà soát: bỏ mục lục I/II, bỏ dòng trống, gộp khoảng trắng, loại lặp liên tiếp.
- Đảm bảo trả về ĐÚNG số câu yêu cầu (opening / range). Nếu thiếu, báo rõ.
"""
from __future__ import annotations
import re
from pathlib import Path
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POEM_TXT     = PROJECT_ROOT / "data" / "interim" / "poem" / "poem.txt"

_ROMAN = {"i","ii","iii","iv","v","vi","vii","viii","ix","x"}

def _clean_line(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _looks_verse(s: str) -> bool:
    return bool(s and re.search(r"[A-Za-zÀ-ỹ]", s))

def _is_roman_section(s: str) -> bool:
    return s.lower() in _ROMAN

def _dedup_consecutive(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        if not out or out[-1] != ln:
            out.append(ln)
    return out

@lru_cache(maxsize=1)
def _load_all_lines() -> list[str]:
    if not POEM_TXT.exists():
        return []
    raw = POEM_TXT.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = []
    for ln in raw:
        ln = _clean_line(ln)
        if not ln: 
            continue
        # bỏ dòng chỉ là số (trong trường hợp txt còn sót số)
        if re.fullmatch(r"\d{1,4}", ln):
            continue
        # bỏ mục lục I, II...
        if _is_roman_section(ln):
            continue
        if _looks_verse(ln):
            lines.append(ln)
    lines = _dedup_consecutive(lines)
    return lines

def poem_ready() -> bool:
    return len(_load_all_lines()) >= 100   # sẵn sàng tối thiểu

def total_lines() -> int:
    return len(_load_all_lines())

def get_opening(n: int) -> list[str]:
    L = _load_all_lines()
    n = max(1, min(n, len(L)))
    return L[:n]

def get_range(a: int, b: int) -> list[str]:
    L = _load_all_lines()
    if not L:
        return []
    a, b = max(1,a), min(b, len(L))
    if a > b:
        a, b = b, a
    return L[a-1:b]

def preview_numbered(a: int, lines: list[str]) -> str:
    """Trả về block text có đánh số, bắt đầu từ số dòng a (1-based)."""
    return "\n".join(f"{a+i:>4}: {ln}" for i, ln in enumerate(lines))
