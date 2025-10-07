# app/poem_tools.py
# -*- coding: utf-8 -*-
"""
Trích NGUYÊN VĂN Truyện Kiều theo số dòng (opening / range).
- Ưu tiên đọc: data/interim/poem/poem.txt (mỗi câu 1 dòng, có thể còn số -> sẽ tự lọc)
- Fallback: ghép từ data/rag_chunks/ (meta.type == "poem"), chỉ dùng tạm.

API:
- poem_ready() -> bool
- get_opening(n: int) -> list[str]
- get_range(a: int, b: int) -> list[str]
- get_total_lines() -> int
"""

from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import re, unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POEM_TXT   = PROJECT_ROOT / "data" / "interim" / "poem" / "poem.txt"
CHUNK_DIR  = PROJECT_ROOT / "data" / "rag_chunks"

_ROMAN_SET = {"I","II","III","IV","V","VI","VII","VIII","IX","X",
              "XI","XII","XIII","XIV","XV","XVI","XVII","XVIII","XIX","XX"}

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).replace("\u00A0", " ")

def _strip_numbers_everywhere(s: str) -> str:
    """
    Dọn các dạng số thường gặp khi copy:
      - Dòng toàn số: '123'
      - Số đầu dòng: '123:', '001.', '12) ', '12 - '
      - Số lẻ rơi cuối dòng: '... câu thơ 5' (hiếm)
    Không đụng vào dấu câu/ chữ.
    """
    t = _nfc(s).strip()
    if not t:
        return ""
    # bỏ dòng toàn số
    if re.fullmatch(r"\d+", t):
        return ""
    # bỏ số đầu dòng (có thể kèm :,.-) và khoảng trắng
    t = re.sub(r"^\s*\d{1,5}\s*[:\.\)\-–—]*\s*", "", t)
    # bóc số lẻ ở CUỐI dòng (tránh trường hợp số trang chui vào)
    t = re.sub(r"\s*\d{1,5}\s*$", "", t)
    # gọn khoảng trắng
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _looks_like_verse(s: str) -> bool:
    # có ít nhất 1 ký tự chữ cái tiếng Việt/Latin
    return bool(re.search(r"[A-Za-zÀ-ỹ]", s))

def _is_roman_heading(s: str) -> bool:
    # Loại các dòng roman ‘I’, ‘II’… (thường là đề mục)
    return s.strip() in _ROMAN_SET

def _clean_lines(raw_text: str) -> list[str]:
    out = []
    for ln in raw_text.splitlines():
        s = _strip_numbers_everywhere(ln)
        if not s:
            continue
        if _is_roman_heading(s):
            continue
        if not _looks_like_verse(s):
            continue
        # tránh trùng do copy/paste
        if not out or out[-1] != s:
            out.append(s)
    return out

def _read_poem_txt() -> list[str]:
    if not POEM_TXT.exists():
        return []
    raw = POEM_TXT.read_text(encoding="utf-8", errors="ignore")
    return _clean_lines(raw)

def _read_poem_from_chunks() -> list[str]:
    if not CHUNK_DIR.exists():
        return []
    verses: list[str] = []
    for p in CHUNK_DIR.glob("*.txt"):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if not raw.startswith("###META###"):
            continue
        meta_line, _, body = raw.partition("\n")
        m = re.search(r'"type"\s*:\s*"([^"]+)"', meta_line)
        if not m or m.group(1) != "poem":
            continue
        verses.extend(_clean_lines(body))
    return verses

@lru_cache(maxsize=1)
def _load_poem_lines() -> list[str]:
    lines = _read_poem_txt()
    if not lines:
        lines = _read_poem_from_chunks()
    return lines

def poem_ready() -> bool:
    # Kiểm tra “tương đối”. Truyện Kiều ~3.254 câu, nhưng để an toàn chỉ cần >= 500 đã coi là có dữ liệu.
    return get_total_lines() >= 500

def get_total_lines() -> int:
    return len(_load_poem_lines())

def get_opening(n: int) -> list[str]:
    lines = _load_poem_lines()
    if not lines:
        return []
    n = max(1, min(n, len(lines)))
    return lines[:n]

def get_range(a: int, b: int) -> list[str]:
    lines = _load_poem_lines()
    if not lines:
        return []
    if a > b:
        a, b = b, a
    a = max(1, a)
    b = min(b, len(lines))
    return lines[a-1:b]
