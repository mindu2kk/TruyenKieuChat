# app/poem_tools.py
# -*- coding: utf-8 -*-
"""
Tiện ích NGUYÊN VĂN Truyện Kiều theo số dòng + tìm kiếm câu liên quan từ kho thơ.
Yêu cầu: data/interim/poem/poem.txt (mỗi câu 1 dòng, không có số; nếu còn số sẽ được lọc).
API:
- poem_ready() -> bool
- get_opening(n) -> list[str]
- get_range(a,b) -> list[str]
- get_line(n) -> str
- search_lines_by_keywords(query, top=6) -> list[(lineno, line, score)]
- search_lines_by_span(span_text, top=6) -> list[(lineno, line, score)]
"""

from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import re
import unicodedata
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POEM_TXT = PROJECT_ROOT / "data" / "interim" / "poem" / "poem.txt"

_RX_NUM_LEAD = re.compile(r"^\s*\d{1,4}[\s\.:)\-]*")

def _clean_line(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").replace("\u00A0", " ")
    s = _RX_NUM_LEAD.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _is_texty(s: str) -> bool:
    return bool(re.search(r"[A-Za-zÀ-ỹ]", s))

@lru_cache(maxsize=1)
def _load_lines() -> list[str]:
    if not POEM_TXT.exists():
        return []
    ls = []
    for raw in POEM_TXT.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = _clean_line(raw)
        if s and _is_texty(s) and s not in {"I","II","III","IV"}:
            ls.append(s)
    # chống trùng liên tiếp (do OCR/clean)
    out = []
    for x in ls:
        if not out or out[-1] != x:
            out.append(x)
    return out

def poem_ready() -> bool:
    return len(_load_lines()) >= 100  # đủ lớn coi như sẵn sàng

def get_opening(n: int) -> list[str]:
    lines = _load_lines()
    n = max(1, min(n, len(lines)))
    return lines[:n]

def get_range(a: int, b: int) -> list[str]:
    lines = _load_lines()
    if not lines: return []
    if a > b: a, b = b, a
    a = max(1, a); b = min(len(lines), b)
    return lines[a-1:b]

def get_line(n: int) -> str:
    lines = _load_lines()
    if 1 <= n <= len(lines):
        return lines[n-1]
    return ""

# ======= Tìm câu thơ gần nghĩa câu hỏi/ngữ cảnh =======
_WORD_RX = re.compile(r"[0-9A-Za-zÀ-ỹ']+")

def _tokenize(s: str) -> list[str]:
    return [w.lower() for w in _WORD_RX.findall(s or "")]

@lru_cache(maxsize=1)
def _inverted_index():
    """Chỉ số nghèo nàn: word -> {lineno: tf}"""
    idx = {}
    lines = _load_lines()
    for i, ln in enumerate(lines, start=1):
        toks = _tokenize(ln)
        if not toks: continue
        tf = Counter(toks)
        for w, c in tf.items():
            idx.setdefault(w, {})[i] = c
    return idx

def _score_query_tokens(qtoks: list[str], lineno: int, lines: list[str]) -> float:
    """Điểm sơ sài = chồng lấn từ (trọng số nhẹ theo độ dài câu)."""
    if not qtoks: return 0.0
    ln = lines[lineno-1]
    toks = _tokenize(ln)
    if not toks: return 0.0
    overlap = sum(1 for t in qtoks if t in toks)
    return overlap / (1.5 + len(toks)**0.5)

def _search_core(qtoks: list[str], top=6):
    lines = _load_lines()
    idx = _inverted_index()
    cand = set()
    for t in set(qtoks):
        for ln in idx.get(t, {}).keys():
            cand.add(ln)
    scored = [(ln, lines[ln-1], _score_query_tokens(qtoks, ln, lines)) for ln in cand]
    scored.sort(key=lambda x: (-x[2], x[0]))
    return [(ln, s, sc) for ln, s, sc in scored[:top] if sc > 0]

def search_lines_by_keywords(query: str, top=6):
    qtoks = _tokenize(query)
    return _search_core(qtoks, top=top)

def search_lines_by_span(span_text: str, top=6):
    # dùng chính token câu span_text để tìm câu tương tự
    qtoks = _tokenize(span_text)
    return _search_core(qtoks, top=top)
