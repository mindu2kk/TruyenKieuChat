# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from pathlib import Path
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POEM_TXT   = PROJECT_ROOT / "data" / "interim" / "poem" / "poem.txt"
CHUNK_DIR  = PROJECT_ROOT / "data" / "rag_chunks"

def _strip_leading_number(s: str) -> str:
    return re.sub(r"^\s*\d{1,4}\s*[:\.\)]\s*", "", s).strip()

def _clean_line(s: str) -> str:
    s = s.replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _looks_like_verse(s: str) -> bool:
    return bool(s and re.search(r"[A-Za-zÀ-ỹ]", s))

def _read_poem_txt() -> list[str]:
    if not POEM_TXT.exists():
        return []
    lines = []
    for raw in POEM_TXT.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = _clean_line(_strip_leading_number(raw))
        if ln and _looks_like_verse(ln):
            lines.append(ln)
    return lines

def _read_poem_from_chunks() -> list[str]:
    if not CHUNK_DIR.exists():
        return []
    verses = []
    for p in CHUNK_DIR.glob("*.txt"):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if not raw.startswith("###META###"):
            continue
        meta_line, _, body = raw.partition("\n")
        m = re.search(r'"type"\s*:\s*"([^"]+)"', meta_line)
        if not m or m.group(1) != "poem":
            continue
        for rawln in body.splitlines():
            ln = _clean_line(_strip_leading_number(rawln))
            if ln and _looks_like_verse(ln):
                verses.append(ln)
    return verses

@lru_cache(maxsize=1)
def _load_poem_lines() -> list[str]:
    lines = _read_poem_txt() or _read_poem_from_chunks()
    cleaned = []
    for ln in lines:
        if ln.strip() in {"I", "II", "III", "IV"}:
            continue
        if not cleaned or cleaned[-1] != ln:
            cleaned.append(ln)
    return cleaned

def poem_ready() -> bool:
    return len(_load_poem_lines()) >= 100

def get_opening(n: int) -> list[str]:
    lines = _load_poem_lines()
    n = max(1, min(n, len(lines)))
    return lines[:n]

def get_range(a: int, b: int) -> list[str]:
    lines = _load_poem_lines()
    if a > b: a, b = b, a
    a = max(1, a); b = min(b, len(lines))
    return lines[a-1:b]

def get_single(n: int) -> str | None:
    lines = _load_poem_lines()
    if 1 <= n <= len(lines):
        return lines[n-1]
    return None
