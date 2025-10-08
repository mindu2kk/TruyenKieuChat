# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POEM_TXT = PROJECT_ROOT / "data" / "interim" / "poem" / "poem.txt"
MOTIF_JSON = PROJECT_ROOT / "data" / "interim" / "poem" / "motifs.json"
CHUNK_DIR = PROJECT_ROOT / "data" / "rag_chunks"


@dataclass(frozen=True)
class PoemLine:
    number: int
    text: str
    motifs: Tuple[str, ...]


@dataclass(frozen=True)
class PoemCouplet:
    start: int
    end: int
    text: str
    motifs: Tuple[str, ...]
    lines: Tuple[PoemLine, ...]


def _strip_leading_number(s: str) -> str:
    return re.sub(r"^\s*\d{1,4}\s*[:\.\)]\s*", "", s).strip()


def _clean_line(s: str) -> str:
    s = s.replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _looks_like_verse(s: str) -> bool:
    return bool(s and re.search(r"[A-Za-zÀ-ỹ]", s))


def _read_poem_txt() -> List[str]:
    if not POEM_TXT.exists():
        return []
    lines: List[str] = []
    for raw in POEM_TXT.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = _clean_line(_strip_leading_number(raw))
        if ln and _looks_like_verse(ln):
            lines.append(ln)
    return lines


def _read_poem_from_chunks() -> List[str]:
    if not CHUNK_DIR.exists():
        return []
    verses: List[str] = []
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
def _load_motif_ranges() -> List[Tuple[int, int, str]]:
    if not MOTIF_JSON.exists():
        return []
    data = json.loads(MOTIF_JSON.read_text(encoding="utf-8"))
    ranges: List[Tuple[int, int, str]] = []
    for item in data:
        rng = item.get("range") or []
        if not isinstance(rng, list) or len(rng) != 2:
            continue
        a, b = int(rng[0]), int(rng[1])
        if a > b:
            a, b = b, a
        motif = str(item.get("motif") or "").strip()
        if not motif:
            continue
        ranges.append((a, b, motif))
    ranges.sort()
    return ranges


def _motifs_for_line(n: int) -> Tuple[str, ...]:
    motifs: List[str] = []
    for a, b, motif in _load_motif_ranges():
        if a <= n <= b:
            motifs.append(motif)
    return tuple(dict.fromkeys(motifs))


@lru_cache(maxsize=1)
def _load_poem_lines() -> List[PoemLine]:
    raw_lines = _read_poem_txt() or _read_poem_from_chunks()
    cleaned: List[PoemLine] = []
    number = 0
    for ln in raw_lines:
        if ln.strip() in {"I", "II", "III", "IV"}:
            continue
        if cleaned and cleaned[-1].text == ln:
            continue
        number += 1
        cleaned.append(PoemLine(number=number, text=ln, motifs=_motifs_for_line(number)))
    return cleaned


@lru_cache(maxsize=1)
def _poem_text_only() -> List[str]:
    return [line.text for line in _load_poem_lines()]


@lru_cache(maxsize=1)
def _load_couplets() -> List[PoemCouplet]:
    lines = _load_poem_lines()
    couplets: List[PoemCouplet] = []
    idx = 0
    while idx < len(lines):
        pair = lines[idx : idx + 2]
        if not pair:
            break
        start = pair[0].number
        end = pair[-1].number
        motifs = tuple(dict.fromkeys(m for ln in pair for m in ln.motifs))
        couplets.append(
            PoemCouplet(
                start=start,
                end=end,
                text="\n".join(ln.text for ln in pair),
                motifs=motifs,
                lines=tuple(pair),
            )
        )
        idx += 2
    return couplets


def poem_ready() -> bool:
    return len(_load_poem_lines()) >= 100


def get_opening(n: int) -> List[str]:
    lines = _poem_text_only()
    n = max(1, min(n, len(lines)))
    return lines[:n]


def get_range(a: int, b: int) -> List[str]:
    lines = _poem_text_only()
    if a > b:
        a, b = b, a
    a = max(1, a)
    b = min(b, len(lines))
    return lines[a - 1 : b]


def get_single(n: int) -> str | None:
    lines = _poem_text_only()
    if 1 <= n <= len(lines):
        return lines[n - 1]
    return None


def get_lines(numbers: Sequence[int]) -> List[PoemLine]:
    wanted = {int(n) for n in numbers if isinstance(n, (int, str))}
    if not wanted:
        return []
    index = {line.number: line for line in _load_poem_lines()}
    return [index[n] for n in sorted(wanted) if n in index]


def get_couplet_for_line(n: int) -> PoemCouplet | None:
    for couplet in _load_couplets():
        if couplet.start <= n <= couplet.end:
            return couplet
    return None


def compare_lines(a: int, b: int) -> Tuple[PoemLine | None, PoemLine | None]:
    index = {line.number: line for line in _load_poem_lines()}
    return index.get(int(a)), index.get(int(b))


def motif_overview() -> List[Dict[str, str]]:
    overview: List[Dict[str, str]] = []
    for start, end, motif in _load_motif_ranges():
        overview.append({"range": f"{start}–{end}", "motif": motif})
    return overview


def iter_corpus() -> Iterable[PoemCouplet]:
    return iter(_load_couplets())


def all_poem_lines() -> List[PoemLine]:
    return list(_load_poem_lines())


__all__ = [
    "PoemLine",
    "PoemCouplet",
    "poem_ready",
    "get_opening",
    "get_range",
    "get_single",
    "get_lines",
    "get_couplet_for_line",
    "compare_lines",
    "motif_overview",
    "iter_corpus",
    "all_poem_lines",
]

