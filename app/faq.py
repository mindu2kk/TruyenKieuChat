# -*- coding: utf-8 -*-
"""
FAQ lookup: nếu câu hỏi khớp các mẫu đơn giản -> trả thẳng (không gọi RAG).
"""
import json
from difflib import SequenceMatcher
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Any, List

from .router import normalize_query


@lru_cache(maxsize=1)
def _load_facts() -> List[Dict[str, Any]]:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    # ``fag`` is the historical directory name committed by the project.
    # Prefer a corrected ``faq`` directory when it exists, but keep the
    # deployed data readable without a risky repository-wide move.
    paths = (data_dir / "faq" / "facts.json", data_dir / "fag" / "facts.json")
    p = next((candidate for candidate in paths if candidate.exists()), None)
    if p is None:
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _norm(s: str) -> str:
    return normalize_query(s)


_CHARACTER_NAMES = (
    "thuy kieu",
    "thuy van",
    "kim trong",
    "tu hai",
    "hoan thu",
    "thuc sinh",
    "ma giam sinh",
    "so khanh",
    "tu ba",
    "giac duyen",
    "dam tien",
)

_DIRECT_CHARACTER_CUES = (
    "la ai",
    "la aj",
    "la gi cua",
    "vai tro gi",
    "vai tro cua",
    "nhan vat",
    "gioi thieu",
    "la nguoi the nao",
)


def _mentions_character(query: str, character: str) -> bool:
    """Match a known name while tolerating one small typing error."""
    if character in query:
        return True
    query_tokens = query.split()
    name_tokens = character.split()
    width = len(name_tokens)
    return any(
        SequenceMatcher(None, " ".join(query_tokens[index : index + width]), character).ratio() >= 0.86
        for index in range(max(0, len(query_tokens) - width + 1))
    )


def lookup_faq(query: str) -> Optional[Dict[str, Any]]:
    q = _norm(query)
    facts = _load_facts()
    exact_matches = []
    for item in facts:
        for pat in item["patterns"]:
            normalized_pattern = _norm(pat)
            if normalized_pattern in q:
                exact_matches.append((len(normalized_pattern), item))
    if exact_matches:
        # A specific phrase such as "giá trị nội dung" must outrank the
        # generic substring "Truyện Kiều là gì" contained in the same query.
        item = max(exact_matches, key=lambda match: match[0])[1]
        return {"answer": item["answer"], "sources": item.get("sources", [])}

    # Character identity/relationship questions are common and should not be
    # sent through RAG merely because the wording or one keystroke differs from
    # a curated pattern.  Restrict fuzzy matching to explicit character cues so
    # ordinary analysis questions still use corpus evidence.
    if any(cue in q for cue in _DIRECT_CHARACTER_CUES):
        matched_characters = [character for character in _CHARACTER_NAMES if _mentions_character(q, character)]
        matched_characters.sort(key=lambda character: q.find(character) if character in q else len(q) + 1)
        for character in matched_characters:
            for item in facts:
                if any(character in _norm(pattern) for pattern in item["patterns"]):
                    return {"answer": item["answer"], "sources": item.get("sources", [])}
    return None
