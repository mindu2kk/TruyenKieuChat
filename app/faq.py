# -*- coding: utf-8 -*-
"""
FAQ lookup: nếu câu hỏi khớp các mẫu đơn giản -> trả thẳng (không gọi RAG).
"""
import json
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


def lookup_faq(query: str) -> Optional[Dict[str, Any]]:
    q = _norm(query)
    for item in _load_facts():
        for pat in item["patterns"]:
            if _norm(pat) in q:
                return {"answer": item["answer"], "sources": item.get("sources", [])}
    return None
