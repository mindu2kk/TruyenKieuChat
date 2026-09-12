"""Intent-aware retrieval lanes for Truyện Kiều questions."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .router import normalize_query


def extract_line_range(query: str) -> tuple[int, int] | None:
    """Extract an explicit Kiều line range from a Vietnamese query."""
    normalized = normalize_query(query)
    match = re.search(r"\bcau\s+(?:so\s+)?(\d{1,4})\s+(?:den|-)\s+(\d{1,4})\b", normalized)
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if start < 1 or end < start or end > 3254:
        return None
    return start, end


@dataclass(frozen=True)
class RetrievalPolicy:
    lane: str
    allowed_types: tuple[str, ...]
    prefer_poem: bool = False
    max_per_source: int = 2

    def mongo_filter(self) -> dict:
        return {"meta.type": {"$in": list(self.allowed_types)}}


def select_retrieval_policy(query: str) -> RetrievalPolicy:
    normalized = normalize_query(query)
    if any(term in normalized for term in ("trich", "nguyen van", "cau tho", "tim cau", "cau so")):
        return RetrievalPolicy("verse", ("poem",), prefer_poem=True, max_per_source=2)
    if any(term in normalized for term in ("tu co", "giai nghia", "dien tich", "dien co", "nghia la gi")):
        return RetrievalPolicy("glossary", ("poem", "analysis"), prefer_poem=True, max_per_source=2)
    if any(term in normalized for term in ("nhan vat", "la ai", "tinh cach", "tam ly", "quan he")):
        return RetrievalPolicy("character", ("analysis", "poem", "summary"), max_per_source=2)
    if any(term in normalized for term in ("dong truyen", "su kien", "sau khi", "truoc khi", "dien bien")):
        return RetrievalPolicy("timeline", ("summary", "poem", "analysis"), max_per_source=2)
    if any(term in normalized for term in ("phan tich", "binh giang", "nghe thuat", "cam nhan", "so sanh")):
        return RetrievalPolicy("literary-analysis", ("analysis", "poem"), prefer_poem=True, max_per_source=2)
    return RetrievalPolicy("general", ("analysis", "poem", "summary", "bio"), max_per_source=2)


__all__ = ["RetrievalPolicy", "extract_line_range", "select_retrieval_policy"]
