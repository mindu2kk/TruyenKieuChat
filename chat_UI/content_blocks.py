"""Build safe, structured presentation blocks from grounded chatbot output."""

from __future__ import annotations

import re
from typing import Any

from app.router import normalize_query


CHARACTERS = {
    "thuy kieu": {
        "name": "Thúy Kiều",
        "role": "Nhân vật trung tâm",
        "traits": ["tài hoa", "đa cảm", "hiếu thảo"],
    },
    "thuy van": {
        "name": "Thúy Vân",
        "role": "Em gái Thúy Kiều",
        "traits": ["đoan trang", "phúc hậu", "điềm tĩnh"],
    },
    "kim trong": {
        "name": "Kim Trọng",
        "role": "Người tri kỷ của Thúy Kiều",
        "traits": ["trọng tình", "chung thủy", "nho nhã"],
    },
    "tu hai": {
        "name": "Từ Hải",
        "role": "Người anh hùng lý tưởng",
        "traits": ["khí phách", "tự do", "trọng nghĩa"],
    },
    "hoan thu": {
        "name": "Hoạn Thư",
        "role": "Nhân vật đối trọng giàu phức tạp",
        "traits": ["sắc sảo", "ghen tuông", "khôn ngoan"],
    },
    "thuc sinh": {
        "name": "Thúc Sinh",
        "role": "Người từng cứu Kiều khỏi lầu xanh",
        "traits": ["đa tình", "yếu đuối", "thiếu quyết đoán"],
    },
    "ma giam sinh": {
        "name": "Mã Giám Sinh",
        "role": "Kẻ buôn người đội lốt nho sinh",
        "traits": ["giả dối", "thô lỗ", "vụ lợi"],
    },
    "tu ba": {
        "name": "Tú Bà",
        "role": "Chủ lầu xanh",
        "traits": ["lọc lõi", "tàn nhẫn", "mưu mô"],
    },
}


def _paragraphs(answer: str) -> list[str]:
    return [item.strip() for item in re.split(r"\n\s*\n", answer or "") if item.strip()]


def _timeline_items(answer: str) -> list[str]:
    items = []
    for line in (answer or "").splitlines():
        match = re.match(r"\s*(?:\d+[.)]|[-•])\s+(.+)", line)
        if match:
            value = re.sub(r"\*\*", "", match.group(1)).strip()
            if value:
                items.append(value[:180])
    return items[:8]


def _verified_verse_blocks(verification: dict[str, Any] | None) -> list[dict[str, Any]]:
    accepted = (verification or {}).get("accepted") or []
    blocks = []
    seen = set()
    for quote in accepted:
        text = str(quote.get("matched_text") or quote.get("quote") or "").strip()
        line = quote.get("matched_line")
        key = (text, line)
        if not text or key in seen:
            continue
        seen.add(key)
        blocks.append(
            {
                "type": "verse_quote",
                "text": text,
                "source": "Truyện Kiều · Nguyễn Du",
                "line_start": line if isinstance(line, int) else None,
                "line_end": line if isinstance(line, int) else None,
                "verified": True,
            }
        )
    return blocks


def build_content_blocks(
    answer: str,
    *,
    query: str = "",
    intent: str = "domain",
    verification: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return presentation data without changing the canonical answer text."""
    paragraphs = _paragraphs(answer)
    blocks: list[dict[str, Any]] = []
    if paragraphs:
        blocks.append({"type": "summary", "markdown": paragraphs[0]})

    blocks.extend(_verified_verse_blocks(verification))

    normalized_query = normalize_query(query)
    for key, character in CHARACTERS.items():
        if key in normalized_query:
            blocks.append({"type": "character_card", **character})
            break

    timeline = _timeline_items(answer) if intent == "plot" else []
    if len(timeline) >= 2:
        blocks.append({"type": "timeline", "title": "Dòng sự kiện", "items": timeline})

    remaining = "\n\n".join(paragraphs[1:]).strip()
    if remaining:
        context_match = re.search(
            r"(?:^|\n)#{0,3}\s*(?:Bối cảnh|Hoàn cảnh)\s*\n(?P<body>.*?)(?=\n#{1,3}\s|\Z)",
            remaining,
            re.IGNORECASE | re.DOTALL,
        )
        if context_match:
            context = context_match.group("body").strip()
            if context:
                blocks.append({"type": "context", "title": "Bối cảnh", "markdown": context})
        blocks.append({"type": "analysis", "markdown": remaining})

    if verification and verification.get("quotes"):
        blocks.append(
            {
                "type": "reference",
                "label": "Kiểm chứng văn bản",
                "coverage": float(verification.get("coverage") or 0),
                "quote_count": len(verification.get("quotes") or []),
            }
        )
    return blocks


__all__ = ["build_content_blocks"]
