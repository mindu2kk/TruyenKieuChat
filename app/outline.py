# -*- coding: utf-8 -*-
"""Load structured outlines for common Truyện Kiều prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTLINE_PATH = PROJECT_ROOT / "docs" / "outlines.md"


@dataclass(frozen=True)
class Outline:
    title: str
    body: str


def load_outlines() -> List[Outline]:
    if not OUTLINE_PATH.exists():
        return []
    text = OUTLINE_PATH.read_text(encoding="utf-8")
    chunks = re.split(r"^##\s+", text, flags=re.MULTILINE)
    outlines: List[Outline] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or chunk.startswith("# "):
            continue
        title, _, body = chunk.partition("\n")
        outlines.append(Outline(title=title.strip(), body=body.strip()))
    return outlines


__all__ = ["Outline", "load_outlines"]