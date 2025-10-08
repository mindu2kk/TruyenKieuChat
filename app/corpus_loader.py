# -*- coding: utf-8 -*-
"""Utility helpers to expose a unified Truyện Kiều corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

try:  # pragma: no cover - support execution without package context
    from .poem_tools import PoemCouplet, iter_corpus
except ImportError:  # pragma: no cover
    from poem_tools import PoemCouplet, iter_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    text: str
    metadata: Dict[str, object]


def _iter_text_files(folder: Path) -> Iterator[Path]:
    if not folder.exists():
        return iter(())
    patterns = ("*.txt", "*.md")
    return (path for pattern in patterns for path in folder.glob(pattern))


def _clean_paragraphs(text: str, *, min_words: int = 18) -> List[str]:
    paragraphs: List[str] = []
    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        if not block:
            continue
        block = re.sub(r"\s+", " ", block)
        if len(block.split()) < min_words:
            continue
        paragraphs.append(block)
    return paragraphs


def _load_poem_documents() -> List[CorpusDocument]:
    docs: List[CorpusDocument] = []
    for couplet in iter_corpus():
        meta = {
            "type": "poem",
            "source": "Truyện Kiều",
            "line_start": couplet.start,
            "line_end": couplet.end,
            "motifs": list(couplet.motifs),
        }
        docs.append(
            CorpusDocument(
                doc_id=f"poem::{couplet.start:04d}-{couplet.end:04d}",
                text=couplet.text,
                metadata=meta,
            )
        )
    return docs


def _load_folder(folder: Path, *, doc_type: str) -> List[CorpusDocument]:
    docs: List[CorpusDocument] = []
    for path in _iter_text_files(folder):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        paragraphs = _clean_paragraphs(raw)
        if not paragraphs:
            continue
        for idx, para in enumerate(paragraphs, start=1):
            docs.append(
                CorpusDocument(
                    doc_id=f"{doc_type}::{path.stem}::{idx}",
                    text=para,
                    metadata={
                        "type": doc_type,
                        "source": path.name,
                        "paragraph": idx,
                    },
                )
            )
    return docs


def load_corpus(include: Sequence[str] | None = None) -> List[CorpusDocument]:
    if include is None:
        include = ("poem", "analysis", "summary", "bio", "ana")

    include_set = {item.lower() for item in include}
    docs: List[CorpusDocument] = []

    if "poem" in include_set:
        docs.extend(_load_poem_documents())

    mapping = {
        "analysis": INTERIM_DIR / "analysis",
        "summary": INTERIM_DIR / "summary",
        "bio": INTERIM_DIR / "bio",
        "ana": INTERIM_DIR / "ana",
    }

    for doc_type, folder in mapping.items():
        if doc_type in include_set:
            docs.extend(_load_folder(folder, doc_type=doc_type))

    return docs


__all__ = ["CorpusDocument", "load_corpus"]