"""Corpus quality primitives shared by builders, indexers and retrieval.

The module is deliberately dependency free so corpus validation and tests do
not need MongoDB, an embedding model, or network access.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence


SOURCE_TIER_WEIGHT = {
    "primary": 4,
    "scholarly": 3,
    "educational": 2,
    "reference": 1,
}

_SCHOLARLY_MARKERS = (
    "tran-dinh-su",
    "luan-van",
    "nguvan.hnue.edu.vn",
    "tapchicongsan.org.vn",
    "baotanglichsuquocgia.vn",
    "nguyendu.vn",
    "nguyendu.com.vn",
)
_EDUCATIONAL_MARKERS = (
    "edu.vn",
    "loigiaihay.com",
    "vietjack.com",
    "vuihoc.vn",
    "hoctotnguvan.vn",
    "hoclagioi.vn",
)
_KNOWN_AUTHORS = {
    "tran-dinh-su": "Trần Đình Sử",
    "phan-ngoc": "Phan Ngọc",
}

POEM_SECTIONS = (
    (1, 34, "mo_dau_nhan_sinh", "Mở đầu: cảm hứng nhân sinh"),
    (35, 80, "chi_em_thuy_kieu", "Chân dung chị em Thuý Kiều"),
    (81, 150, "hoi_dap_thanh", "Ngày xuân và hội Đạp Thanh"),
    (151, 210, "kieu_kim_trong", "Mối tình Kiều – Kim Trọng"),
    (211, 340, "gia_bien_ban_minh", "Gia biến và bán mình"),
    (341, 620, "lau_xanh_so_khanh", "Lầu xanh và mưu đồ Sở Khanh"),
    (621, 1050, "gia_dinh_hoan_thu", "Gia đình Hoạn Thư"),
    (1051, 1500, "chi_khi_tu_hai", "Chí khí Từ Hải"),
    (1501, 2300, "luu_lac_bao_an_bao_oan", "Lưu lạc và báo ân báo oán"),
    (2301, 2448, "doan_tu_loi_ket", "Đoàn tụ và lời kết"),
)


def canonical_text(text: str) -> str:
    """Return stable NFC text used for exact duplicate detection."""

    value = unicodedata.normalize("NFC", text or "").replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value


def content_hash(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def infer_source_tier(meta: Mapping[str, Any]) -> str:
    if str(meta.get("type") or "").lower() == "poem":
        return "primary"
    source = " ".join(
        str(meta.get(key) or "").lower()
        for key in ("source", "source_id", "source_url", "publisher", "title")
    )
    if any(marker in source for marker in _SCHOLARLY_MARKERS):
        return "scholarly"
    if any(marker in source for marker in _EDUCATIONAL_MARKERS):
        return "educational"
    return "reference"


def infer_publisher(meta: Mapping[str, Any]) -> str:
    if str(meta.get("type") or "").lower() == "poem":
        return str(meta.get("publisher") or "Bản văn Truyện Kiều dùng trong dự án")
    source = str(meta.get("source_url") or meta.get("source") or "")
    match = re.search(r"(?:www\.|m\.)?([a-z0-9.-]+\.(?:com\.vn|edu\.vn|gov\.vn|vn|com|org|net|fr))", source)
    if match:
        return match.group(1).lower()
    return str(meta.get("publisher") or "Chưa xác định")


def infer_author(meta: Mapping[str, Any]) -> str:
    if str(meta.get("type") or "").lower() == "poem":
        return "Nguyễn Du"
    source = " ".join(str(meta.get(key) or "").lower() for key in ("source", "source_id", "title"))
    for marker, author in _KNOWN_AUTHORS.items():
        if marker in source:
            return author
    return str(meta.get("author") or "Chưa xác định")


def enrich_metadata(meta: Mapping[str, Any], text: str) -> Dict[str, Any]:
    """Add a stable metadata contract without inventing unknown scholarship."""

    enriched = dict(meta)
    doc_type = str(enriched.get("type") or "reference")
    tags = [str(tag) for tag in enriched.get("tags", []) if str(tag).strip()]
    enriched["tags"] = list(dict.fromkeys(tags))
    enriched["content_hash"] = content_hash(text)
    enriched["source_tier"] = str(enriched.get("source_tier") or infer_source_tier(enriched))
    enriched["publisher"] = infer_publisher(enriched)
    current_author = str(enriched.get("author") or "")
    if not current_author or current_author == "Chưa xác định":
        enriched["author"] = infer_author(enriched)
    else:
        enriched["author"] = current_author
    enriched["work"] = str(enriched.get("work") or ("Truyện Kiều" if doc_type == "poem" else enriched.get("title") or ""))
    enriched["section"] = str(enriched.get("section") or _first_tag(enriched["tags"], "section:"))
    for key, prefix in (
        ("characters", "char:"),
        ("events", "event:"),
        ("allusions", "allusion:"),
        ("archaic_terms", "archaic:"),
        ("literary_devices", "device:"),
    ):
        existing = [str(value) for value in enriched.get(key, []) if str(value).strip()]
        enriched[key] = list(dict.fromkeys(existing + _tag_values(enriched["tags"], prefix)))
    return enriched


def poem_section_for_line(line_number: int) -> tuple[str, str] | None:
    for start, end, section_id, title in POEM_SECTIONS:
        if start <= line_number <= end:
            return section_id, title
    return None


def _tag_values(tags: Sequence[str], prefix: str) -> List[str]:
    return [tag[len(prefix) :] for tag in tags if tag.startswith(prefix)]


def _first_tag(tags: Sequence[str], prefix: str) -> str:
    values = _tag_values(tags, prefix)
    return values[0] if values else ""


def record_quality_key(record: Mapping[str, Any]) -> tuple[int, int, int, str]:
    meta = record.get("meta") or {}
    tier = str(meta.get("source_tier") or infer_source_tier(meta))
    source_id = str(meta.get("source_id") or "").lower()
    canonical_poem = int(meta.get("type") == "poem" and source_id in {"poem", "truyen-kieu-canonical"})
    valid_position = int(
        (meta.get("type") == "poem" and isinstance(meta.get("line_start"), int))
        or (
            meta.get("type") != "poem"
            and isinstance(meta.get("char_start"), int)
            and int(meta.get("char_start", -1)) >= 0
        )
    )
    return SOURCE_TIER_WEIGHT.get(tier, 0), canonical_poem, valid_position, str(meta.get("id") or "")


def deduplicate_records(records: Iterable[Mapping[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Yield one best record per normalized content hash.

    The canonical ``poem`` source wins over ``poem.raw``. For other collisions,
    source tier and usable location metadata decide deterministically.
    """

    best: Dict[str, Dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        record["meta"] = enrich_metadata(record.get("meta") or {}, str(record.get("text") or ""))
        digest = str(record["meta"]["content_hash"])
        existing = best.get(digest)
        if existing is None or record_quality_key(record) > record_quality_key(existing):
            best[digest] = record
    yield from sorted(best.values(), key=lambda item: str((item.get("meta") or {}).get("id") or ""))


def diversify_hits(
    hits: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    max_per_source: int = 2,
) -> List[Dict[str, Any]]:
    """Remove duplicate evidence and cap domination by one source."""

    selected: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()
    source_counts: Dict[str, int] = defaultdict(int)

    for raw in hits:
        hit = dict(raw)
        meta = dict(hit.get("meta") or hit.get("metadata") or {})
        digest = str(meta.get("content_hash") or content_hash(str(hit.get("text") or "")))
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        source = str(meta.get("source_id") or meta.get("source") or "unknown")
        if source_counts[source] >= max_per_source:
            continue
        source_counts[source] += 1
        selected.append(hit)
        if len(selected) >= limit:
            return selected
    return selected


@dataclass(frozen=True)
class CorpusStats:
    total: int
    unique: int
    duplicates: int
    by_type: Dict[str, int]
    invalid_positions: int


def corpus_stats(records: Iterable[Mapping[str, Any]]) -> CorpusStats:
    rows = list(records)
    hashes: set[str] = set()
    by_type: Dict[str, int] = defaultdict(int)
    invalid_positions = 0
    for row in rows:
        meta = row.get("meta") or {}
        text = str(row.get("text") or "")
        hashes.add(str(meta.get("content_hash") or content_hash(text)))
        doc_type = str(meta.get("type") or "unknown")
        by_type[doc_type] += 1
        if doc_type == "poem":
            invalid_positions += not (
                isinstance(meta.get("line_start"), int)
                and isinstance(meta.get("line_end"), int)
                and int(meta["line_start"]) >= 1
                and int(meta["line_end"]) >= int(meta["line_start"])
            )
        else:
            invalid_positions += not (
                isinstance(meta.get("char_start"), int)
                and isinstance(meta.get("char_end"), int)
                and int(meta["char_start"]) >= 0
                and int(meta["char_end"]) > int(meta["char_start"])
            )
    return CorpusStats(
        total=len(rows),
        unique=len(hashes),
        duplicates=len(rows) - len(hashes),
        by_type=dict(by_type),
        invalid_positions=int(invalid_positions),
    )


def safe_chunk_dir(path: Path, project_root: Path) -> Path:
    """Resolve and constrain a generated chunk directory to ``data``."""

    resolved = path.resolve()
    data_root = (project_root / "data").resolve()
    if resolved == data_root or data_root not in resolved.parents:
        raise ValueError(f"Chunk output must stay below {data_root}")
    return resolved


__all__ = [
    "CorpusStats",
    "SOURCE_TIER_WEIGHT",
    "canonical_text",
    "content_hash",
    "corpus_stats",
    "deduplicate_records",
    "diversify_hits",
    "enrich_metadata",
    "infer_source_tier",
    "infer_author",
    "safe_chunk_dir",
]
