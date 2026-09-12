# -*- coding: utf-8 -*-
"""Build a clean, traceable RAG corpus from ``data/interim``.

Only ``data/interim/poem/poem.txt`` is treated as the canonical poem. The raw
poem transcription remains available for comparison but is never indexed.
Output is written to ``data/rag_chunks_clean`` by default, leaving the current
production corpus untouched until it has been validated and promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.corpus_quality import enrich_metadata, safe_chunk_dir

SRC = ROOT / "data" / "interim"
DEFAULT_DST = ROOT / "data" / "rag_chunks_clean"
CANONICAL_POEM = SRC / "poem" / "poem.txt"
MOTIF_FILE = SRC / "poem" / "motifs.jsonl"
PROSE_MAX_WORDS = 220
POEM_LINES_PER_BLOCK = 4
POEM_OVERLAP_LINES = 1

TYPE_BY_DIR = {
    "poem": "poem",
    "analysis": "analysis",
    "ana": "analysis",
    "summary": "summary",
    "bio": "bio",
}

TAG_PATTERNS = {
    "char:thuy_kieu": (r"\bth[uú]y\s+ki[eề]u\b",),
    "char:thuy_van": (r"\bth[uú]y\s+v[aâ]n\b",),
    "char:kim_trong": (r"\bkim\s+tr[oọ]ng\b",),
    "char:tu_hai": (r"\bt[ừu]\s+h[aả]i\b",),
    "char:hoan_thu": (r"\bho[aạ]n\s+th[ưủu]\b",),
    "char:thuc_sinh": (r"\bth[uú]c\s+sinh\b",),
    "char:ma_giam_sinh": (r"\bm[aã]\s+gi[aá]m\s+sinh\b",),
    "char:so_khanh": (r"\bs[ởo]\s+khanh\b",),
    "char:tu_ba": (r"\bt[uú]\s+b[aà]\b",),
    "char:giac_duyen": (r"\bgi[aá]c\s+duy[eê]n\b",),
    "char:dam_tien": (r"\b[đd][aạ]m\s+ti[eê]n\b",),
    "device:uoc_le": (r"\bước\s+lệ\b", r"\btượng\s+trưng\b"),
    "device:dien_co": (r"\bđiển\s+(?:cố|tích)\b",),
    "device:an_du": (r"\bẩn?\s*dụ\b",),
    "device:nhan_hoa": (r"\bnhân\s+hóa\b",),
    "device:ta_canh_ngu_tinh": (r"\btả\s+cảnh\s+ngụ\s+tình\b",),
    "theme:tai_menh": (r"\btài\s+mệnh\b", r"\btài\s+vận\b"),
    "theme:chu_tam": (r"\bchữ\s+tâm\b",),
    "theme:nhan_dao": (r"\bnhân\s+đạo\b",),
    "theme:tinh_yeu": (r"\btình\s+yêu\b", r"\bduyên\b"),
    "theme:so_phan": (r"\bsố\s+phận\b", r"\bbạc\s+mệnh\b"),
    "section:trao_duyen": (r"\btrao\s+duyên\b",),
    "section:canh_ngay_xuan": (r"\bcảnh\s+ngày\s+xuân\b",),
    "section:kieu_o_lau_ngung_bich": (r"\blầu\s+ngưng\s+bích\b",),
    "section:bao_an_bao_oan": (r"\bbáo\s+ân\b.*\bbáo\s+oán\b",),
    "event:gap_kim_trong": (r"\bgặp\s+kim\s+trọng\b",),
    "event:ban_minh_chuoc_cha": (r"\bbán\s+mình\b", r"\bchuộc\s+cha\b"),
    "event:doan_vien": (r"\bđoàn\s+viên\b",),
    "allusion:quat_nong_ap_lanh": (r"\bquạt\s+nồng\b", r"\bấp\s+lạnh\b"),
    "allusion:san_lai": (r"\bsân\s+lai\b",),
    "archaic:phong_tinh": (r"\bphong\s+tình\b",),
    "archaic:thanh_minh": (r"\bthanh\s+minh\b",),
}


def normalize_prose(text: str) -> str:
    value = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_poem(text: str) -> str:
    value = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    return "\n".join(re.sub(r"[ \t]+$", "", line) for line in value.splitlines()).strip()


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    return [match.span() for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.DOTALL)]


def _sentence_spans(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    segment = text[start:end]
    spans = []
    for match in re.finditer(r"\S.*?(?:[.!?…:;](?=\s|\Z)|\Z)", segment, flags=re.DOTALL):
        left, right = match.span()
        spans.append((start + left, start + right))
    return spans or [(start, end)]


def _pack_spans(text: str, spans: Iterable[Tuple[int, int]], max_words: int) -> Iterator[Dict[str, object]]:
    current_start = current_end = None
    for start, end in spans:
        proposed_start = start if current_start is None else current_start
        proposed = text[proposed_start:end].strip()
        if current_start is not None and len(proposed.split()) > max_words:
            yield {
                "text": text[current_start:current_end].strip(),
                "char_start": current_start,
                "char_end": current_end,
            }
            current_start, current_end = start, end
        else:
            current_start, current_end = proposed_start, end
    if current_start is not None and current_end is not None:
        yield {
            "text": text[current_start:current_end].strip(),
            "char_start": current_start,
            "char_end": current_end,
        }


def split_prose(text: str, max_words: int = PROSE_MAX_WORDS) -> List[Dict[str, object]]:
    blocks: List[Dict[str, object]] = []
    short_spans: List[Tuple[int, int]] = []

    def flush_short() -> None:
        nonlocal short_spans
        blocks.extend(_pack_spans(text, short_spans, max_words))
        short_spans = []

    for start, end in _paragraph_spans(text):
        if len(text[start:end].split()) <= max_words:
            short_spans.append((start, end))
            continue
        flush_short()
        blocks.extend(_pack_spans(text, _sentence_spans(text, start, end), max_words))
    flush_short()
    return blocks


def split_poem(text: str) -> List[Dict[str, object]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks: List[Dict[str, object]] = []
    step = max(1, POEM_LINES_PER_BLOCK - POEM_OVERLAP_LINES)
    for index in range(0, len(lines), step):
        selected = lines[index : index + POEM_LINES_PER_BLOCK]
        if selected:
            blocks.append(
                {
                    "text": "\n".join(selected),
                    "line_start": index + 1,
                    "line_end": index + len(selected),
                }
            )
    return blocks


@lru_cache(maxsize=1)
def _motif_ranges() -> List[Tuple[int, int, str]]:
    try:
        payload = json.loads(MOTIF_FILE.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return []
    ranges = []
    for item in payload:
        bounds = item.get("range") or []
        if len(bounds) == 2 and item.get("motif"):
            ranges.append((int(bounds[0]), int(bounds[1]), str(item["motif"])))
    return ranges


def _section_for_lines(line_start: int, line_end: int) -> str:
    for start, end, motif in _motif_ranges():
        if line_start <= end and line_end >= start:
            return motif
    return ""


def _slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFD", value)
    ascii_value = "".join(ch for ch in ascii_value if unicodedata.category(ch) != "Mn")
    ascii_value = ascii_value.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_")


def extract_tags(text: str, doc_type: str) -> List[str]:
    tags = {f"type:{doc_type}"}
    for tag, patterns in TAG_PATTERNS.items():
        if any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns):
            tags.add(tag)
    return sorted(tags)


def _source_url(text: str) -> str:
    match = re.search(r"<!--\s*source:\s*(https?://[^\s>]+)\s*-->", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _document_title(raw: str, fallback: str) -> str:
    for line in raw.splitlines()[:30]:
        candidate = re.sub(r"^#{1,6}\s*", "", line).strip()
        if candidate and not candidate.startswith("<!--") and 5 <= len(candidate) <= 180:
            return candidate
    return fallback


def _source_files() -> Iterator[Path]:
    for path in sorted(SRC.rglob("*.txt")):
        relative = path.relative_to(SRC)
        if not relative.parts or relative.parts[0] not in TYPE_BY_DIR:
            continue
        if relative.parts[0] == "poem" and path.resolve() != CANONICAL_POEM.resolve():
            continue
        yield path


def _base_meta(path: Path, doc_type: str, raw: str) -> Dict[str, object]:
    relative = path.relative_to(SRC)
    is_poem = doc_type == "poem"
    return {
        "source": str(relative).replace("\\", "/"),
        "source_id": "truyen-kieu-canonical" if is_poem else path.stem,
        "title": "Truyện Kiều — văn bản chuẩn của dự án" if is_poem else _document_title(raw, path.stem),
        "type": doc_type,
        "source_url": _source_url(raw),
        "author": "Nguyễn Du" if is_poem else "Chưa xác định",
        "work": "Truyện Kiều" if is_poem else path.stem,
        "edition": "project-canonical-v1" if is_poem else "",
        "source_tier": "primary" if is_poem else "",
    }


def _chunk_id(source_id: str, index: int, text: str, meta: Dict[str, object]) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    if meta.get("type") == "poem":
        return f"{source_id}_L{int(meta['line_start']):04d}-{int(meta['line_end']):04d}_{digest}"
    return f"{source_id}_{index:04d}_{digest}"


def build(output: Path) -> Dict[str, int]:
    output = safe_chunk_dir(output, ROOT)
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("*.txt"):
        stale.unlink()

    source_count = chunk_count = duplicate_count = 0
    seen_hashes: set[str] = set()
    for path in _source_files():
        raw = path.read_text(encoding="utf-8", errors="ignore")
        doc_type = TYPE_BY_DIR[path.relative_to(SRC).parts[0]]
        text = normalize_poem(raw) if doc_type == "poem" else normalize_prose(raw)
        blocks = split_poem(text) if doc_type == "poem" else split_prose(text)
        base = _base_meta(path, doc_type, raw)
        for index, block in enumerate(blocks):
            body = str(block["text"]).strip()
            meta = dict(base)
            meta.update({key: value for key, value in block.items() if key != "text"})
            meta["chunk_index"] = index
            meta["tags"] = extract_tags(body, doc_type)
            if doc_type == "poem":
                section = _section_for_lines(int(meta["line_start"]), int(meta["line_end"]))
                meta["section"] = section
                if section:
                    meta["tags"].append(f"section:{_slug(section)}")
            meta = enrich_metadata(meta, body)
            digest = str(meta["content_hash"])
            if digest in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(digest)
            meta["id"] = _chunk_id(str(meta["source_id"]), index, body, meta)
            destination = output / f"{meta['id']}.txt"
            header = "###META### " + json.dumps(meta, ensure_ascii=False, sort_keys=True)
            destination.write_text(f"{header}\n{body}\n", encoding="utf-8")
            chunk_count += 1
        source_count += 1
    return {"sources": source_count, "chunks": chunk_count, "duplicates_skipped": duplicate_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_DST)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({**result, "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
