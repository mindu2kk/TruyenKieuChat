"""Promote legacy chunks into a clean corpus while preserving valid IDs/text."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunk_store import read_chunk
from app.corpus_quality import deduplicate_records, enrich_metadata, poem_section_for_line, safe_chunk_dir

LEGACY_DIR = ROOT / "data" / "rag_chunks"
DEFAULT_OUTPUT = ROOT / "data" / "rag_chunks_clean"
INTERIM_DIR = ROOT / "data" / "interim"
CURATED_GLOSSARY = ROOT / "data" / "curated" / "glossary.jsonl"


def _normalize_prose(text: str) -> str:
    value = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _relaxed_span(source: str, body: str) -> tuple[int, int]:
    exact = source.find(body)
    if exact >= 0:
        return exact, exact + len(body)
    tokens = re.findall(r"\S+", body)
    if not tokens:
        return -1, -1
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, source, flags=re.IGNORECASE | re.DOTALL)
    return match.span() if match else (-1, -1)


def _repair_position(meta: dict, text: str) -> bool:
    if meta.get("type") == "poem":
        return (
            isinstance(meta.get("line_start"), int)
            and isinstance(meta.get("line_end"), int)
            and meta["line_start"] >= 1
            and meta["line_end"] >= meta["line_start"]
        )
    start, end = meta.get("char_start"), meta.get("char_end")
    if isinstance(start, int) and isinstance(end, int) and start >= 0 and end > start:
        return True
    source_path = INTERIM_DIR / str(meta.get("source") or "")
    if not source_path.is_file():
        return False
    source = _normalize_prose(source_path.read_text(encoding="utf-8", errors="ignore"))
    start, end = _relaxed_span(source, text)
    if start < 0 or end <= start:
        return False
    meta["char_start"] = start
    meta["char_end"] = end
    meta["position_repaired"] = True
    return True


def _canonicalize_poem(meta: dict) -> None:
    if meta.get("type") != "poem":
        return
    meta.update(
        {
            "source": "poem/poem.txt",
            "source_id": "poem",
            "title": "Truyện Kiều — văn bản chuẩn của dự án",
            "author": "Nguyễn Du",
            "work": "Truyện Kiều",
            "edition": "project-canonical-v1",
            "source_tier": "primary",
        }
    )
    section = poem_section_for_line(int(meta.get("line_start") or 0))
    if section:
        section_id, section_title = section
        meta["section"] = section_id
        meta["section_title"] = section_title
        meta["events"] = list(dict.fromkeys([*meta.get("events", []), section_title]))


def clean(output: Path) -> dict:
    output = safe_chunk_dir(output, ROOT)
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("*.txt"):
        stale.unlink()

    candidates = []
    skipped_raw_poem = invalid = repaired = 0
    for path in sorted(LEGACY_DIR.glob("*.txt")):
        record = read_chunk(path)
        if record is None:
            invalid += 1
            continue
        meta = dict(record["meta"])
        if meta.get("type") == "poem" and str(meta.get("source_id") or "").lower() == "poem.raw":
            skipped_raw_poem += 1
            continue
        _canonicalize_poem(meta)
        had_valid_position = _repair_position(meta, record["text"])
        if not had_valid_position:
            invalid += 1
            continue
        repaired += int(bool(meta.get("position_repaired")))
        record["meta"] = enrich_metadata(meta, record["text"])
        candidates.append(record)

    curated_added = 0
    if CURATED_GLOSSARY.is_file():
        for line in CURATED_GLOSSARY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            text = str(item.pop("text")).strip()
            item.setdefault("type", "analysis")
            item.setdefault("source", "curated/glossary.jsonl")
            item.setdefault("char_start", 0)
            item.setdefault("char_end", len(text))
            item.setdefault("tags", [])
            item["tags"] = list(dict.fromkeys([*item["tags"], "section:glossary"]))
            item["id"] = str(item["id"])
            candidates.append({"_id": item["id"], "text": text, "meta": enrich_metadata(item, text)})
            curated_added += 1

    clean_records = list(deduplicate_records(candidates))
    for record in clean_records:
        meta = record["meta"]
        destination = output / f"{meta['id']}.txt"
        header = "###META### " + json.dumps(meta, ensure_ascii=False, sort_keys=True)
        destination.write_text(f"{header}\n{record['text']}\n", encoding="utf-8")
    return {
        "legacy": len(list(LEGACY_DIR.glob("*.txt"))),
        "clean": len(clean_records),
        "duplicates_removed": len(candidates) - len(clean_records),
        "raw_poem_removed": skipped_raw_poem,
        "positions_repaired": repaired,
        "invalid_dropped": invalid,
        "curated_glossary_added": curated_added,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(clean(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
