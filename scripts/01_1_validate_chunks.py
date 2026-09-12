# scripts/validate_chunks.py
# -*- coding: utf-8 -*-
"""
Validator cho data/rag_chunks:
- Đọc dòng ###META### {json}
- Thơ (type=poem): bắt buộc line_start>=1, line_end>=line_start
- Văn xuôi (analysis/summary/bio): bắt buộc char_start>=0, char_end>char_start
- Kiểm tra thêm: id/source/source_id/type/tags định dạng hợp lệ
- Báo cáo tóm tắt + liệt kê lỗi. Có --json để in JSON report.
- Trả mã thoát 1 nếu phát hiện lỗi.

Cách dùng:
  python scripts/validate_chunks.py
  python scripts/validate_chunks.py --json > chunk_report.json
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CHUNK_DIR = ROOT / "data" / "rag_chunks_clean"
LEGACY_CHUNK_DIR = ROOT / "data" / "rag_chunks"
DEFAULT_CHUNK_DIR = CLEAN_CHUNK_DIR if CLEAN_CHUNK_DIR.exists() else LEGACY_CHUNK_DIR

VALID_TYPES = {"poem", "analysis", "summary", "bio"}

@dataclass
class ItemError:
    file: str
    reason: str
    meta: Dict[str, Any]

@dataclass
class Report:
    total_files: int
    poem_ok: int
    poem_err: int
    prose_ok: int
    prose_err: int
    other_type: int
    errors: List[ItemError]

def _load_meta_first_line(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
        if not first.startswith("###META###"):
            return None
        json_part = first.replace("###META###", "", 1).strip()
        return json.loads(json_part)
    except Exception:
        return None

def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)

def _validate_basic(meta: Dict[str, Any]) -> List[str]:
    errs = []
    if not isinstance(meta, dict):
        return ["meta không phải dict"]
    if not meta.get("id"):
        errs.append("thiếu id")
    if not meta.get("source"):
        errs.append("thiếu source")
    if not meta.get("source_id"):
        errs.append("thiếu source_id")
    t = meta.get("type")
    if t not in VALID_TYPES:
        errs.append(f"type không hợp lệ: {t!r}")
    tags = meta.get("tags")
    if not isinstance(tags, list):
        errs.append("tags không phải list")
    for field in (
        "content_hash",
        "source_tier",
        "author",
        "publisher",
        "work",
        "section",
        "characters",
        "events",
        "allusions",
        "archaic_terms",
        "literary_devices",
    ):
        if field not in meta:
            errs.append(f"thiếu metadata chuẩn: {field}")
    if meta.get("source_tier") not in {"primary", "scholarly", "educational", "reference"}:
        errs.append(f"source_tier không hợp lệ: {meta.get('source_tier')!r}")
    return errs

def _validate_poem(meta: Dict[str, Any]) -> List[str]:
    errs = []
    ls = meta.get("line_start")
    le = meta.get("line_end")
    if not _is_int(ls):
        errs.append("line_start không phải int")
    if not _is_int(le):
        errs.append("line_end không phải int")
    if _is_int(ls) and ls < 1:
        errs.append("line_start < 1")
    if _is_int(ls) and _is_int(le) and le < ls:
        errs.append("line_end < line_start")
    # id gợi ý đúng pattern Lxxxx-yyyy
    _id = meta.get("id", "")
    if isinstance(_id, str) and "L" in _id:
        if not re.search(r"_L\d{4}-\d{4}_", _id):
            errs.append("id không theo pattern thơ (_L####-####_)")
    return errs

def _validate_prose(meta: Dict[str, Any]) -> List[str]:
    errs = []
    cs = meta.get("char_start")
    ce = meta.get("char_end")
    if not _is_int(cs):
        errs.append("char_start không phải int")
    if not _is_int(ce):
        errs.append("char_end không phải int")
    if _is_int(cs) and cs < 0:
        errs.append("char_start < 0")
    if _is_int(cs) and _is_int(ce) and ce <= cs:
        errs.append("char_end <= char_start")
    return errs

def _validate_file(path: Path) -> tuple[str, bool, Optional[ItemError]]:
    meta = _load_meta_first_line(path)
    if meta is None:
        return "prose", False, ItemError(str(path), "không tìm thấy dòng ###META### hoặc JSON lỗi", {})
    doc_type = str(meta.get("type") or "other")
    problems = _validate_basic(meta)
    if not problems:
        if doc_type == "poem":
            problems = _validate_poem(meta)
        elif doc_type in {"analysis", "summary", "bio"}:
            problems = _validate_prose(meta)
    category = "poem" if doc_type == "poem" else ("prose" if doc_type in VALID_TYPES else "other")
    error = ItemError(str(path), "; ".join(problems), meta) if problems else None
    return category, not problems, error


def validate_chunks(chunk_dir: Path = DEFAULT_CHUNK_DIR) -> Report:
    files = sorted(chunk_dir.glob("*.txt"))
    poem_ok = poem_err = prose_ok = prose_err = other_type = 0
    errors: List[ItemError] = []

    workers = min(32, max(4, (os.cpu_count() or 1) * 2))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="chunk-validation") as executor:
        results = executor.map(_validate_file, files)
        for category, valid, error in results:
            if error is not None:
                errors.append(error)
            if category == "poem":
                poem_ok += int(valid)
                poem_err += int(not valid)
            elif category == "prose":
                prose_ok += int(valid)
                prose_err += int(not valid)
            else:
                other_type += 1

    return Report(
        total_files=len(files),
        poem_ok=poem_ok,
        poem_err=poem_err,
        prose_ok=prose_ok,
        prose_err=prose_err,
        other_type=other_type,
        errors=errors,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    args = parser.parse_args()
    json_mode = args.json
    rpt = validate_chunks(args.chunk_dir)

    if json_mode:
        out = asdict(rpt)
        out["errors"] = [asdict(e) for e in rpt.errors]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("=== CHUNK VALIDATION SUMMARY ===")
        print(f"Total chunk files : {rpt.total_files}")
        print(f"Poem    OK/ERR    : {rpt.poem_ok} / {rpt.poem_err}")
        print(f"Prose   OK/ERR    : {rpt.prose_ok} / {rpt.prose_err}")
        if rpt.other_type:
            print(f"Other types       : {rpt.other_type}")
        print()
        if rpt.errors:
            print(f"ERROR: Found {len(rpt.errors)} errors. Showing up to 40:")
            for e in rpt.errors[:40]:
                print(f"- {e.file}: {e.reason}")
        else:
            print("OK: No errors detected.")

    # Exit code: 1 if any errors
    sys.exit(1 if rpt.errors else 0)

if __name__ == "__main__":
    main()
