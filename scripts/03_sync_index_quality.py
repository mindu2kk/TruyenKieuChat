"""Audit or prune stale/duplicate Mongo chunk documents.

Dry-run is the default. Pass ``--apply`` only after the clean corpus has been
embedded successfully. The command never drops a collection or an index.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunk_store import configured_chunk_dir, iter_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Delete stale and lower-quality duplicate documents")
    parser.add_argument(
        "--backup-collection",
        default="",
        help="Collection receiving recoverable copies before --apply (default: <collection>_backup_quality_<UTC timestamp>)",
    )
    args = parser.parse_args()

    load_dotenv()
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise SystemExit("Thiếu MONGO_URI")
    database = MongoClient(uri)[os.getenv("MONGO_DB", "kieu_bot")]
    collection_name = os.getenv("MONGO_COL", "chunks")
    collection = database[collection_name]

    clean_records = list(iter_chunks(configured_chunk_dir()))
    clean_ids = {str(record["_id"]) for record in clean_records}
    remote = list(collection.find({}, {"_id": 1, "meta.content_hash": 1, "meta.source_tier": 1}))
    remote_id_by_string = {str(record["_id"]): record["_id"] for record in remote}
    remote_ids = {str(record["_id"]) for record in remote}
    stale_ids = sorted(remote_ids - clean_ids)
    stale_native_ids = [remote_id_by_string[item] for item in stale_ids]
    embedded_clean = clean_ids & remote_ids
    missing_clean = clean_ids - remote_ids

    by_hash = defaultdict(list)
    for record in remote:
        digest = ((record.get("meta") or {}).get("content_hash") or "").strip()
        if digest:
            by_hash[digest].append(record["_id"])
    duplicate_ids = []
    for ids in by_hash.values():
        ordered = sorted(ids, key=lambda item: (str(item) not in clean_ids, str(item)))
        duplicate_ids.extend(ordered[1:])
    delete_ids = sorted(set(stale_native_ids) | set(duplicate_ids), key=str)

    print(
        f"clean_local={len(clean_ids)} embedded_clean={len(embedded_clean)} missing_clean={len(missing_clean)} "
        f"remote={len(remote_ids)} stale={len(stale_ids)} duplicates={len(duplicate_ids)}"
    )
    if not args.apply:
        print("DRY RUN: no data deleted. Use --apply only after embedding the clean corpus.")
        return
    coverage = len(embedded_clean) / max(1, len(clean_ids))
    if coverage < 0.95:
        raise SystemExit(
            f"REFUSED: only {coverage:.1%} of the clean corpus is present remotely; embed it before pruning."
        )
    if delete_ids:
        backup_name = args.backup_collection or (
            f"{collection_name}_backup_quality_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        backup = database[backup_name]
        backup.create_index("backup.original_id")
        backup_docs = []
        for document in collection.find({"_id": {"$in": delete_ids}}):
            original_id = document.pop("_id")
            document["backup"] = {
                "original_id": original_id,
                "source_collection": collection_name,
                "created_at": datetime.now(timezone.utc),
            }
            backup_docs.append(document)
        if len(backup_docs) != len(delete_ids):
            raise SystemExit(
                f"REFUSED: expected {len(delete_ids)} backup docs, found {len(backup_docs)}."
            )
        if backup_docs:
            backup.insert_many(backup_docs, ordered=False)
        print(f"backup_collection={backup_name} backup_docs={len(backup_docs)}")
        result = collection.delete_many({"_id": {"$in": delete_ids}})
        if result.deleted_count != len(delete_ids):
            raise SystemExit(
                f"WARNING: backed up {len(delete_ids)} docs but deleted {result.deleted_count}; inspect MongoDB."
            )
        print(f"deleted={result.deleted_count}")
    else:
        print("deleted=0")


if __name__ == "__main__":
    main()
