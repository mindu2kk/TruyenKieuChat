# Kiều Bot corpus quality runbook

The workflow is intentionally two-phase: build and validate locally first,
then promote the exact clean corpus to MongoDB. Source files and the current
Mongo collection are never deleted by the builder.

## 1. Build the clean corpus

```powershell
python scripts/01_2_clean_chunks.py
python scripts/01_1_validate_chunks.py --chunk-dir data/rag_chunks_clean
```

The recommended cleaner preserves valid legacy IDs and boundaries, uses only
`data/interim/poem/poem.txt` as the canonical poem, repairs prose offsets,
normalizes metadata and removes exact duplicates. This allows existing vectors
to be reused by content hash. `01_build_chunks.py` remains available when a
full rechunk from source documents is intentionally required.

## 2. Run the offline benchmark

```powershell
python scripts/04_build_quality_benchmark.py
python eval/run_quality_eval.py
```

`quality_v1.jsonl` contains 260 cases across exact verse lookup, quote repair,
line ranges, plot/timeline, characters, archaic terms and allusions, literary
analysis, comparisons, ambiguity and out-of-scope behavior. Human-rubric cases
are retained but excluded from the automatic pass rate.

## 3. Embed without pruning production

Set the normal `MONGO_URI`, `MONGO_DB`, `MONGO_COL` and embedding credentials,
then run one of the existing embedding paths. The clean directory is selected
automatically when present, or can be pinned explicitly:

```powershell
$env:RAG_CHUNKS_DIR = "data/rag_chunks_clean"
python scripts/embed_gemini.py
```

Run the Mongo quality sync in dry-run mode:

```powershell
python scripts/03_sync_index_quality.py
python eval/run_quality_eval.py --live-retrieval --max-live-cases 60
```

Review Recall@5, Recall@10, MRR and nDCG@10 before pruning. The sync command
does not modify MongoDB unless `--apply` is provided.

## 4. Promote only after gates pass

```powershell
python scripts/03_sync_index_quality.py --apply
python eval/run_quality_eval.py --live-retrieval
```

This first copies every document selected for deletion to a timestamped backup
collection, then deletes only stale document IDs and lower-priority duplicate
IDs. It does not drop the live collection or its vector index. Keep the backup
until the post-prune benchmark passes.
