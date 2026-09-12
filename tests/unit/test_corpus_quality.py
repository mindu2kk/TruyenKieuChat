from pathlib import Path

import pytest

from app.corpus_quality import (
    content_hash,
    deduplicate_records,
    diversify_hits,
    enrich_metadata,
    infer_source_tier,
    safe_chunk_dir,
)
from app.chunk_store import iter_chunks, read_chunk


@pytest.mark.unit
def test_enrich_metadata_adds_traceability_contract():
    meta = enrich_metadata(
        {"type": "poem", "source": "poem/poem.txt", "tags": ["char:thuy_kieu", "device:an_du"]},
        "Trăm năm trong cõi người ta",
    )

    assert meta["author"] == "Nguyễn Du"
    assert meta["work"] == "Truyện Kiều"
    assert meta["source_tier"] == "primary"
    assert meta["characters"] == ["thuy_kieu"]
    assert meta["literary_devices"] == ["an_du"]
    assert len(meta["content_hash"]) == 64


@pytest.mark.unit
def test_source_tier_prefers_scholarship_over_educational_sites():
    assert infer_source_tier({"type": "analysis", "source": "tran-dinh-su-thi-phap.txt"}) == "scholarly"
    assert infer_source_tier({"type": "analysis", "source": "vietjack.com-bai-van.txt"}) == "educational"


@pytest.mark.unit
def test_deduplicate_records_keeps_canonical_poem_source():
    text = "Trăm năm trong cõi người ta"
    records = [
        {"_id": "raw", "text": text, "meta": {"id": "raw", "type": "poem", "source_id": "poem.raw"}},
        {"_id": "canonical", "text": text, "meta": {"id": "canonical", "type": "poem", "source_id": "poem"}},
    ]

    result = list(deduplicate_records(records))

    assert len(result) == 1
    assert result[0]["meta"]["source_id"] == "poem"


@pytest.mark.unit
def test_diversify_hits_caps_one_source_and_removes_duplicate_text():
    hits = [
        {"text": "A", "meta": {"source_id": "one"}},
        {"text": "B", "meta": {"source_id": "one"}},
        {"text": "C", "meta": {"source_id": "one"}},
        {"text": "A", "meta": {"source_id": "two"}},
        {"text": "D", "meta": {"source_id": "two"}},
    ]

    result = diversify_hits(hits, limit=4, max_per_source=2)

    assert [hit["text"] for hit in result] == ["A", "B", "D"]
    assert len({content_hash(hit["text"]) for hit in result}) == 3


@pytest.mark.unit
def test_safe_chunk_dir_rejects_broad_or_external_paths(tmp_path):
    root = tmp_path / "project"
    data = root / "data"
    data.mkdir(parents=True)

    assert safe_chunk_dir(data / "clean", root) == (data / "clean").resolve()
    with pytest.raises(ValueError):
        safe_chunk_dir(data, root)
    with pytest.raises(ValueError):
        safe_chunk_dir(Path(tmp_path) / "outside", root)


@pytest.mark.unit
def test_chunk_store_reads_enriches_and_deduplicates(tmp_path):
    header = '{"id":"a","source":"poem.txt","source_id":"poem","type":"poem","line_start":1,"line_end":1,"tags":[]}'
    (tmp_path / "a.txt").write_text(f"###META### {header}\nTrăm năm trong cõi người ta\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text(
        f"###META### {header.replace(chr(34) + 'a' + chr(34), chr(34) + 'b' + chr(34), 1)}\n"
        "Trăm năm trong cõi người ta\n",
        encoding="utf-8",
    )

    record = read_chunk(tmp_path / "a.txt")
    records = list(iter_chunks(tmp_path))

    assert record is not None
    assert record["meta"]["source_tier"] == "primary"
    assert len(records) == 1
