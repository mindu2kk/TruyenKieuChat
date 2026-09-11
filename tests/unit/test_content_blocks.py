import pytest

from chat_UI.content_blocks import build_content_blocks


@pytest.mark.unit
def test_content_blocks_use_verified_poem_line():
    blocks = build_content_blocks(
        "Hai câu thơ cho thấy nỗi buồn của Kiều.",
        query="Phân tích câu thơ về nỗi buồn",
        intent="poem_analysis",
        verification={
            "coverage": 1.0,
            "quotes": [{"quote": "Người buồn cảnh có vui đâu bao giờ"}],
            "accepted": [
                {
                    "quote": "Người buồn cảnh có vui đâu bao giờ",
                    "matched_text": "Người buồn cảnh có vui đâu bao giờ",
                    "matched_line": 1246,
                }
            ],
        },
    )

    verse = next(block for block in blocks if block["type"] == "verse_quote")
    assert verse["verified"] is True
    assert verse["line_start"] == 1246
    assert any(block["type"] == "reference" for block in blocks)


@pytest.mark.unit
def test_content_blocks_add_character_card_from_query():
    blocks = build_content_blocks(
        "Thúy Kiều là nhân vật trung tâm của tác phẩm.",
        query="Thúy Kiều là ai?",
        intent="core_fact",
    )

    card = next(block for block in blocks if block["type"] == "character_card")
    assert card["name"] == "Thúy Kiều"
    assert card["role"] == "Nhân vật trung tâm"
    assert "hiếu thảo" in card["traits"]


@pytest.mark.unit
def test_content_blocks_add_timeline_for_plot_list():
    blocks = build_content_blocks(
        "Các chặng chính:\n\n1. Gặp Kim Trọng\n2. Gia biến\n3. Bán mình chuộc cha",
        query="Tóm tắt dòng truyện",
        intent="plot",
    )

    timeline = next(block for block in blocks if block["type"] == "timeline")
    assert timeline["items"] == ["Gặp Kim Trọng", "Gia biến", "Bán mình chuộc cha"]
