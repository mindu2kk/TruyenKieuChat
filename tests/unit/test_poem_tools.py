from app.poem_tools import all_poem_lines, get_range


def test_poem_corpus_has_all_3254_lines():
    assert len(all_poem_lines()) == 3254


def test_final_range_returns_four_numbered_source_lines():
    assert get_range(3251, 3254) == [
        "Thiện căn ở tại lòng ta,",
        "Chữ tâm kia mới bằng ba chữ tài!",
        "Lời quê chắp nhặt dông dài,",
        "Mua vui cũng được một vài trống canh.",
    ]
