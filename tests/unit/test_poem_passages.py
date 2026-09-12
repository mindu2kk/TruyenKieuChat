import pytest

from app.poem_passages import context_for_range


@pytest.mark.unit
def test_context_for_1795_1796_identifies_thuc_sinh_not_kieu():
    context = context_for_range(1795, 1796)

    assert context is not None
    assert "Thúc Sinh" in context.summary
    assert "tưởng Thúy Kiều đã chết" in context.summary
    assert "không phải tâm trạng Kiều" in context.summary
    assert "**Bối cảnh:**" in context.close_reading
    assert "**Nghệ thuật:**" in context.close_reading
    assert "**sầu dài – ngày ngắn**" in context.close_reading


@pytest.mark.unit
def test_unknown_passage_does_not_invent_context():
    assert context_for_range(1, 2) is None
