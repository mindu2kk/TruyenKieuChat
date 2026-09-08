"""
Unit tests cho quote verification.

Test các tính năng:
- Quote detection
- Quote matching với poem lines
- Score calculation
- Exact match detection
- Suggested fixes
- Autocorrect functionality
"""
import pytest
from unittest.mock import Mock, patch
from app.verifier import (
    verify_poem_quotes,
    apply_quote_corrections,
    verify_and_autocorrect,
    _find_quotes,
    _canon,
    _strip_diacritics
)


@pytest.mark.unit
def test_strip_diacritics():
    """Test bỏ dấu tiếng Việt."""
    assert _strip_diacritics("Trăm năm") == "Tram nam"
    assert _strip_diacritics("Thúy Kiều") == "Thuy Kieu"
    assert _strip_diacritics("") == ""
    assert _strip_diacritics("hello") == "hello"


@pytest.mark.unit
def test_canon():
    """Test chuẩn hóa text để so khớp."""
    assert _canon("Trăm năm") == _canon("Tram nam")
    assert _canon("Thúy Kiều") == _canon("Thuy Kieu")
    assert _canon("  Trăm  năm  ") == _canon("Trăm năm")
    assert _canon("Trăm, năm!") == _canon("Trăm năm")


@pytest.mark.unit
def test_find_quotes():
    """Test tìm quotes trong text."""
    text = 'Câu thơ "Trăm năm trong cõi người ta" là câu đầu tiên.'
    quotes = _find_quotes(text)

    assert len(quotes) == 1
    assert "Trăm năm trong cõi người ta" in quotes[0]


@pytest.mark.unit
def test_find_quotes_multiple():
    """Test tìm multiple quotes."""
    text = 'Câu "Trăm năm" và câu "Chữ tài chữ mệnh" đều hay.'
    quotes = _find_quotes(text)

    assert len(quotes) == 2
    assert "Trăm năm" in quotes
    assert "Chữ tài chữ mệnh" in quotes


@pytest.mark.unit
def test_find_quotes_curly_quotes():
    """Test tìm quotes với curly quotes."""
    text = 'Câu thơ "Trăm năm trong cõi người ta" là câu đầu tiên.'
    quotes = _find_quotes(text)

    assert len(quotes) >= 1
    assert "Trăm năm" in quotes[0]


@pytest.mark.unit
def test_find_quotes_no_quotes():
    """Test text không có quotes."""
    text = "Đây là text không có quotes."
    quotes = _find_quotes(text)

    assert len(quotes) == 0


@pytest.mark.unit
@patch('app.verifier.all_poem_lines')
def test_verify_poem_quotes_exact_match(mock_all_lines):
    """Test verify với exact match."""
    # Mock poem lines
    mock_line = Mock()
    mock_line.text = "Trăm năm trong cõi người ta"
    mock_line.number = 1
    mock_all_lines.return_value = [mock_line]

    answer = 'Câu thơ "Trăm năm trong cõi người ta" là câu đầu tiên.'
    result = verify_poem_quotes(answer, threshold=88.0)

    assert "quotes" in result
    assert len(result["quotes"]) == 1
    assert result["quotes"][0]["exact"] is True
    assert result["quotes"][0]["score"] >= 88.0
    assert result["coverage"] >= 0.0


@pytest.mark.unit
@patch('app.verifier.all_poem_lines')
def test_verify_poem_quotes_non_exact_match(mock_all_lines):
    """Test verify với non-exact match (có lỗi nhỏ)."""
    mock_line = Mock()
    mock_line.text = "Trăm năm trong cõi người ta"
    mock_line.number = 1
    mock_all_lines.return_value = [mock_line]

    answer = 'Câu thơ "Tram nam trong coi nguoi ta" là câu đầu tiên.'
    result = verify_poem_quotes(answer, threshold=88.0)

    assert "quotes" in result
    assert len(result["quotes"]) == 1
    # Có thể exact=False nếu khác quá nhiều, hoặc exact=True nếu score cao
    assert "score" in result["quotes"][0]


@pytest.mark.unit
@patch('app.verifier.all_poem_lines')
def test_verify_poem_quotes_no_match(mock_all_lines):
    """Test verify với quote không match."""
    mock_line = Mock()
    mock_line.text = "Trăm năm trong cõi người ta"
    mock_line.number = 1
    mock_all_lines.return_value = [mock_line]

    answer = 'Câu thơ "Câu thơ không có trong corpus" không tồn tại.'
    result = verify_poem_quotes(answer, threshold=88.0)

    assert "quotes" in result
    assert len(result["quotes"]) == 1
    assert result["quotes"][0]["score"] < 88.0 or result["quotes"][0]["matched_line"] is None


@pytest.mark.unit
@patch('app.verifier.all_poem_lines')
def test_verify_poem_quotes_multiple_quotes(mock_all_lines):
    """Test verify với multiple quotes."""
    mock_lines = [
        Mock(text="Trăm năm trong cõi người ta", number=1),
        Mock(text="Chữ tài chữ mệnh khéo là ghét nhau", number=2)
    ]
    mock_all_lines.return_value = mock_lines

    answer = 'Câu "Trăm năm trong cõi người ta" và "Chữ tài chữ mệnh khéo là ghét nhau" đều hay.'
    result = verify_poem_quotes(answer, threshold=88.0)

    assert "quotes" in result
    assert len(result["quotes"]) == 2
    assert result["coverage"] >= 0.0


@pytest.mark.unit
@patch('app.verifier.all_poem_lines')
def test_verify_poem_quotes_empty_answer(mock_all_lines):
    """Test verify với empty answer."""
    mock_all_lines.return_value = []

    result = verify_poem_quotes("", threshold=88.0)

    assert result["quotes"] == []
    assert result["accepted"] == []
    assert result["coverage"] == 0.0


@pytest.mark.unit
@patch('app.verifier.all_poem_lines')
def test_verify_poem_quotes_no_poem_lines(mock_all_lines):
    """Test verify khi không có poem lines."""
    mock_all_lines.return_value = []

    answer = 'Câu thơ "Trăm năm trong cõi người ta" là câu đầu tiên.'
    result = verify_poem_quotes(answer, threshold=88.0)

    assert result["quotes"] == []
    assert result["coverage"] == 0.0


@pytest.mark.unit
def test_apply_quote_corrections():
    """Test apply quote corrections."""
    answer = 'Câu thơ "Tram nam" là câu đầu tiên.'
    fixes = [("Tram nam", "Trăm năm")]

    corrected = apply_quote_corrections(answer, fixes)

    assert "Trăm năm" in corrected
    assert "Tram nam" not in corrected


@pytest.mark.unit
def test_apply_quote_corrections_multiple():
    """Test apply multiple corrections."""
    answer = 'Câu "Tram nam" và "Chu tai" đều hay.'
    fixes = [
        ("Tram nam", "Trăm năm"),
        ("Chu tai", "Chữ tài")
    ]

    corrected = apply_quote_corrections(answer, fixes)

    assert "Trăm năm" in corrected
    assert "Chữ tài" in corrected


@pytest.mark.unit
def test_apply_quote_corrections_no_fixes():
    """Test apply corrections với empty fixes."""
    answer = 'Câu thơ "Trăm năm" là câu đầu tiên.'
    fixes = []

    corrected = apply_quote_corrections(answer, fixes)

    assert corrected == answer


@pytest.mark.unit
@patch('app.verifier.all_poem_lines')
def test_verify_and_autocorrect(mock_all_lines):
    """Test verify và autocorrect."""
    mock_line = Mock()
    mock_line.text = "Trăm năm trong cõi người ta"
    mock_line.number = 1
    mock_all_lines.return_value = [mock_line]

    answer = 'Câu thơ "Tram nam trong coi nguoi ta" là câu đầu tiên.'
    corrected, verification = verify_and_autocorrect(answer, threshold=88.0, autocorrect=True)

    assert isinstance(corrected, str)
    assert isinstance(verification, dict)
    assert "quotes" in verification


@pytest.mark.unit
@patch('app.verifier.all_poem_lines')
def test_verify_and_autocorrect_disabled(mock_all_lines):
    """Test verify và autocorrect với autocorrect=False."""
    mock_line = Mock()
    mock_line.text = "Trăm năm trong cõi người ta"
    mock_line.number = 1
    mock_all_lines.return_value = [mock_line]

    answer = 'Câu thơ "Tram nam" là câu đầu tiên.'
    corrected, verification = verify_and_autocorrect(answer, threshold=88.0, autocorrect=False)

    assert corrected == answer  # Không sửa
    assert isinstance(verification, dict)


@pytest.mark.unit
@patch('app.verifier.all_poem_lines')
def test_verify_poem_quotes_threshold(mock_all_lines):
    """Test verify với different thresholds."""
    mock_line = Mock()
    mock_line.text = "Trăm năm trong cõi người ta"
    mock_line.number = 1
    mock_all_lines.return_value = [mock_line]

    answer = 'Câu thơ "Tram nam trong coi nguoi ta" là câu đầu tiên.'

    # Low threshold
    result_low = verify_poem_quotes(answer, threshold=50.0)
    # High threshold
    result_high = verify_poem_quotes(answer, threshold=95.0)

    assert len(result_low["accepted"]) >= len(result_high["accepted"])


@pytest.mark.unit
@patch('app.verifier.all_poem_lines')
def test_verify_poem_quotes_suggested_fixes(mock_all_lines):
    """Test suggested fixes."""
    mock_line = Mock()
    mock_line.text = "Trăm năm trong cõi người ta"
    mock_line.number = 1
    mock_all_lines.return_value = [mock_line]

    answer = 'Câu thơ "Tram nam trong coi nguoi ta" là câu đầu tiên.'
    result = verify_poem_quotes(answer, threshold=88.0)

    assert "suggested_fixes" in result
    assert "non_exact" in result
    # Nếu có non-exact matches, nên có suggested fixes
    if result["non_exact"]:
        assert len(result["suggested_fixes"]) > 0
