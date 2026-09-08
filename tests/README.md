# 🧪 TESTS

## 📁 Cấu trúc thư mục

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── unit/                # Unit tests (nhanh, không cần external services)
│   └── __init__.py
├── integration/         # Integration tests (cần database, API)
│   └── __init__.py
└── e2e/                 # End-to-end tests (full system)
    └── __init__.py
```

## 🚀 Cách chạy tests

### Chạy tất cả tests:
```bash
pytest tests/
# Hoặc
make test
```

### Chạy tests với coverage:
```bash
pytest --cov=app --cov=chat_UI --cov-report=html
# Hoặc
make test-coverage
```

### Chạy unit tests chỉ:
```bash
pytest tests/unit/ -m unit
# Hoặc
make test-unit
```

### Chạy tests nhanh (bỏ qua slow và requires_api):
```bash
pytest tests/ -m "not slow" -m "not requires_api"
# Hoặc
make test-fast
```

## 📝 Viết tests mới

### Ví dụ: Unit test

Tạo file `tests/unit/test_poem_tools.py`:

```python
import pytest
from app.poem_tools import find_poem_line

@pytest.mark.unit
def test_find_poem_line():
    """Test tìm câu thơ."""
    result = find_poem_line("Trăm năm")
    assert result is not None
    assert "Trăm năm" in result
```

### Ví dụ: Integration test

Tạo file `tests/integration/test_rag_pipeline.py`:

```python
import pytest
from app.rag_pipeline import RAGPipeline

@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.requires_mongo
def test_rag_pipeline(mock_gemini_client, mock_mongo_client, sample_query):
    """Test RAG pipeline."""
    pipeline = RAGPipeline()
    result = pipeline.process_query(sample_query)
    assert result is not None
```

## 🎯 Test Markers

Sử dụng markers để phân loại tests:

- `@pytest.mark.unit` - Unit tests (nhanh)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Tests mất > 5 giây
- `@pytest.mark.requires_api` - Tests cần external API
- `@pytest.mark.requires_db` - Tests cần database
- `@pytest.mark.requires_mongo` - Tests cần MongoDB

## 🔧 Fixtures có sẵn

Trong `conftest.py` có các fixtures:

- `mock_mongo_client` - Mock MongoDB client
- `mock_gemini_client` - Mock Google Gemini API
- `sample_poem_line` - Sample poem line
- `sample_query` - Sample query
- `sample_chunk` - Sample chunk với metadata
- `mock_embedding` - Mock embedding vector
- `env_vars` - Set environment variables

## 📊 Coverage

Coverage threshold: **80%**

Xem coverage report:
```bash
pytest --cov=app --cov=chat_UI --cov-report=html
# Sau đó mở htmlcov/index.html
```

## 📚 Tài liệu tham khảo

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- Xem `HƯỚNG_DẪN_TEST_INFRASTRUCTURE.md` để biết chi tiết
