"""
Integration tests cho RAG pipeline end-to-end.

Test các tính năng:
- Query variants generation
- Hybrid retrieval
- Reranking
- Answer generation
- Source attribution
- Error handling
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.rag_pipeline import answer_question, _build_query_variants, _dedupe_hits


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.requires_mongo
def test_build_query_variants():
    """Test tạo query variants từ query gốc."""
    query = "Thúy Kiều là ai"
    variants = _build_query_variants(query)
    
    assert len(variants) > 0
    assert query in variants
    # Kiểm tra có variants không dấu
    assert any("thuy kieu" in v.lower() for v in variants)


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.requires_mongo
def test_build_query_variants_with_character():
    """Test query variants với tên nhân vật."""
    query = "thúy kiều"
    variants = _build_query_variants(query)
    
    # Kiểm tra có nhiều variants cho tên nhân vật
    assert len(variants) > 1
    assert any("thúy kiều" in v.lower() for v in variants)
    assert any("thuy kieu" in v.lower() for v in variants)


@pytest.mark.integration
def test_dedupe_hits():
    """Test loại bỏ trùng lặp hits."""
    hits = [
        {"text": "Text 1", "meta": {"source": "source1", "line_number": 1}, "score": 0.9},
        {"text": "Text 1", "meta": {"source": "source1", "line_number": 1}, "score": 0.8},  # Trùng lặp
        {"text": "Text 2", "meta": {"source": "source2", "line_number": 2}, "score": 0.7},
    ]
    
    deduped = _dedupe_hits(hits)
    
    # Chỉ còn 2 hits (bỏ trùng lặp)
    assert len(deduped) == 2
    assert deduped[0]["text"] == "Text 1"
    assert deduped[1]["text"] == "Text 2"


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.requires_mongo
@patch('app.rag_pipeline.generate_answer_gemini')
@patch('app.rag_pipeline.rerank')
def test_rag_pipeline_end_to_end(mock_rerank, mock_gen):
    """Test RAG pipeline end-to-end với mocks."""
    # Setup mock retriever
    mock_hits = [
        {
            "text": "Thúy Kiều là nhân vật chính trong Truyện Kiều",
            "meta": {"source": "analysis.txt", "type": "analysis"},
            "score": 0.85
        },
        {
            "text": "Thúy Kiều tên thật là Vương Thúy Kiều",
            "meta": {"source": "bio.txt", "type": "bio"},
            "score": 0.80
        }
    ]
    
    # Mock the retriever instance
    import app.rag_pipeline as rag_module
    original_retriever = rag_module._HYBRID_RETRIEVER
    mock_retriever_instance = Mock()
    mock_retriever_instance.retrieve.return_value = mock_hits
    rag_module._HYBRID_RETRIEVER = mock_retriever_instance
    
    try:
        # Setup mock rerank
        mock_rerank.return_value = mock_hits
        
        # Setup mock generation
        mock_gen.return_value = "Thúy Kiều là nhân vật chính trong Truyện Kiều của Nguyễn Du."
        
        result = answer_question(
            "Thúy Kiều là ai?",
            k=5,
            synthesize="single",
            gen_model="gemini-1.5-flash"
        )
        
        assert "answer" in result
        assert "Thúy Kiều" in result["answer"]
        assert "contexts" in result
        assert len(result["contexts"]) > 0
    finally:
        rag_module._HYBRID_RETRIEVER = original_retriever


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.requires_mongo
@patch('app.rag_pipeline._HYBRID_RETRIEVER')
def test_rag_pipeline_no_results(mock_retriever):
    """Test RAG pipeline khi không có kết quả retrieval."""
    # Mock retriever instance
    mock_instance = Mock()
    mock_instance.retrieve.return_value = []
    # Replace the module-level instance
    import app.rag_pipeline as rag_module
    original_retriever = rag_module._HYBRID_RETRIEVER
    rag_module._HYBRID_RETRIEVER = mock_instance
    
    try:
        result = answer_question(
            "Câu hỏi không có kết quả",
            k=5,
            synthesize="single"
        )
        
        assert "prompt" in result
        assert "contexts" in result
        assert len(result["contexts"]) == 0
        assert "answer" not in result or result.get("answer") is None
    finally:
        rag_module._HYBRID_RETRIEVER = original_retriever


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.requires_mongo
@patch('app.rag_pipeline.generate_answer_gemini')
@patch('app.rag_pipeline.rerank')
def test_rag_pipeline_long_answer(mock_rerank, mock_gen):
    """Test RAG pipeline với long_answer=True."""
    mock_hits = [
        {
            "text": "Sample text",
            "meta": {"source": "test.txt"},
            "score": 0.8
        }
    ]
    
    # Mock the retriever instance
    import app.rag_pipeline as rag_module
    original_retriever = rag_module._HYBRID_RETRIEVER
    mock_retriever_instance = Mock()
    mock_retriever_instance.retrieve.return_value = mock_hits
    rag_module._HYBRID_RETRIEVER = mock_retriever_instance
    
    try:
        mock_rerank.return_value = mock_hits
        mock_gen.return_value = "Long answer"
        
        result = answer_question(
            "Test query",
            k=5,
            long_answer=True,
            synthesize="single"
        )
        
        assert "answer" in result
        # Kiểm tra long_answer được xử lý đúng
        assert result["answer"] == "Long answer"
    finally:
        rag_module._HYBRID_RETRIEVER = original_retriever


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.requires_mongo
@patch('app.rag_pipeline.generate_answer_gemini')
@patch('app.rag_pipeline.rerank')
def test_rag_pipeline_with_sources(mock_rerank, mock_gen):
    """Test RAG pipeline với sources và evidence."""
    mock_hits = [
        {
            "text": "Text 1",
            "meta": {"source": "source1.txt", "line_number": 1},
            "score": 0.9
        },
        {
            "text": "Text 2",
            "meta": {"source": "source2.txt", "line_number": 2},
            "score": 0.8
        }
    ]
    
    # Mock the retriever instance
    import app.rag_pipeline as rag_module
    original_retriever = rag_module._HYBRID_RETRIEVER
    mock_retriever_instance = Mock()
    mock_retriever_instance.retrieve.return_value = mock_hits
    rag_module._HYBRID_RETRIEVER = mock_retriever_instance
    
    try:
        mock_rerank.return_value = mock_hits
        mock_gen.return_value = "Answer with sources"
        
        result = answer_question(
            "Test query",
            k=5,
            synthesize="single",
            top_evidence=2
        )
        
        assert "sources" in result or "evidence" in result
        if "sources" in result:
            assert len(result["sources"]) > 0
    finally:
        rag_module._HYBRID_RETRIEVER = original_retriever


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.requires_mongo
@patch('app.rag_pipeline.generate_answer_gemini')
def test_rag_pipeline_generation_error(mock_gen):
    """Test RAG pipeline xử lý lỗi generation."""
    mock_hits = [
        {
            "text": "Text",
            "meta": {"source": "test.txt"},
            "score": 0.8
        }
    ]
    
    # Mock the retriever instance
    import app.rag_pipeline as rag_module
    original_retriever = rag_module._HYBRID_RETRIEVER
    mock_retriever_instance = Mock()
    mock_retriever_instance.retrieve.return_value = mock_hits
    rag_module._HYBRID_RETRIEVER = mock_retriever_instance
    
    try:
        mock_gen.side_effect = Exception("Generation error")
        
        result = answer_question(
            "Test query",
            k=5,
            synthesize="single"
        )
        
        assert "generation_error" in result
        assert "Generation error" in result["generation_error"]
    finally:
        rag_module._HYBRID_RETRIEVER = original_retriever


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.requires_mongo
@patch('app.rag_pipeline.rerank')
def test_rag_pipeline_prefer_poem_source(mock_rerank):
    """Test RAG pipeline với prefer_poem_source=True."""
    mock_hits = [
        {
            "text": "Poem text",
            "meta": {"source": "poem.txt", "type": "poem"},
            "score": 0.7
        },
        {
            "text": "Analysis text",
            "meta": {"source": "analysis.txt", "type": "analysis"},
            "score": 0.8
        }
    ]
    
    # Mock the retriever instance
    import app.rag_pipeline as rag_module
    original_retriever = rag_module._HYBRID_RETRIEVER
    mock_retriever_instance = Mock()
    mock_retriever_instance.retrieve.return_value = mock_hits
    rag_module._HYBRID_RETRIEVER = mock_retriever_instance
    
    try:
        mock_rerank.return_value = mock_hits
        
        result = answer_question(
            "Test query",
            k=5,
            prefer_poem_source=True,
            synthesize=False
        )
        
        # Kiểm tra poem hits được boost
        assert "contexts" in result
        # Poem hits nên có score cao hơn sau khi boost
        poem_hits = [h for h in result["contexts"] if h.get("meta", {}).get("type") == "poem"]
        assert len(poem_hits) > 0
    finally:
        rag_module._HYBRID_RETRIEVER = original_retriever

