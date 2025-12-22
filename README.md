# 📚 Kieu-Bot - RAG System for Vietnamese Literature

> AI-powered Q&A system for "Truyện Kiều" using advanced RAG techniques with hybrid search, multi-stage reranking, and quote verification.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)

---

## 🎯 Overview

An intelligent chatbot that answers questions about Vietnam's classic literary masterpiece "Truyện Kiều" (3,254 verses). Built with production-grade RAG architecture optimized for Vietnamese language and classical poetry.

**Key Features:**
- 🔍 Hybrid search (vector + text) with RRF fusion
- 🎯 Intent-based routing for optimized processing
- ✅ Automatic quote verification and correction
- 🇻🇳 Vietnamese-optimized (diacritics, name variants)
- 📊 Multi-stage reranking for precision
- ⚡ Smart caching (LRU + TTL)

---

## 🛠️ Tech Stack

**Core:** Python 3.10+ • MongoDB Atlas • Google Gemini API  
**ML/AI:** Sentence Transformers (SBERT/E5) • Cross-encoder • RapidFuzz  
**Framework:** Streamlit • PyMongo • Pytest

**Architecture Highlights:**
- Modular RAG pipeline with clean separation of concerns
- Progressive query relaxation for better recall
- Multi-model reranking support (Cross-encoder, BGE, Cohere)
- Fuzzy matching for quote validation

---

## 🚀 Quick Start

```bash
# 1. Setup
git clone https://github.com/yourusername/kieu-bot.git
cd kieu-bot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your MONGO_URI and GOOGLE_API_KEY

# 2. Prepare data
python scripts/01_build_chunks.py
python scripts/02_embed_and_index_mongo.py
# Create Vector Search Index in MongoDB Atlas (use scripts/03_create_mongo_vector_index.js)

# 3. Run
streamlit run app/ui_streamlit.py
```

**Requirements:** Python 3.10+, MongoDB Atlas (free tier), Google API key, 4GB RAM

---

## 💡 Usage Examples

```python
# Character query
"Thúy Kiều là ai?" → Detailed character analysis

# Verse retrieval
"Trích 10 câu đầu" → First 10 verses with line numbers

# Literary analysis
"Phân tích ẩn dụ trong câu 100" → Deep analysis with citations

# Comparison
"So sánh câu 1 với câu 100" → Comparative analysis
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Response Time | ~2-3s |
| Cache Hit Rate | ~40-50% |
| Quote Accuracy | ~95% |
| Retrieval Precision | ~85% |

---

## 🏗️ Architecture

```
UI (Streamlit)
    ↓
Orchestrator → [FAQ|Chat|Poem|Generic|Domain]
    ↓
RAG Pipeline → Query Expansion → Hybrid Search → Rerank → Generate
    ↓
MongoDB Atlas + Gemini API + Quote Verifier
```

**Key Components:**
- `orchestrator.py` - Intent routing & caching
- `rag_pipeline.py` - Core RAG implementation
- `hybrid_retriever.py` - Vector + text search with RRF
- `rerank.py` - Multi-model reranking
- `verifier.py` - Quote validation system

---

## 📁 Project Structure

```
kieu-bot/
├── app/                    # Core application
│   ├── orchestrator.py    # Main coordinator
│   ├── rag_pipeline.py    # RAG engine
│   ├── hybrid_retriever.py # Search layer
│   └── ...
├── scripts/               # Data processing
├── tests/                 # Unit & integration tests
└── data/                  # Raw, interim, chunks
```

---

## 🧪 Testing

```bash
pytest                              # Run all tests
pytest --cov=app --cov-report=html # With coverage
```

---

## 🔮 Roadmap

**Current (v1.0):** ✅ Core RAG • ✅ Hybrid search • ✅ Quote verification • ✅ UI

**Next (v2.0):** REST API • Multi-turn dialogue • Analytics • Mobile UI • TTS

---

## 👤 Author

**[Your Name]**  
GitHub: [mindu2kk](https://github.com/mindu2kk)  •  Email: minhducphan2005@gmail.com

---

## 📝 License

MIT License

---

<div align="center">

**Made with ❤️ for Vietnamese Literature**

⭐ Star this repo if you find it useful!

</div>
