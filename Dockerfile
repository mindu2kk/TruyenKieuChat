# Legacy local Streamlit image. Production deploys Django through Vercel;
# see docs/VERCEL.md for that path.
# ---- Base image
FROM python:3.12-slim

# ---- System deps (nhỏ gọn, đủ để build wheels cơ bản)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Workdir
WORKDIR /app

# ---- Copy code
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# copy source
COPY app /app/app
COPY data /app/data

# ---- Cấu hình cache model (đặt theo ENV, có thể override khi run)
ENV HF_HOME=/cache/hf_home \
    TRANSFORMERS_CACHE=/cache/transformers \
    RAG_CACHE_DIR=/cache/rag_cache

# tạo sẵn thư mục cache
RUN mkdir -p /cache/hf_home /cache/transformers /cache/rag_cache

# Secrets are injected at runtime (for example, by Docker Compose or the
# deployment platform), never copied into the image.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app /cache
USER appuser

# ---- Expose và run Streamlit
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://127.0.0.1:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/ui_chat.py", "--server.port=8501", "--server.address=0.0.0.0"]
