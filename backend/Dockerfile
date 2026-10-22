FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/opt/nltk_data MODEL_DEVICE=cpu PRELOAD_MODELS=true \
    TOKENIZERS_PARALLELISM=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements/web.txt ./requirements/web.txt
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements/web.txt \
    && python -m nltk.downloader -d /opt/nltk_data punkt punkt_tab

# Copy source code and backend code
COPY src/ ./src/
COPY backend/ ./backend/
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/.cache/huggingface /app/models /app/outputs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,json; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')); assert d['vietnamese_model_ready']" || exit 1
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
