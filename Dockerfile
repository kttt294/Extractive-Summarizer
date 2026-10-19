# Stage 1: Build React Frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up user and permissions
RUN useradd -m -u 1000 user && \
    mkdir -p /home/user/app/models /home/user/app/outputs && \
    chown -R user:user /home/user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/app/.cache/huggingface \
    NLTK_DATA=/opt/nltk_data \
    MODEL_DEVICE=cpu \
    PRELOAD_MODELS=true \
    TOKENIZERS_PARALLELISM=false

WORKDIR $HOME/app

# Install python dependencies
COPY --chown=user requirements-web.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-web.txt && \
    python -m nltk.downloader -d /opt/nltk_data punkt punkt_tab

# Copy source code and backend
COPY --chown=user src/ ./src/
COPY --chown=user backend/ ./backend/

# Copy built frontend dist from Stage 1
COPY --chown=user --from=frontend-builder /frontend/dist ./frontend/dist

RUN mkdir -p /home/user/app/.cache/huggingface && chown -R user:user /home/user/app

USER user

ENV PORT=10000
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import os,json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','10000')+'/api/v1/health')); assert d['vietnamese_model_ready']" || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
