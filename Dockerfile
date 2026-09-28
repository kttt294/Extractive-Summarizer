# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend
FROM python:3.10-slim

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
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and backend
COPY --chown=user src/ ./src/
COPY --chown=user backend/ ./backend/

# Copy built frontend dist from Stage 1
COPY --chown=user --from=frontend-builder /frontend/dist ./frontend/dist

RUN chown -R user:user /home/user/app

USER user

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
