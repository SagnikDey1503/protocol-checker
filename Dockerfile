# ============================================================
# AI Research Protocol Assistant — Dockerfile
# Multi-stage build for production deployment
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim as builder

WORKDIR /build

# Install system deps for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

# --- Stage 2: Runtime ---
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Render injects PORT; default to 8000 for local Docker Compose.
ENV PORT=8000
EXPOSE 8000

# Health check (uses the same port the app listens on)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os, httpx; port=os.environ.get('PORT', '8000'); httpx.get(f'http://localhost:{port}/api/v1/health')" || exit 1

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
