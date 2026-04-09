# Backend Dockerfile — AutoVision FastAPI Service
# Multi-layer optimization for builds and runtime security

FROM python:3.11-slim

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies (libgl1, libglib2.0 needed for image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mlops && chown -R mlops:mlops /app

# Copy requirements first for Docker layer caching
COPY --chown=mlops:mlops requirements.txt .

# ── Step 1: Install CPU-only PyTorch BEFORE requirements.txt ──────────────────
# ultralytics (in requirements.txt) declares torch as a dependency. If torch is
# already installed when pip processes ultralytics, pip will not try to re-install
# it, so we avoid pulling the 2+ GB CUDA wheel entirely.
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir \
        torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Install remaining dependencies (torch already satisfied) ──────────
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=mlops:mlops . .

# Create storage directories with proper permissions
RUN mkdir -p /tmp/autovision/uploads /tmp/autovision/models /tmp/autovision/logs && \
    chown -R mlops:mlops /tmp/autovision

# Switch to non-root user
USER mlops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/system/info || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
