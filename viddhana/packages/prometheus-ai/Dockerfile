# Prometheus AI Engine Docker Image
#
# Multi-stage build for production deployment

# ============================================================================
# Stage 1: Build Stage
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Install package
RUN pip install --no-cache-dir -e .


# ============================================================================
# Stage 2: Runtime Stage
# ============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 prometheus

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app/src ./src

# Copy additional files
COPY pyproject.toml ./

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/checkpoints && \
    chown -R prometheus:prometheus /app

# Switch to non-root user
USER prometheus

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/models
ENV PROMETHEUS_ENV=production

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "prometheus.inference.server"]


# ============================================================================
# Stage 3: Development Stage (Optional)
# ============================================================================
FROM runtime AS development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    httpx \
    mypy \
    ruff

# Copy test files
COPY tests/ ./tests/

USER prometheus

# Development command
CMD ["pytest", "tests/", "-v", "--cov=prometheus"]


# ============================================================================
# Stage 4: GPU Runtime (Optional)
# ============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu-runtime

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 prometheus

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY --from=builder /app/src ./src
COPY pyproject.toml ./

# Create directories
RUN mkdir -p /app/models /app/data /app/checkpoints && \
    chown -R prometheus:prometheus /app

USER prometheus

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models
ENV PROMETHEUS_ENV=production
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "prometheus.inference.server"]
