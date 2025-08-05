# Multi-stage Dockerfile for AV-Separation-Transformer
# Optimized for production deployment with security and performance

# Build stage
FROM python:3.10-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_DEPS="build-essential cmake git"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ${BUILD_DEPS} \
    pkg-config \
    libsndfile1-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and install the package
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.10-slim as production

# Set labels for metadata
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>"
LABEL version="1.0.0"
LABEL description="AV-Separation-Transformer for real-time audio-visual speech separation"
LABEL org.opencontainers.image.source="https://github.com/danieleschmidt/quantum-inspired-task-planner"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Runtime arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG RUNTIME_DEPS="ffmpeg libsndfile1 curl"

# Create non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ${RUNTIME_DEPS} \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser . /app

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp /app/models \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.av_separation.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Development stage (optional)
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    jupyter \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Switch back to app user
USER appuser

# Override command for development
CMD ["uvicorn", "src.av_separation.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1"]