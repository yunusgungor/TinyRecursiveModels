# syntax=docker/dockerfile:1.4
# Multi-stage build for Python backend with BuildKit optimizations

# Base stage - minimal dependencies, rarely changes (optimal for cache)
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies (layer ordering: system deps before app deps)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    # Playwright dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies stage - cached layer for dependency files
FROM base as dependencies

WORKDIR /app

# Copy only dependency files first (cache optimization - changes less frequently than code)
COPY requirements.txt requirements-dev.txt ./

# Development stage - includes dev dependencies
FROM dependencies as development

# Use cache mount for pip to speed up builds (BuildKit cache optimization)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt && \
    playwright install chromium

# Copy application code last (changes most frequently)
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Builder stage - compile runtime dependencies only
FROM dependencies as builder

# Use cache mount for pip (BuildKit cache optimization)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --target=/app/packages

# Production stage - minimal final image (optimized for size)
FROM python:3.10-slim as production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/packages/bin:$PATH" \
    PYTHONPATH="/app/packages"

WORKDIR /app

# Copy only compiled packages from builder (runtime dependencies only)
COPY --from=builder /app/packages /app/packages

# Copy application code (last layer - changes most frequently)
COPY . .

# Clean up unnecessary files to reduce image size
RUN find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete && \
    find /app -type f -name "*.pyo" -delete && \
    rm -rf /app/tests /app/.pytest_cache /app/.hypothesis

# Create non-root user for security (UID 1000)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    # Set minimal file permissions (least privilege principle)
    # Remove write permissions for group and others
    chmod -R u=rwX,g=rX,o= /app && \
    # Ensure only necessary files are executable
    find /app -type f -name "*.py" -exec chmod u=rw,g=r,o= {} \; && \
    find /app -type f -name "*.sh" -exec chmod u=rwx,g=rx,o= {} \; && \
    # Packages should be readable but not writable
    chmod -R u=rX,g=rX,o= /app/packages

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
