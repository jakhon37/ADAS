# Production Dockerfile for ADAS Core
# Multi-stage build for optimized image size

# Stage 1: Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies and build wheel
RUN pip install --no-cache-dir build && \
    python -m build --wheel

# Stage 2: Runtime stage
FROM python:3.11-slim

LABEL maintainer="ADAS Core Team"
LABEL description="Advanced Driver Assistance System - Production Container"

# Create non-root user
RUN useradd -m -u 1000 adas && \
    mkdir -p /app /data /logs && \
    chown -R adas:adas /app /data /logs

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheel from builder and install
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Switch to non-root user
USER adas

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ADAS_LOG_LEVEL=INFO
ENV ADAS_CONFIG_PATH=/app/config.json

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import adas; print('healthy')" || exit 1

# Entry point
ENTRYPOINT ["adas-run"]
CMD ["--frames", "0"]
