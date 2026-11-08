# ==============================================================================
# VisionGuard Production Dockerfile (Task 6.2)
# Student: Bissam Iftikhar (23L-0913) | Project: 23L-0913_A2_Ops
# ==============================================================================

# 1. Base Image: Official Debian-based Python 3.12 slim image (minimizes CVE attack surface & image size)
FROM python:3.12-slim as base

# Set environment variables for non-interactive installs and unbuffered Python logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5001 \
    DEBIAN_FRONTEND=noninteractive

# 2. Install essential runtime dependencies and curl (for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Create non-root system user for security compliance (Principle of Least Privilege)
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# 4. LAYER CACHE OPTIMIZATION: Copy requirements first and install dependencies
# Docker only rebuilds this expensive layer when requirements.txt changes.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy application source code, models, static assets, and templates
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py .

# Create models directory and assign ownership to non-root user
RUN mkdir -p /app/models /app/data /app/instance && \
    chown -R appuser:appgroup /app

# 6. Switch to non-root user
USER appuser

# 7. Expose Flask web service port
EXPOSE 5001

# 8. Container Healthcheck (Task 6.2 & 6.3)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# 9. Default container entrypoint using production WSGI server (Gunicorn)
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--timeout", "120", "app:app"]

