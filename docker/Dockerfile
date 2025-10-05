# Multi-stage build for OAI Inpainting
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app
COPY --chown=app:app . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev,ml]"

# Switch to non-root user
USER app

# Default command
CMD ["python", "-m", "scripts.train", "--help"]

# Development stage
FROM base as development
USER root
RUN pip install -e ".[dev,docs]"
USER app
CMD ["bash"]

# Production stage
FROM base as production
# Remove development dependencies
RUN pip uninstall -y pytest pytest-cov pytest-mock mypy ruff pre-commit
CMD ["python", "-m", "scripts.train"]
