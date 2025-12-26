# PathWAS Dockerfile
# Build: docker build -t pathwas:latest .
# Run:   docker run --rm -v $(pwd):/project pathwas:latest --config /project/configs/myconfig.yaml

# Use Python 3.11 slim image as base
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    zlib1g-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy the package source
COPY setup.py .
COPY README.md .
COPY LICENSE .
COPY pathwas/ pathwas/

# Install PathWAS
RUN pip install .


# Production image
FROM python:3.11-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash pathwas

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/pathwas /usr/local/bin/pathwas

# Set working directory
WORKDIR /project

# Change ownership of project directory
RUN chown -R pathwas:pathwas /project

# Switch to non-root user
USER pathwas

# Set the entrypoint to the pathwas CLI
ENTRYPOINT ["pathwas"]

# Default command shows help
CMD ["--help"]

# Labels for container metadata
LABEL org.opencontainers.image.title="PathWAS" \
      org.opencontainers.image.description="Pathway-Wide Association Studies Analysis Framework" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.source="https://github.com/yousef-mustafa/PathWAS" \
      org.opencontainers.image.licenses="MIT"
