# =============================================================================== #
# PathWAS Dockerfile                                                              #
# =============================================================================== #
# Provides a containerized environment for running PathWAS analyses with all     #
# dependencies pre-installed, including support for LD reference panel setup.    #
# =============================================================================== #

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATHWAS_LD_ROOT=/data/ld_reference

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 pathwas \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash pathwas

# Create directories
RUN mkdir -p /app /data/ld_reference \
    && chown -R pathwas:pathwas /app /data

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY --chown=pathwas:pathwas requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code
COPY --chown=pathwas:pathwas . .

# Install pathwas package
RUN pip install -e .

# Switch to non-root user
USER pathwas

# Default command
ENTRYPOINT ["pathwas"]
CMD ["--help"]

# =============================================================================== #
# Usage Examples:
# ---------------
# Build the image:
#   docker build -t pathwas .
#
# Run LD reference setup:
#   docker run -v $(pwd)/ld_reference:/data/ld_reference pathwas setup-ld \
#       --ancestry EUR --chromosomes 22
#
# Run PAS computation:
#   docker run -v $(pwd)/data:/data pathwas pas /data/expression.csv \
#       --msigdb KEGG_2021_Human --out /data/pas.csv
#
# Interactive shell:
#   docker run -it --entrypoint /bin/bash pathwas
# =============================================================================== #
