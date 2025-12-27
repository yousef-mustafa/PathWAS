#!/bin/bash
# ============================================
# PathWAS Directory Setup Script
# ============================================
# Creates the required directory structure for PathWAS testing
#
# Usage: ./scripts/setup_directories.sh [--clean]
#
# Options:
#   --clean    Remove existing data directories before creating

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
CLEAN=false
if [[ "$1" == "--clean" ]]; then
    CLEAN=true
fi

echo "============================================"
echo "PathWAS Directory Setup"
echo "============================================"
echo "Project directory: $PROJECT_DIR"
echo ""

# Clean if requested
if [[ "$CLEAN" == true ]]; then
    echo "Cleaning existing directories..."
    rm -rf "$PROJECT_DIR/data/raw"
    rm -rf "$PROJECT_DIR/data/processed"
    rm -rf "$PROJECT_DIR/logs"
fi

# Create directory structure
echo "Creating directory structure..."

# Data directories
mkdir -p "$PROJECT_DIR/data/raw"
mkdir -p "$PROJECT_DIR/data/processed/experiments"

# Output directories
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results"
mkdir -p "$PROJECT_DIR/reports"
mkdir -p "$PROJECT_DIR/models"

# Script directories
mkdir -p "$PROJECT_DIR/scripts/tests"

# Config directory
mkdir -p "$PROJECT_DIR/configs"

echo ""
echo "Directory structure created:"
echo ""
echo "  data/"
echo "  ├── raw/                 # Synthetic input data"
echo "  └── processed/"
echo "      └── experiments/     # Experiment outputs"
echo ""
echo "  logs/                    # Log files"
echo "  results/                 # Analysis results"
echo "  reports/                 # Generated reports"
echo "  models/                  # Saved models"
echo ""
echo "  scripts/"
echo "  └── tests/               # API test scripts"
echo ""
echo "  configs/                 # Configuration files"
echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
