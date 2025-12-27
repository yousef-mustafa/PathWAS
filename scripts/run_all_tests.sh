#!/bin/bash
# ============================================
# PathWAS Master Test Runner
# ============================================
#
# This script runs the complete PathWAS test suite:
# 1. Generates synthetic data
# 2. Runs API tests for all modules
# 3. Runs CLI tests
# 4. Runs experiment configurations
#
# Usage:
#   ./scripts/run_all_tests.sh [options]
#
# Options:
#   --skip-data-gen    Skip data generation (use existing data)
#   --api-only         Only run API tests
#   --cli-only         Only run CLI tests
#   --quick            Quick mode: smaller data, fewer tests
#   --verbose          Show detailed output
#   --help             Show this help message
#
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
SKIP_DATA_GEN=false
API_ONLY=false
CLI_ONLY=false
QUICK_MODE=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data-gen)
            SKIP_DATA_GEN=true
            shift
            ;;
        --api-only)
            API_ONLY=true
            shift
            ;;
        --cli-only)
            CLI_ONLY=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

run_test() {
    local name="$1"
    local cmd="$2"

    if [[ "$VERBOSE" == true ]]; then
        log_info "Running: $name"
        if eval "$cmd"; then
            log_success "$name"
            ((TESTS_PASSED++))
        else
            log_error "$name"
            ((TESTS_FAILED++))
        fi
    else
        log_info "Running: $name"
        if eval "$cmd" > /dev/null 2>&1; then
            log_success "$name"
            ((TESTS_PASSED++))
        else
            log_error "$name"
            ((TESTS_FAILED++))
        fi
    fi
}

# ============================================
# Main Test Execution
# ============================================

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              PathWAS Testing Suite                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

log_info "Project directory: $PROJECT_DIR"
log_info "Python version: $(python --version 2>&1)"
log_info "Date: $(date)"
echo ""

# ============================================
# Step 0: Setup directories
# ============================================

log_section "Step 0: Setup Directories"

mkdir -p data/raw
mkdir -p data/processed/experiments
mkdir -p logs
mkdir -p results

log_success "Directory structure created"

# ============================================
# Step 1: Generate synthetic data
# ============================================

if [[ "$SKIP_DATA_GEN" == false && "$CLI_ONLY" == false ]]; then
    log_section "Step 1: Generating Synthetic Data"

    # Set data generation parameters
    if [[ "$QUICK_MODE" == true ]]; then
        N_SAMPLES=50
        N_GENES=500
        N_VARIANTS=100
        N_PATHWAYS=10
    else
        N_SAMPLES=200
        N_GENES=2000
        N_VARIANTS=500
        N_PATHWAYS=30
    fi

    log_info "Generating expression data (${N_SAMPLES} samples, ${N_GENES} genes)..."
    python scripts/generate_synthetic_expression.py \
        --n-samples "$N_SAMPLES" \
        --n-genes "$N_GENES" \
        --n-de-genes $((N_GENES / 10)) \
        --output-dir data/raw \
        --seed 42

    if [[ $? -eq 0 ]]; then
        log_success "Expression data generated"
    else
        log_error "Expression data generation failed"
        exit 1
    fi

    log_info "Generating genotype data (${N_VARIANTS} variants per chromosome)..."
    python scripts/generate_synthetic_genotypes.py \
        --n-samples "$N_SAMPLES" \
        --n-variants-per-chr "$N_VARIANTS" \
        --chromosomes 1,22 \
        --output-dir data/raw \
        --sample-ids data/raw/sample_metadata.csv \
        --seed 42

    if [[ $? -eq 0 ]]; then
        log_success "Genotype data generated"
    else
        log_error "Genotype data generation failed"
        exit 1
    fi

    log_info "Generating pathway definitions (${N_PATHWAYS} pathways)..."
    python scripts/generate_synthetic_pathways.py \
        --n-pathways "$N_PATHWAYS" \
        --gene-list data/raw/gene_info.csv \
        --output-dir data/raw \
        --seed 42

    if [[ $? -eq 0 ]]; then
        log_success "Pathway definitions generated"
    else
        log_error "Pathway generation failed"
        exit 1
    fi

    echo ""
    log_info "Data generation complete. Files:"
    ls -lh data/raw/*.csv data/raw/*.json data/raw/*.gmt data/raw/*.vcf.gz 2>/dev/null | while read line; do
        echo "  $line"
    done
else
    log_section "Step 1: Skipping Data Generation"
    log_info "Using existing data in data/raw/"
fi

# ============================================
# Step 2: Run API tests
# ============================================

if [[ "$CLI_ONLY" == false ]]; then
    log_section "Step 2: Running API Tests"

    API_TESTS=(
        "test_expression_api.py:Expression Module"
        "test_pas_api.py:PAS Module"
        "test_statistics_api.py:Statistics Module"
        "test_modeling_api.py:Modeling Module"
        "test_gene_sets_api.py:Gene Sets Module"
    )

    for test_spec in "${API_TESTS[@]}"; do
        test_file="${test_spec%%:*}"
        test_name="${test_spec##*:}"

        if [[ -f "scripts/tests/${test_file}" ]]; then
            run_test "$test_name" "python scripts/tests/${test_file}"
        else
            log_warn "Test file not found: ${test_file}"
            ((TESTS_SKIPPED++))
        fi
    done
fi

# ============================================
# Step 3: Run CLI tests
# ============================================

if [[ "$API_ONLY" == false ]]; then
    log_section "Step 3: Running CLI Tests"

    if [[ -f "scripts/tests/test_cli_integration.py" ]]; then
        run_test "CLI Integration" "python scripts/tests/test_cli_integration.py"
    else
        log_warn "CLI test file not found"
        ((TESTS_SKIPPED++))
    fi
fi

# ============================================
# Step 4: Run experiment configurations
# ============================================

if [[ "$API_ONLY" == false && "$QUICK_MODE" == false ]]; then
    log_section "Step 4: Running Experiment Configurations"

    # Test basic PAS config
    if [[ -f "configs/test_pas_basic.yaml" ]]; then
        log_info "Testing basic PAS configuration..."
        run_test "Basic PAS Config" "python -c \"
from pathwas.experiment import run_experiment, load_config
config = load_config('configs/test_pas_basic.yaml')
results = run_experiment(config)
print(f'Experiment completed: {results.get(\\\"experiment_dir\\\", \\\"unknown\\\")}')
\""
    fi

    # Test differential analysis config
    if [[ -f "configs/test_pas_differential.yaml" ]]; then
        log_info "Testing differential analysis configuration..."
        run_test "Differential Config" "python -c \"
from pathwas.experiment import run_experiment, load_config
config = load_config('configs/test_pas_differential.yaml')
results = run_experiment(config)
print(f'Experiment completed: {results.get(\\\"experiment_dir\\\", \\\"unknown\\\")}')
\""
    fi
fi

# ============================================
# Summary
# ============================================

log_section "Test Summary"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))

echo -e "Total tests run: ${TOTAL_TESTS}"
echo -e "${GREEN}Passed: ${TESTS_PASSED}${NC}"
echo -e "${RED}Failed: ${TESTS_FAILED}${NC}"
echo -e "${YELLOW}Skipped: ${TESTS_SKIPPED}${NC}"
echo ""

if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              All tests passed successfully!                ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║              Some tests failed!                            ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
