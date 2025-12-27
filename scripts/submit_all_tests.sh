#!/bin/bash
# ============================================
# Submit All PathWAS Tests to SLURM
# ============================================
#
# This script submits all test configurations to the HPC cluster
# as a SLURM array job.
#
# Usage:
#   ./scripts/submit_all_tests.sh [options]
#
# Options:
#   --dry-run    Show what would be submitted without actually submitting
#   --single N   Submit only config N (1-5)
#   --serial     Submit jobs one at a time (not as array)
#
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse arguments
DRY_RUN=false
SINGLE_JOB=""
SERIAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --single)
            SINGLE_JOB="$2"
            shift 2
            ;;
        --serial)
            SERIAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================
# Setup
# ============================================

echo "============================================"
echo "PathWAS HPC Job Submission"
echo "============================================"
echo ""

# Create logs directory
mkdir -p logs

# Check for required files
if [[ ! -f scripts/run_pathwas_hpc.slurm ]]; then
    echo "ERROR: SLURM script not found: scripts/run_pathwas_hpc.slurm"
    exit 1
fi

# List available configs
echo "Available configurations:"
CONFIGS=(
    "test_pas_basic.yaml"
    "test_pas_differential.yaml"
    "test_full_pipeline.yaml"
    "test_bayes_mixture.yaml"
    "test_statistical_methods.yaml"
)

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    if [[ -f "configs/${config}" ]]; then
        echo "  $((i+1)). ${config} [OK]"
    else
        echo "  $((i+1)). ${config} [MISSING]"
    fi
done
echo ""

# ============================================
# Submit Jobs
# ============================================

if [[ -n "$SINGLE_JOB" ]]; then
    # Submit single job
    config_idx=$((SINGLE_JOB - 1))
    if [[ $config_idx -lt 0 || $config_idx -ge ${#CONFIGS[@]} ]]; then
        echo "ERROR: Invalid job number: $SINGLE_JOB (must be 1-${#CONFIGS[@]})"
        exit 1
    fi

    config="${CONFIGS[$config_idx]}"
    echo "Submitting single job: ${config}"

    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN: Would submit: CONFIG_FILE=${config} sbatch scripts/run_pathwas_hpc.slurm"
    else
        CONFIG_FILE="${config}" sbatch --array=1 scripts/run_pathwas_hpc.slurm
    fi

elif [[ "$SERIAL" == true ]]; then
    # Submit jobs one at a time
    echo "Submitting jobs serially..."

    for i in "${!CONFIGS[@]}"; do
        config="${CONFIGS[$i]}"
        if [[ ! -f "configs/${config}" ]]; then
            echo "Skipping missing config: ${config}"
            continue
        fi

        if [[ "$DRY_RUN" == true ]]; then
            echo "DRY RUN: Would submit: CONFIG_FILE=${config} sbatch scripts/run_pathwas_hpc.slurm"
        else
            echo "Submitting: ${config}"
            CONFIG_FILE="${config}" sbatch --array=1 scripts/run_pathwas_hpc.slurm
        fi
    done

else
    # Submit array job
    echo "Submitting array job for all configurations..."

    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN: Would submit: sbatch scripts/run_pathwas_hpc.slurm"
    else
        sbatch scripts/run_pathwas_hpc.slurm
    fi
fi

echo ""
echo "============================================"

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN complete (no jobs submitted)"
else
    echo "Jobs submitted successfully!"
    echo ""
    echo "Monitor with: squeue -u $USER"
    echo "View logs:    tail -f logs/pathwas_*.out"
fi

echo "============================================"
