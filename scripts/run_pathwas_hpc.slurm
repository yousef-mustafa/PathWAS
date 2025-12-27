#!/bin/bash
#SBATCH --job-name=pathwas
#SBATCH --output=logs/pathwas_%A_%a.out
#SBATCH --error=logs/pathwas_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-5%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@institution.edu

# ============================================
# PathWAS HPC Runner via Singularity
# ============================================
#
# This script runs PathWAS experiments on an HPC cluster using
# Singularity containers. It supports array jobs for parallel
# execution of multiple configurations.
#
# Usage:
#   sbatch scripts/run_pathwas_hpc.slurm
#
# Environment Variables:
#   SINGULARITY_IMAGE - Path to PathWAS Singularity image
#   WORK_DIR          - Working directory (default: current directory)
#   CONFIG_FILE       - Specific config file (overrides array job)
#
# ============================================

set -euo pipefail

# ============================================
# Configuration
# ============================================

# Load required modules (adjust for your HPC system)
module load singularity/3.8.0 2>/dev/null || true

# Set up paths
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-/path/to/pathwas.sif}"
WORK_DIR="${WORK_DIR:-$(pwd)}"
CONFIG_DIR="${WORK_DIR}/configs"
DATA_DIR="${WORK_DIR}/data"
RESULTS_DIR="${WORK_DIR}/results"
LOGS_DIR="${WORK_DIR}/logs"

# Config files array (for array jobs)
CONFIG_FILES=(
    "test_pas_basic.yaml"
    "test_pas_differential.yaml"
    "test_full_pipeline.yaml"
    "test_bayes_mixture.yaml"
    "test_statistical_methods.yaml"
)

# ============================================
# Job Setup
# ============================================

# Create required directories
mkdir -p "${LOGS_DIR}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${DATA_DIR}/processed/experiments"

# Determine which config to use
if [[ -n "${CONFIG_FILE:-}" ]]; then
    # Use environment variable if set
    CONFIG_NAME="${CONFIG_FILE}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Use array task ID
    CONFIG_NAME="${CONFIG_FILES[$SLURM_ARRAY_TASK_ID-1]}"
else
    # Default to first config
    CONFIG_NAME="${CONFIG_FILES[0]}"
fi

CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}"

# ============================================
# Logging
# ============================================

echo "============================================"
echo "PathWAS HPC Job"
echo "============================================"
echo ""
echo "Job Information:"
echo "  Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "  Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "  Node:          ${SLURMD_NODENAME:-$(hostname)}"
echo "  CPUs:          ${SLURM_CPUS_PER_TASK:-1}"
echo "  Memory:        ${SLURM_MEM_PER_NODE:-N/A}"
echo ""
echo "Paths:"
echo "  Work dir:      ${WORK_DIR}"
echo "  Config:        ${CONFIG_NAME}"
echo "  Singularity:   ${SINGULARITY_IMAGE}"
echo ""
echo "Started: $(date)"
echo "============================================"
echo ""

# ============================================
# Validation
# ============================================

# Check if config file exists
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG_PATH}"
    echo ""
    echo "Available configs:"
    ls -1 "${CONFIG_DIR}"/*.yaml 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Check if Singularity image exists
if [[ ! -f "${SINGULARITY_IMAGE}" ]]; then
    echo "WARNING: Singularity image not found: ${SINGULARITY_IMAGE}"
    echo "Attempting to run with local Python installation..."
    USE_SINGULARITY=false
else
    USE_SINGULARITY=true
fi

# ============================================
# Execution
# ============================================

echo "Running PathWAS with config: ${CONFIG_NAME}"
echo ""

if [[ "${USE_SINGULARITY}" == true ]]; then
    # Run via Singularity
    singularity exec \
        --cleanenv \
        --bind "${WORK_DIR}:/workspace" \
        --pwd /workspace \
        "${SINGULARITY_IMAGE}" \
        python -m pathwas.experiment \
            --config "/workspace/configs/${CONFIG_NAME}"
else
    # Run with local Python
    cd "${WORK_DIR}"
    python -m pathwas.experiment \
        --config "${CONFIG_PATH}"
fi

EXIT_CODE=$?

# ============================================
# Completion
# ============================================

echo ""
echo "============================================"
echo "Job Complete"
echo "============================================"
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo ""

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "SUCCESS: Experiment completed successfully"

    # Find and report output directory
    EXPERIMENT_DIR=$(grep -r "experiment_dir" "${LOGS_DIR}/pathwas_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-1}.out" 2>/dev/null | tail -1 || true)
    if [[ -n "${EXPERIMENT_DIR}" ]]; then
        echo "Results saved to: ${EXPERIMENT_DIR}"
    fi
else
    echo "FAILED: Experiment failed with exit code ${EXIT_CODE}"
    echo ""
    echo "Check error log for details:"
    echo "  ${LOGS_DIR}/pathwas_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-1}.err"
fi

echo "============================================"

exit ${EXIT_CODE}
