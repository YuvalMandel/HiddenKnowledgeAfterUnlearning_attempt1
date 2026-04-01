#!/bin/bash
# submit_kfold.sh — Submit 5-fold cross-validation probe jobs.
#
# Usage:
#   bash submit_kfold.sh              # submit training array + summary (with dependency)
#   bash submit_kfold.sh --train-only # submit only training array
#   bash submit_kfold.sh --summary-only # submit only summary (assumes training done)
#
# The training array runs 45 parallel jobs (5 folds × 9 models), each needing
# no GPU. The summary job depends on all training jobs completing successfully.

set -euo pipefail

TRAIN_ONLY=0
SUMMARY_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --train-only)    TRAIN_ONLY=1 ;;
        --summary-only)  SUMMARY_ONLY=1 ;;
    esac
done

mkdir -p logs checkpoints/kfold data

if [[ "$SUMMARY_ONLY" -eq 1 ]]; then
    echo "Submitting summary job only..."
    SUMMARY_JOB=$(sbatch slurm_kfold_summary.sh | awk '{print $NF}')
    echo "  Summary job: ${SUMMARY_JOB}"
    exit 0
fi

echo "Submitting kfold training array (45 tasks: 5 folds × 9 models)..."
TRAIN_JOB=$(sbatch slurm_kfold.sh | awk '{print $NF}')
echo "  Training array job: ${TRAIN_JOB}"

if [[ "$TRAIN_ONLY" -eq 1 ]]; then
    echo "  (--train-only: skipping summary submission)"
    exit 0
fi

echo "Submitting summary job (depends on training array)..."
SUMMARY_JOB=$(sbatch --dependency=afterok:${TRAIN_JOB} slurm_kfold_summary.sh | awk '{print $NF}')
echo "  Summary job: ${SUMMARY_JOB}"

echo ""
echo "Pipeline submitted:"
echo "  Training array : ${TRAIN_JOB}  (45 tasks)"
echo "  Summary        : ${SUMMARY_JOB}  (after ${TRAIN_JOB})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/kfold_0_*.out"
