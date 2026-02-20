#!/bin/bash
# submit_pipeline.sh
#
# Submit the full hidden-knowledge pipeline as three dependent SLURM jobs:
#
#   Stage 1 (base)    — one job         — processes the base LLaMA model
#   Stage 2 (methods) — job array [0-7] — one task per unlearning method
#   Stage 3 (summary) — one job         — aggregates results and prints table
#
# Usage:
#   bash submit_pipeline.sh            # submit all three stages
#   bash submit_pipeline.sh --base-only  # only (re-)submit the base stage
#
# If a stage is already done (its checkpoint exists) the Python script exits
# immediately without recomputing, so it is safe to re-submit after failure.

set -e

mkdir -p logs

# ── Parse arguments ────────────────────────────────────────────────────────
BASE_ONLY=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --base-only) BASE_ONLY=true ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Stage 1: base ──────────────────────────────────────────────────────────
echo "Submitting base stage..."
BASE_JOB=$(sbatch --parsable slurm_base.sh)
echo "  Base job ID : ${BASE_JOB}"

if $BASE_ONLY; then
    echo ""
    echo "Submitted base job only.  Run submit_pipeline.sh again (without"
    echo "--base-only) once it completes to submit methods + summary."
    exit 0
fi

# ── Stage 2: methods (job array, depends on base) ─────────────────────────
echo "Submitting method array (depends on base job ${BASE_JOB})..."
METHOD_JOB=$(sbatch --parsable --dependency=afterok:${BASE_JOB} slurm_methods.sh)
echo "  Method array job ID : ${METHOD_JOB}  (tasks 0–7)"

# ── Stage 3: summary (depends on all method tasks) ────────────────────────
echo "Submitting summary stage (depends on method array ${METHOD_JOB})..."
SUMMARY_JOB=$(sbatch --parsable --dependency=afterok:${METHOD_JOB} slurm_summary.sh)
echo "  Summary job ID : ${SUMMARY_JOB}"

echo ""
echo "Pipeline submitted successfully:"
echo "  Stage 1 — base    : ${BASE_JOB}"
echo "  Stage 2 — methods : ${METHOD_JOB} (array 0–7)"
echo "  Stage 3 — summary : ${SUMMARY_JOB}"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "Logs are written to: logs/"
echo "  Base    : logs/base_${BASE_JOB}.out"
echo "  Methods : logs/method_<task>_<jobid>.out"
echo "  Summary : logs/summary_${SUMMARY_JOB}.out"
