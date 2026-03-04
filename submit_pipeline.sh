#!/bin/bash
# submit_pipeline.sh
#
# Submit the full hidden-knowledge pipeline as dependent SLURM jobs:
#
#   Stage 0 (sanity)  — one job         — quick padding/decoding sanity checks
#   Stage 1 (base)    — one job         — processes the base LLaMA-3-8B model
#   Stage 2 (methods) — job array [0-8] — one task per unlearning method + Llama3-8B reference
#   Stage 2b (llama70b) — one job       — Llama-3-70B reference (gen + logit + MCQ only)
#   Stage 3 (summary) — one job         — aggregates results and prints tables
#
# Usage:
#   bash submit_pipeline.sh                    # submit all stages (no 70B by default)
#   bash submit_pipeline.sh --with-70b         # include Llama-3-70B stage
#   bash submit_pipeline.sh --base-only        # only (re-)submit base (no sanity)
#   bash submit_pipeline.sh --skip-sanity      # skip sanity, submit base→methods→summary
#   bash submit_pipeline.sh --skip-sanity --with-70b
#   bash submit_pipeline.sh --reset            # delete result JSONs before submitting
#                                              # (keeps partial caches; model reloads only
#                                              #  for new MCQ logit scoring)
#
# If a stage is already done (its checkpoint exists) the Python script exits
# immediately without recomputing, so it is safe to re-submit after failure.

set -e

mkdir -p logs

# ── Parse arguments ────────────────────────────────────────────────────────
BASE_ONLY=false
SKIP_SANITY=false
WITH_70B=false
RESET=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --base-only)    BASE_ONLY=true ;;
        --skip-sanity)  SKIP_SANITY=true ;;
        --with-70b)     WITH_70B=true ;;
        --reset)        RESET=true ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Optional reset: remove result JSONs so stages recompute with new code ──
# Partial caches (*_partial.json) are preserved to avoid re-running generation.
if $RESET; then
    echo "Removing result checkpoint JSONs (partial caches preserved)..."
    rm -fv \
        checkpoints/base_results.json \
        checkpoints/GradDiff_results.json \
        checkpoints/RMU_results.json \
        checkpoints/RMU-LAT_results.json \
        checkpoints/RepNoise_results.json \
        checkpoints/ELM_results.json \
        checkpoints/RR_results.json \
        checkpoints/TAR_results.json \
        checkpoints/PB_J_results.json \
        checkpoints/Llama3-8B_results.json \
        checkpoints/llama70b_results.json
    echo "Reset done."
    echo ""
fi

# ── Stage 0: sanity ────────────────────────────────────────────────────────
if $SKIP_SANITY || $BASE_ONLY; then
    SANITY_DEP=""
    echo "Skipping sanity stage."
else
    echo "Submitting sanity stage..."
    SANITY_JOB=$(sbatch --parsable slurm_sanity.sh)
    echo "  Sanity job ID : ${SANITY_JOB}"
    SANITY_DEP="--dependency=afterok:${SANITY_JOB}"
fi

# ── Stage 1: base ──────────────────────────────────────────────────────────
echo "Submitting base stage..."
BASE_JOB=$(sbatch --parsable ${SANITY_DEP} slurm_base.sh)
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
echo "  Method array job ID : ${METHOD_JOB}  (tasks 0–8)"

# ── Stage 2b: Llama-3-70B reference (optional, runs in parallel with methods)
if $WITH_70B; then
    echo "Submitting Llama-3-70B stage (smart, depends on base job ${BASE_JOB})..."
    JOB70B_IDS=$(bash submit_70b_smart.sh --parsable --dependency afterok:${BASE_JOB})
    echo "  Llama-3-70B job IDs: ${JOB70B_IDS}"
    SUMMARY_DEP="--dependency=afterok:${METHOD_JOB}:${JOB70B_IDS}"
else
    SUMMARY_DEP="--dependency=afterok:${METHOD_JOB}"
fi

# ── Stage 3: summary (depends on methods and optionally 70B) ──────────────
echo "Submitting summary stage..."
SUMMARY_JOB=$(sbatch --parsable ${SUMMARY_DEP} slurm_summary.sh)
echo "  Summary job ID : ${SUMMARY_JOB}"

echo ""
echo "Pipeline submitted successfully:"
if ! $SKIP_SANITY && ! $BASE_ONLY; then
    echo "  Stage 0 — sanity   : ${SANITY_JOB}"
fi
echo "  Stage 1 — base     : ${BASE_JOB}"
echo "  Stage 2 — methods  : ${METHOD_JOB} (array 0–8)"
if $WITH_70B; then
    echo "  Stage 2b— llama70b : ${JOB70B_IDS}"
fi
echo "  Stage 3 — summary  : ${SUMMARY_JOB}"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "Logs are written to: logs/"
if ! $SKIP_SANITY && ! $BASE_ONLY; then
    echo "  Sanity   : logs/sanity_${SANITY_JOB}.out"
fi
echo "  Base     : logs/base_${BASE_JOB}.out"
echo "  Methods  : logs/method_<task>_<jobid>.out"
if $WITH_70B; then
    echo "  Llama70B : logs/70b_w1_<jobid>.out  (smart workers)"
fi
echo "  Summary  : logs/summary_${SUMMARY_JOB}.out"
