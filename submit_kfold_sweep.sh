#!/bin/bash
# submit_kfold_sweep.sh
#
# Submit 5-fold cross-validation probe jobs for ALL sweep checkpoints.
# Each (method, checkpoint, fold) triple is one SLURM array task.
#
# Task layout in slurm_kfold_sweep.sh (320 tasks):
#   method_idx = task_id // 40      (0-7  → GradDiff … PB&J)
#   ck         = (task_id % 40) // 5 + 1  (1-8)
#   fold       = (task_id % 40) % 5        (0-4)
#
# Usage:
#   bash submit_kfold_sweep.sh                        # full 320-task array + summary
#   bash submit_kfold_sweep.sh --train-only           # training array only
#   bash submit_kfold_sweep.sh --summary-only         # summary + tables only
#   bash submit_kfold_sweep.sh --method GradDiff      # one method (40 tasks) + summary
#   bash submit_kfold_sweep.sh --method ELM --checkpoint 3          # single ck (5 tasks)
#   bash submit_kfold_sweep.sh --method RMU --checkpoint 5 --fold 2 # single task
#
# Requires: sweep hidden-state .npy files must already exist in
#   checkpoints/sweep_{method}/ck{N}/hs_{train,val,test}.npy
#   (run submit_sweep.sh first if they don't).

set -euo pipefail

TRAIN_ONLY=0
SUMMARY_ONLY=0
SINGLE_METHOD=""
SINGLE_CK=""
SINGLE_FOLD=""

METHODS=("GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB&J")

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train-only)   TRAIN_ONLY=1 ;;
        --summary-only) SUMMARY_ONLY=1 ;;
        --method)       SINGLE_METHOD="$2"; shift ;;
        --checkpoint)   SINGLE_CK="$2";    shift ;;
        --fold)         SINGLE_FOLD="$2";  shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p logs checkpoints/kfold_sweep data

# ── Resolve method index ──────────────────────────────────────────────────────
METHOD_IDX=-1
if [[ -n "$SINGLE_METHOD" ]]; then
    for i in "${!METHODS[@]}"; do
        if [[ "${METHODS[$i]}" == "$SINGLE_METHOD" ]]; then
            METHOD_IDX=$i
        fi
    done
    if [[ $METHOD_IDX -lt 0 ]]; then
        echo "ERROR: Unknown method '${SINGLE_METHOD}'."
        echo "       Choose from: ${METHODS[*]}"
        exit 1
    fi
fi

# ── Determine array range ─────────────────────────────────────────────────────
if [[ -n "$SINGLE_METHOD" && -n "$SINGLE_CK" && -n "$SINGLE_FOLD" ]]; then
    # Single (method, ck, fold) task
    TASK_ID=$(( METHOD_IDX * 40 + (SINGLE_CK - 1) * 5 + SINGLE_FOLD ))
    ARRAY_RANGE="${TASK_ID}"
    DESC="${SINGLE_METHOD} ck${SINGLE_CK} fold${SINGLE_FOLD} (task ${TASK_ID})"

elif [[ -n "$SINGLE_METHOD" && -n "$SINGLE_CK" ]]; then
    # One method × one checkpoint × all 5 folds
    START=$(( METHOD_IDX * 40 + (SINGLE_CK - 1) * 5 ))
    END=$(( START + 4 ))
    ARRAY_RANGE="${START}-${END}"
    DESC="${SINGLE_METHOD} ck${SINGLE_CK} folds 0-4 (tasks ${START}-${END})"

elif [[ -n "$SINGLE_METHOD" ]]; then
    # One method × all 8 checkpoints × all 5 folds = 40 tasks
    START=$(( METHOD_IDX * 40 ))
    END=$(( START + 39 ))
    ARRAY_RANGE="${START}-${END}"
    DESC="${SINGLE_METHOD} ck1-ck8 folds 0-4 (tasks ${START}-${END})"

else
    # Full 320-task array
    ARRAY_RANGE="0-319"
    DESC="all 8 methods × 8 checkpoints × 5 folds (tasks 0-319)"
fi

# ── Summary-only: skip training ───────────────────────────────────────────────
if [[ "$SUMMARY_ONLY" -eq 1 ]]; then
    echo "Submitting summary+tables only (no training)..."
    SUM_JOB=$(sbatch slurm_kfold_sweep_summary.sh | awk '{print $NF}')
    echo "  Summary job: ${SUM_JOB}"
    echo ""
    echo "Monitor:  squeue -u \$USER"
    echo "Logs:     logs/kfold_sweep_summary_${SUM_JOB}.out"
    exit 0
fi

# ── Submit training array ─────────────────────────────────────────────────────
echo "Submitting kfold_sweep training: ${DESC}"
TRAIN_JOB=$(sbatch --parsable --array="${ARRAY_RANGE}" slurm_kfold_sweep.sh)
echo "  Training array job: ${TRAIN_JOB}"

if [[ "$TRAIN_ONLY" -eq 1 ]]; then
    echo "  (--train-only: skipping summary submission)"
    echo ""
    echo "When done, run:  bash submit_kfold_sweep.sh --summary-only"
    exit 0
fi

# ── Submit summary with dependency ───────────────────────────────────────────
echo "Submitting summary+tables job (depends on training array)..."
SUM_JOB=$(sbatch --parsable \
    --dependency=afterok:${TRAIN_JOB} \
    slurm_kfold_sweep_summary.sh)
echo "  Summary job: ${SUM_JOB}  (depends on ${TRAIN_JOB})"

echo ""
echo "Pipeline submitted:"
echo "  Training array : ${TRAIN_JOB}  (${DESC})"
echo "  Summary+tables : ${SUM_JOB}    (after ${TRAIN_JOB})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/kfold_sweep_0_*.out"
echo ""
echo "When complete, results are in:"
echo "  checkpoints/kfold_sweep/sweep_{method}/ck{N}/f{fold}_results.json"
echo "  data/kfold_sweep_results_aggregated.csv"
echo "  data/kfold_sweep_per_layer_aggregated.csv"
echo "  data/kfold_sweep_table_probes.csv"
echo "  data/kfold_sweep_per_layer_table.csv"
