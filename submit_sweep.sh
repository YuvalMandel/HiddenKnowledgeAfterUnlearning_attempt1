#!/bin/bash
# submit_sweep.sh
#
# Submit checkpoint-sweep jobs as a 64-task SLURM array.
# Each task handles one (method, checkpoint) pair independently so all 64
# can run in parallel.
#
# Task layout (slurm_sweep.sh):
#   task_id = method_idx * 8 + (checkpoint_num - 1)
#   0- 7 : GradDiff  ck1-ck8
#   8-15 : RMU       ck1-ck8
#  16-23 : RMU-LAT   ck1-ck8
#  24-31 : RepNoise  ck1-ck8
#  32-39 : ELM       ck1-ck8
#  40-47 : RR        ck1-ck8
#  48-55 : TAR       ck1-ck8
#  56-63 : PB&J      ck1-ck8
#
# Requires: --stage base must have completed first.
#
# Usage:
#   bash submit_sweep.sh                     # all 8 methods × 8 checkpoints
#   bash submit_sweep.sh --summary           # also submit summary jobs after
#   bash submit_sweep.sh --method GradDiff   # one method only (8 ck jobs)
#   bash submit_sweep.sh --method GradDiff --checkpoint 3  # single job

set -e
mkdir -p logs

WITH_SUMMARY=false
SINGLE_METHOD=""
SINGLE_CK=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --summary)    WITH_SUMMARY=true ;;
        --method)     SINGLE_METHOD="$2"; shift ;;
        --checkpoint) SINGLE_CK="$2";    shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

SWEEP_METHODS=("GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB&J")

# ── Determine array range ──────────────────────────────────────────────────────
if [[ -n "$SINGLE_METHOD" ]]; then
    METHOD_IDX=-1
    for i in "${!SWEEP_METHODS[@]}"; do
        if [[ "${SWEEP_METHODS[$i]}" == "$SINGLE_METHOD" ]]; then
            METHOD_IDX=$i
        fi
    done
    if [[ $METHOD_IDX -lt 0 ]]; then
        echo "ERROR: Unknown method '${SINGLE_METHOD}'."
        echo "       Choose from: ${SWEEP_METHODS[*]}"
        exit 1
    fi

    if [[ -n "$SINGLE_CK" ]]; then
        # Single (method, checkpoint) job
        TASK_ID=$(( METHOD_IDX * 8 + SINGLE_CK - 1 ))
        ARRAY_RANGE="${TASK_ID}"
        DESC="${SINGLE_METHOD} ck${SINGLE_CK} (task ${TASK_ID})"
    else
        # All checkpoints for one method
        START=$(( METHOD_IDX * 8 ))
        END=$(( START + 7 ))
        ARRAY_RANGE="${START}-${END}"
        DESC="${SINGLE_METHOD} ck1-ck8 (tasks ${START}-${END})"
    fi
else
    ARRAY_RANGE="0-63"
    DESC="all 8 methods × 8 checkpoints (tasks 0-63)"
fi

# ── Submit sweep jobs ──────────────────────────────────────────────────────────
echo "Submitting sweep: ${DESC}"
SWEEP_JOB=$(sbatch --parsable --array="${ARRAY_RANGE}" slurm_sweep.sh)
echo "  Sweep job ID: ${SWEEP_JOB}"

# ── Optionally submit summary jobs after sweep completes ──────────────────────
if $WITH_SUMMARY; then
    if [[ -n "$SINGLE_METHOD" ]]; then
        SUM_RANGE="${METHOD_IDX}"
    else
        SUM_RANGE="0-7"
    fi
    SUM_JOB=$(sbatch --parsable \
        --dependency=afterok:${SWEEP_JOB} \
        --array="${SUM_RANGE}" \
        slurm_sweep_summary.sh)
    echo "  Summary job ID: ${SUM_JOB}  (depends on ${SWEEP_JOB})"
fi

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     logs/sweep_<task>_<jobid>.out"
echo ""
echo "When complete, results are in:"
echo "  checkpoints/sweep_<method>/ck<N>/results.json"
echo "  checkpoints/sweep_<method>/<method>_sweep.csv  (after sweep_summary)"
