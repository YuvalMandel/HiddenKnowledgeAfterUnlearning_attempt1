#!/bin/bash
# slurm_70b_auto.sh — per-worker SLURM script for distributed 70B inference.
#
# Do NOT submit this script directly.  Use submit_70b_smart.sh instead, which
# passes the right GPU config and sets SMART_RESUBMIT=1 so this script will
# automatically resubmit if work remains after the job ends or is preempted.
#
# Required env vars (set by submit_70b_smart.sh via --export):
#   SMART_RESUBMIT   set to 1 to enable auto-resubmit on completion
#   SUBMIT_DIR       project root directory

#SBATCH --signal=USR1@120   # send SIGUSR1 120s before wall-time limit / preemption

echo "=========================================================="
echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=========================================================="

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unlearning

export HF_HOME="$HOME/.cache/huggingface"
export HF_TOKEN
HF_TOKEN=$(cat "$HF_HOME/token" 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to the project directory.
# SUBMIT_DIR is set by submit_70b_smart.sh; fall back to SLURM_SUBMIT_DIR.
PROJECT_DIR="${SUBMIT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$0")}}"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints/llama70b_dist/results

# ---------------------------------------------------------------------------
# Run the worker
# ---------------------------------------------------------------------------
echo ""
echo "Starting worker (SLURM_JOB_ID=$SLURM_JOB_ID)..."
python run_70b_distributed.py --worker
WORKER_EXIT=$?

echo ""
echo "Worker exited with code $WORKER_EXIT at $(date)"

# ---------------------------------------------------------------------------
# Auto-resubmit if SMART_RESUBMIT=1 and work remains
# ---------------------------------------------------------------------------
if [[ "${SMART_RESUBMIT:-0}" == "1" ]]; then
    # Check queue status (exit 0 = all done, exit 1 = work remains)
    if python run_70b_distributed.py --status --quiet; then
        echo "[auto] All chunks done — running merge."
        python run_70b_distributed.py --merge
    else
        echo "[auto] Work remains — resubmitting one worker via submit_70b_smart.sh..."
        # Submit just one replacement worker (others may still be running).
        bash submit_70b_smart.sh --workers 1
    fi
fi

echo "Job $SLURM_JOB_ID finished at $(date)"
