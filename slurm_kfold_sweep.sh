#!/bin/bash
#SBATCH --job-name=hk_kfold_sweep
#SBATCH --output=logs/kfold_sweep_%a_%j.out
#SBATCH --error=logs/kfold_sweep_%a_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=public
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
# No GPU needed — uses only sklearn on pre-computed .npy hidden states.
# Array: 8 methods × 8 checkpoints × 5 folds = 320 tasks (0-319).
#SBATCH --array=0-319

echo "Job started at $(date)"
echo "Node: $(hostname)  array_task: ${SLURM_ARRAY_TASK_ID}"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p logs checkpoints/kfold_sweep

# Task layout:
#   method_idx = task_id // 40      (0-7  → 8 methods)
#   ck         = (task_id % 40) // 5 + 1  (1-8  → 8 checkpoints)
#   fold       = (task_id % 40) % 5        (0-4  → 5 folds)
METHODS=("GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB&J")

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 40 ))
INNER=$(( SLURM_ARRAY_TASK_ID % 40 ))
CK=$(( INNER / 5 + 1 ))
FOLD=$(( INNER % 5 ))
METHOD="${METHODS[$METHOD_IDX]}"

echo ""
echo "Running: method=${METHOD}  ck=${CK}  fold=${FOLD}"
python kfold_probe.py --stage kfold_sweep_train \
    --method "${METHOD}" --checkpoint "${CK}" --fold "${FOLD}"

echo "Job finished at $(date)"
