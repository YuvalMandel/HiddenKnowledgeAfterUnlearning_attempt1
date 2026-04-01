#!/bin/bash
#SBATCH --job-name=hk_kfold
#SBATCH --output=logs/kfold_%a_%j.out
#SBATCH --error=logs/kfold_%a_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=public
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
# No GPU needed — uses only sklearn on pre-computed .npy hidden states.
# Array: 5 folds × 9 models = 45 tasks (0-44).
#SBATCH --array=0-44

echo "Job started at $(date)"
echo "Node: $(hostname)  array_task: ${SLURM_ARRAY_TASK_ID}"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p logs checkpoints/kfold

# Map flat array index to (fold, model).
# Models order must match ALL_MODELS in kfold_probe.py.
MODELS=("base" "GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB&J")
N_MODELS=${#MODELS[@]}   # 9

FOLD=$(( SLURM_ARRAY_TASK_ID / N_MODELS ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID % N_MODELS ))
MODEL="${MODELS[$MODEL_IDX]}"

echo ""
echo "Running: fold=${FOLD}  model=${MODEL}"
python kfold_probe.py --stage kfold_train --fold "${FOLD}" --model "${MODEL}"

echo "Job finished at $(date)"
