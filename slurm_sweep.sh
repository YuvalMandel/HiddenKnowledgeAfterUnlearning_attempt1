#!/bin/bash
#SBATCH --job-name=hk_sweep
#SBATCH --output=logs/sweep_%a_%j.out
#SBATCH --error=logs/sweep_%a_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-63

# Task layout:  task_id = method_idx * 8 + (checkpoint_num - 1)
#   tasks  0- 7 : GradDiff  ck1-ck8
#   tasks  8-15 : RMU       ck1-ck8
#   tasks 16-23 : RMU-LAT   ck1-ck8
#   tasks 24-31 : RepNoise  ck1-ck8
#   tasks 32-39 : ELM       ck1-ck8
#   tasks 40-47 : RR        ck1-ck8
#   tasks 48-55 : TAR       ck1-ck8
#   tasks 56-63 : PB&J      ck1-ck8

echo "Job started at $(date)"
echo "Running on node: $(hostname), array task: ${SLURM_ARRAY_TASK_ID}"
echo "GPU info:"
nvidia-smi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)

mkdir -p logs

SWEEP_METHODS=("GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB&J")
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 8))
CK_NUM=$(( (SLURM_ARRAY_TASK_ID % 8) + 1 ))
METHOD="${SWEEP_METHODS[$METHOD_IDX]}"

echo ""
echo "Method: ${METHOD}  Checkpoint: ${CK_NUM}"
python hidden_knowledge_after_unlearning.py \
    --stage sweep \
    --method "${METHOD}" \
    --checkpoint ${CK_NUM}

echo "Job finished at $(date)"
