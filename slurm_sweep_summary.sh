#!/bin/bash
#SBATCH --job-name=hk_sweep_sum
#SBATCH --output=logs/sweep_sum_%a_%j.out
#SBATCH --error=logs/sweep_sum_%a_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=public
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7

echo "Job started at $(date)"
echo "Running on node: $(hostname), array task: ${SLURM_ARRAY_TASK_ID}"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p logs

SWEEP_METHODS=("GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB&J")
METHOD="${SWEEP_METHODS[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "Sweep summary for: ${METHOD}"
python hidden_knowledge_after_unlearning.py \
    --stage sweep_summary \
    --method "${METHOD}"

echo "Job finished at $(date)"
