#!/bin/bash
#SBATCH --job-name=hk_summary
#SBATCH --output=logs/summary_%j.out
#SBATCH --error=logs/summary_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=public
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
# No GPU needed: summary stage only loads checkpoints and prints statistics.

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HF_HOME=$HOME/.cache/huggingface

mkdir -p logs

echo ""
echo "Starting summary stage..."
python hidden_knowledge_after_unlearning.py --stage summary

echo "Job finished at $(date)"
