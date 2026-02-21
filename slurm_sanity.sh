#!/bin/bash
#SBATCH --job-name=hk_sanity
#SBATCH --output=logs/sanity_%j.out
#SBATCH --error=logs/sanity_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=public
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
# Loads the base model, runs two quick sanity checks, and exits.
# Runs before stage 1 (base) so that padding / decoding bugs surface early.

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HF_HOME=$HOME/.cache/huggingface

mkdir -p logs

echo ""
echo "Starting sanity checks..."
python hidden_knowledge_after_unlearning.py --stage sanity

echo "Job finished at $(date)"
