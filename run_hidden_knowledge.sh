#!/bin/bash
#SBATCH --job-name=hidden_knowledge_unlearning
#SBATCH --output=hidden_knowledge_%j.out
#SBATCH --error=hidden_knowledge_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# Activate the existing 'unlearning' conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

# Cache HuggingFace models in home dir (NFS-mounted, persists across jobs)
export HF_HOME=$HOME/.cache/huggingface

echo ""
echo "Starting hidden_knowledge_after_unlearning.py..."
python hidden_knowledge_after_unlearning.py

echo "Job finished at $(date)"
