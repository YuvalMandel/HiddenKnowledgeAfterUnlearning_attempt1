#!/bin/bash
#SBATCH --job-name=hk_70b
#SBATCH --output=logs/llama70b_%j.out
#SBATCH --error=logs/llama70b_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)

mkdir -p logs

echo ""
echo "Starting llama70b stage..."
python hidden_knowledge_after_unlearning.py --stage llama70b

echo "Job finished at $(date)"
