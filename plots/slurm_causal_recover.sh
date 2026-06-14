#!/bin/bash
#SBATCH --job-name=causal_recover
#SBATCH --output=logs/causal_recover_%j.out
#SBATCH --error=logs/causal_recover_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
echo "start $(date) on $(hostname)"; nvidia-smi
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p logs
cd $HOME/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
python plots/causal_recover.py
echo "done $(date)"
