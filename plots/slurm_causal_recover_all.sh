#!/bin/bash
#SBATCH --job-name=recall_all
#SBATCH --output=logs/recall_all_%j.out
#SBATCH --error=logs/recall_all_%j.err
#SBATCH --time=03:00:00
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
for M in GradDiff RMU RMU-LAT RepNoise ELM RR TAR PB_J; do
  echo "##### $M #####"
  python plots/causal_recover.py $M
done
echo "done $(date)"
