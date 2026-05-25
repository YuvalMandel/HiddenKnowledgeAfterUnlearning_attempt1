#!/bin/bash
#SBATCH --job-name=wmdp_categorize
#SBATCH --output=logs/categorize_%j.out
#SBATCH --error=logs/categorize_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs

cd $HOME/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
python categorize_wmdp_bio.py

echo "Job finished at $(date)"
