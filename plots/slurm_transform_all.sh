#!/bin/bash
#SBATCH --job-name=tall
#SBATCH --output=logs/tall_%A_%a.out
#SBATCH --error=logs/tall_%A_%a.err
#SBATCH --time=00:50:00
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7
echo "start $(date) on $(hostname) task $SLURM_ARRAY_TASK_ID"; nvidia-smi
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p logs
cd $HOME/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
METHODS=(GradDiff RMU RMU-LAT RepNoise ELM RR TAR PB_J)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
echo "##### METHOD=$M #####"
python plots/causal_recover_transform.py $M
echo "done $(date)"
