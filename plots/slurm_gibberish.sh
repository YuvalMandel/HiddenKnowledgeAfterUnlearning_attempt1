#!/bin/bash
#SBATCH --job-name=gibber
#SBATCH --partition=public
#SBATCH --array=0-8
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs_gibberish/%A_%a.log
# Generation needs a GPU, so this is Newton (not Darwin) and the GPU type is
# pinned: plain --gres=gpu:1 lands on an 11 GB 2080 Ti and OOMs an 8B model.
cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
export KINT_HF_HOME=$HOME/hf_kint
export HF_HUB_ENABLE_HF_TRANSFER=0
METHODS=(base GradDiff RMU RMU-LAT RepNoise ELM RR TAR PB_J)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
echo "=== $M on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
~/miniconda3/envs/insideout_unlearn_4/bin/python plots/gibberish_full.py "$M"
echo "=== $M done $(date)"
