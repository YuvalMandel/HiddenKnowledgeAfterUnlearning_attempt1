#!/bin/bash
#SBATCH --job-name=gibquad
#SBATCH --partition=public
#SBATCH --array=1-8
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs_gibberish/quad_%A_%a.log
# All four options of every question: 5,092 prompts. Weights are cached from the
# first array, so this is generation only. Task 0 (base) is omitted -- its repo
# is gated and no HF token is present on this account.
cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
export KINT_HF_HOME=$HOME/hf_kint
METHODS=(base GradDiff RMU RMU-LAT RepNoise ELM RR TAR PB_J)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
echo "=== $M quad on $(hostname) $(date)"
~/miniconda3/envs/insideout_unlearn_4/bin/python plots/gibberish_full.py "$M" --quad
echo "=== $M quad done $(date)"
