#!/bin/bash
#SBATCH --job-name=gibbase
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs_gibberish/base_%j.log
# base failed in the first two arrays: meta-llama is gated=manual and no token
# was present. A token is now installed, so both constructions run here.
cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
export KINT_HF_HOME=$HOME/hf_kint
echo "=== base on $(hostname) $(date)"
~/miniconda3/envs/insideout_unlearn_4/bin/python plots/gibberish_full.py base
~/miniconda3/envs/insideout_unlearn_4/bin/python plots/gibberish_full.py base --quad
echo "=== base done $(date)"
