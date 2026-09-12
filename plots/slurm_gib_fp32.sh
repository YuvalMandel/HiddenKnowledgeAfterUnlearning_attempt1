#!/bin/bash
#SBATCH --job-name=gibfp32
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs_gibberish/fp32_%j.log
# The LLM-GAT checkpoints are stored float32, so bfloat16 and float16 are BOTH
# downcasts of the released weights -- neither is "native". float32 is the
# weights as published: 8B x 4B = 32 GB, which fits an L40's 48 GB at batch 4.
# Whichever downcast this agrees with is the faithful one.
cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
export KINT_HF_HOME=$HOME/hf_kint
PY=~/miniconda3/envs/insideout_unlearn_4/bin/python
echo "=== RepNoise float32 $(date)"
GEN_DTYPE=float32 GEN_BATCH=4 GEN_TAG="_float32_b4" $PY plots/gibberish_full.py RepNoise --quad
echo "=== fp32 done $(date)"
