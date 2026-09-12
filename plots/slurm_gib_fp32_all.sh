#!/bin/bash
#SBATCH --job-name=gibf32
#SBATCH --partition=public
#SBATCH --array=0-8
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs_gibberish/f32all_%A_%a.log
# float32 -- the dtype the LLM-GAT checkpoints are released in. bfloat16 costs
# three mantissa bits and moved RepNoise's rate by 4.7 pp; float32 and float16
# agree to 0.1 pp. base is natively bfloat16, so for it this is an upcast, which
# is lossless and keeps the whole table on one dtype.
# 8B x 4B = 32 GB on a 48 GB L40, so batch 4.
cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
export KINT_HF_HOME=$HOME/hf_kint
METHODS=(base GradDiff RMU RMU-LAT RepNoise ELM RR TAR PB_J)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
echo "=== $M float32 on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GEN_DTYPE=float32 GEN_BATCH=4 GEN_TAG="_f32" \
  ~/miniconda3/envs/insideout_unlearn_4/bin/python plots/gibberish_full.py "$M" --quad
echo "=== $M float32 done $(date)"
