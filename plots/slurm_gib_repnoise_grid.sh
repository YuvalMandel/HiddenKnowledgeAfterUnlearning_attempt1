#!/bin/bash
#SBATCH --job-name=gibgrid
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs_gibberish/grid_%j.log
# RepNoise showed the largest published-vs-reproduced gap (93.2 -> 88.2 on the
# same 2,292 prompts). Vary the two knobs that break bit-reproducibility of
# batched greedy decoding and see whether the rate moves within that band.
cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
export KINT_HF_HOME=$HOME/hf_kint
PY=~/miniconda3/envs/insideout_unlearn_4/bin/python
for D in float16 bfloat16; do
  for B in 8 32; do
    echo "=== RepNoise dtype=$D batch=$B $(date)"
    GEN_DTYPE=$D GEN_BATCH=$B GEN_TAG="_${D}_b${B}" $PY plots/gibberish_full.py RepNoise --quad
  done
done
echo "=== grid done $(date)"
