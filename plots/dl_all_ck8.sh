#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
export HF_HOME=$HOME/.cache/huggingface
for s in graddiff rmu rmu-lat elm rr tar pbj; do
  echo "== downloading $s =="
  huggingface-cli download LLM-GAT/llama-3-8b-instruct-$s-checkpoint-8 --exclude "original/*"
done
echo ALLDONE
