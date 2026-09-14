#!/bin/bash
#SBATCH --job-name=probek_pilot
#SBATCH --partition=public
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=inside_out_logs/probek_pilot_%j.out
#SBATCH --error=inside_out_logs/probek_pilot_%j.err
#
# Timing pilot for KNOWN_ISSUES #28: how expensive is a best-layer sweep for a
# non-LR probe family? RF at best_layer fits 33 layers x 5 folds on raw 4096-dim
# features, which is the dominant cost of the full conversion.
#
# CPU only -- stage_probe reads the cached bio_hs.npy and never touches a GPU.
# Cross-probe is off here to isolate the own-probe cost; the full run roughly
# doubles it.

set -euo pipefail

REPO=/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
PYTHON=/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3

cd "$REPO"
mkdir -p inside_out_logs

echo "host=$(hostname) start=$(date -Is)"
/usr/bin/time -v "$PYTHON" inside_out_knowledge.py \
    --stage probe \
    --model_id RMU_ck8 \
    --domains bio \
    --clfs RF \
    --lcs best_layer \
    --no_cross_probe \
    --n_jobs 16 \
    --out_suffix pilot_rf
echo "end=$(date -Is) rc=$?"
