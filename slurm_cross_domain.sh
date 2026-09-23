#!/bin/bash
#SBATCH --job-name=cross_domain
#SBATCH --partition=public
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=inside_out_logs/cross_%A_%a.out
#SBATCH --error=inside_out_logs/cross_%A_%a.err
#
# Cross-domain probe transfer, DARWIN. CPU only -- Newton rejects CPU-only jobs.
#
# Needs no GPU and no extraction: every {bio,cyber}_hs.npy already exists, so
# this only refits probes. Nothing is written but one small parquet per model.
#
# Zephyr released first (33 layers x 4096). Mixtral is the same shape; Yi is
# 61 x 7168, so its layer sweep is roughly 3x longer -- raise --time before
# adding it.
set -euo pipefail

PYTHON=/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3
REPO=/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1

# submit the pair you want: 0-1 zephyr, 2-3 mixtral, 4-5 yi.
# Yi is 61 layers x 7168, roughly 3x the sweep -- raise --time before it.
MODELS=(zephyr_base zephyr_rmu mixtral_base mixtral_rmu yi_base yi_rmu)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

cd "$REPO"
echo "host=$(hostname) model=$MODEL start=$(date -Is)"
"$PYTHON" inside_out_knowledge.py --stage cross_domain --model_id "$MODEL" --n_jobs 16
echo "model=$MODEL end=$(date -Is)"
ls -l "inside_out_out/$MODEL/k_scores_cross_domain.parquet"
