#!/bin/bash
#SBATCH --job-name=probek
#SBATCH --partition=public
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=inside_out_logs/probek_%A_%a.out
#SBATCH --error=inside_out_logs/probek_%A_%a.err
#
# KNOWN_ISSUES #28: app:probes and app:cross-probe report the probe's binary
# classification accuracy under the name "K_int". This run produces the real
# thing -- per-question K for every probe family, layer config and probe
# direction the two appendices claim to report.
#
# Why this was missing: submit_inside_out_v3.sh ran
#     --clfs LR --lcs full --no_cv --no_cross_probe
# so RF, AdaBoost, the 12-22 band and both cross-probe directions were never
# scored per question. The pipeline supported all of them already.
#
# DARWIN, not Newton: stage_probe reads the cached bio_hs.npy and never touches
# a GPU, and Newton rejects CPU-only jobs ("please use a Darwin cluster").
# Darwin shares Newton's home directory, so the 84 GB of hidden states is
# already visible here -- nothing to stage.
#
# Requires the 2026-09-14 fix to _train_and_score: the best-layer sweep used to
# be gated on clf_name=="LR", so RF and AdaBoost would have silently reported a
# layer-16 fit as the selected best layer.
#
# Pilot (job 83466): RF at best_layer, one model, 16 CPUs -> 10m51s, 23.5 GB
# peak, and best_layer varied per fold [2,7,2,1,2] rather than sticking at the
# old default of 16 -- which is the check that the sweep fix engaged.
# 'full' is dropped: app:probes' ML column is layers 12-22, i.e. 'multi'.
#
# Submit:  ssh yuval.mandel@132.68.38.100
#          cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
#          /usr/local/bin/sbatch slurm_probe_k_full.sh

set -euo pipefail

REPO=/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
PYTHON=/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3

MODELS=(base GradDiff_ck8 RMU_ck8 RMU-LAT_ck8 RepNoise_ck8 ELM_ck8 RR_ck8 TAR_ck8 PB_J_ck8)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

cd "$REPO"
mkdir -p inside_out_logs

echo "host=$(hostname) model=$MODEL start=$(date -Is)"

# Cross-probe is ON (no --no_cross_probe): base->method and method->base are
# exactly what app:cross-probe reports and has never had K for.
"$PYTHON" inside_out_knowledge.py \
    --stage probe \
    --model_id "$MODEL" \
    --domains bio \
    --clfs LR RF AdaBoost \
    --lcs best_layer multi \
    --n_jobs 32 \
    --out_suffix kfull

echo "model=$MODEL end=$(date -Is)"
