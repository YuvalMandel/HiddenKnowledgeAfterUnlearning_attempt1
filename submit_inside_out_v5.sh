#!/bin/bash
# submit_inside_out_v5.sh — Full-layer 5-fold CV probe (bio, LR).
#
# Assumes bio_hs.npy and bio_ext.npy already exist for all models.
# Deletes stale k_scores.parquet files before submitting so the new
# 60/40 ShuffleSplit CV implementation runs fresh.
#
# Usage:  bash submit_inside_out_v5.sh

set -e

REPO="/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1"
PYTHON="/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3"
SCRIPT="$REPO/inside_out_knowledge.py"
LOGS="$REPO/inside_out_logs"
OUT="$REPO/inside_out_out"
mkdir -p "$LOGS"

PROBE_FLAGS="--domains bio --clfs LR --lcs full --no_cross_probe --n_jobs 8"
METHODS=("GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB_J")
CONDA_SETUP=". /home/yuval.mandel/miniconda3/etc/profile.d/conda.sh && conda activate unlearning"

needs_hs()    { [[ ! -f "${OUT}/$1/bio_hs.npy" ]]; }
needs_probe() { [[ ! -f "${OUT}/$1/k_scores.parquet" ]]; }

submit_probe() {
    local mid="$1"; local nm="${mid//-/_}"
    sbatch --parsable \
      --job-name="io_prb_cv_${nm}" \
      --output="${LOGS}/prb_cv_${nm}_%j.out" \
      --error="${LOGS}/prb_cv_${nm}_%j.err" \
      --time=01:30:00 --mem=32G --cpus-per-task=8 \
      --partition=public \
      --wrap="set -e
${CONDA_SETUP}
export HF_HOME=\$HOME/.cache/huggingface
cd ${REPO}
if [ -f \"${OUT}/${mid}/k_scores.parquet\" ]; then echo 'Already done: ${mid}'; exit 0; fi
if [ ! -f \"${OUT}/${mid}/bio_hs.npy\" ]; then echo 'Missing HS: ${mid}'; exit 1; fi
echo '=== Probe CV (full): ${mid} ==='
${PYTHON} ${SCRIPT} --stage probe --model_id ${mid} ${PROBE_FLAGS}
echo '=== Done: ${mid} ==='"
}

# ── Delete stale k_scores.parquet so new CV runs fresh ──────────────────────
echo "=== Removing stale k_scores.parquet files ==="
for mid in base $(for m in "${METHODS[@]}"; do for ck in $(seq 1 8); do echo "${m}_ck${ck}"; done; done); do
    f="${OUT}/${mid}/k_scores.parquet"
    if [[ -f "$f" ]]; then
        rm "$f"
        echo "  removed: ${mid}/k_scores.parquet"
    fi
done

# ── Submit all probe jobs ────────────────────────────────────────────────────
echo ""
echo "=== Submitting probe jobs ==="

# base
if needs_hs "base"; then
    echo "  SKIP base: bio_hs.npy missing"
elif needs_probe "base"; then
    jid=$(submit_probe "base")
    echo "  base: ${jid}"
else
    echo "  base: already done"
fi

# methods
for method in "${METHODS[@]}"; do
    for ck in $(seq 1 8); do
        mid="${method}_ck${ck}"
        if needs_hs "$mid"; then
            echo "  SKIP ${mid}: bio_hs.npy missing"
        elif needs_probe "$mid"; then
            jid=$(submit_probe "$mid")
            echo "  ${mid}: ${jid}"
        else
            echo "  ${mid}: already done"
        fi
    done
done

echo ""
echo "All jobs submitted. Monitor: squeue -u yuval.mandel"
echo "Logs: ${LOGS}/"
