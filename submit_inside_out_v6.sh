#!/bin/bash
# submit_inside_out_v6.sh — KFold 65/15/20 CV probe (bio, LR, full + best_layer).
#
# Split design: KFold(n_splits=5), shuffled once by SEED before splitting.
# Non-overlapping test sets; val set used for C (and layer) selection.
# Layer configs: "full" (all layers, PCA-256, C validated) and
#                "best_layer" (single best layer, (layer,C) validated).
#
# Usage: bash submit_inside_out_v6.sh

set -e

REPO="/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1"
PYTHON="/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3"
SCRIPT="$REPO/inside_out_knowledge.py"
LOGS="$REPO/inside_out_logs"
OUT="$REPO/inside_out_out"
mkdir -p "$LOGS"

PROBE_FLAGS="--domains bio --clfs LR --lcs full best_layer --no_cross_probe --n_jobs 8"
METHODS=("GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB_J")
CONDA_SETUP=". /home/yuval.mandel/miniconda3/etc/profile.d/conda.sh && conda activate unlearning"

needs_hs()    { [[ ! -f "${OUT}/$1/bio_hs.npy" ]]; }
needs_probe() { [[ ! -f "${OUT}/$1/k_scores.parquet" ]]; }

submit_probe() {
    local mid="$1"; local nm="${mid//-/_}"
    sbatch --parsable \
      --job-name="io_prb_v6_${nm}" \
      --output="${LOGS}/prb_v6_${nm}_%j.out" \
      --error="${LOGS}/prb_v6_${nm}_%j.err" \
      --time=24:00:00 --mem=48G --cpus-per-task=8 \
      --partition=public \
      --wrap="set -e
${CONDA_SETUP}
export HF_HOME=\$HOME/.cache/huggingface
cd ${REPO}
if [ -f \"${OUT}/${mid}/k_scores.parquet\" ]; then echo 'Already done: ${mid}'; exit 0; fi
if [ ! -f \"${OUT}/${mid}/bio_hs.npy\" ]; then echo 'Missing HS: ${mid}'; exit 1; fi
echo '=== Probe CV v6 (full+best_layer): ${mid} ==='
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
