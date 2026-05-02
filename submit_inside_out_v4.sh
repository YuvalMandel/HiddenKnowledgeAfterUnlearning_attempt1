#!/bin/bash
# submit_inside_out_v4.sh — Per-layer K probes (bio, LR, no CV, no cross-probe).
#
# Output: k_scores_layer.parquet per model (separate from k_scores.parquet).
# Priority queue: base → ck8 of each method → ck1-7 of each method (sequential
# within method, parallel across methods).
#
# Models whose bio_hs.npy already exists skip extract and submit probe immediately.
#
# Usage:  bash submit_inside_out_v4.sh

set -e

REPO="/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1"
PYTHON="/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3"
SCRIPT="$REPO/inside_out_knowledge.py"
LOGS="$REPO/inside_out_logs"
OUT="$REPO/inside_out_out"
mkdir -p "$LOGS"

PROBE_FLAGS="--domains bio --clfs LR --lcs layer --no_cv --no_cross_probe --n_jobs 4 --out_suffix layer"
METHODS_ORDER=("GradDiff" "PB_J" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR")

DISK_CHECK='
AVAIL_GB=$(df -BG "$HOME" | awk "NR==2{gsub(/G/,\"\",\$4); print \$4}")
NEED_GB=20
if [[ "$AVAIL_GB" -lt "$NEED_GB" ]]; then
    echo "ERROR: only ${AVAIL_GB}GB free (need ${NEED_GB}GB). Aborting."
    exit 1
fi
echo "Disk OK: ${AVAIL_GB}GB free."
'

CONDA_SETUP="source /home/yuval.mandel/miniconda3/etc/profile.d/conda.sh && conda activate unlearning"

needs_extract() { [[ ! -f "${OUT}/$1/bio_hs.npy" ]]; }
needs_probe()   { [[ ! -f "${OUT}/$1/k_scores_layer.parquet" ]]; }

submit_extract() {
    local mid="$1"; local dep="${2:-}"; local nm="${mid//-/_}"
    sbatch --parsable \
      --job-name="io_ext_${nm}" \
      --output="${LOGS}/ext_${nm}_%j.out" \
      --error="${LOGS}/ext_${nm}_%j.err" \
      --time=02:30:00 --mem=32G --cpus-per-task=4 \
      --gres=gpu:L40:1 --partition=public \
      ${dep} \
      --wrap="set -e
${CONDA_SETUP}
export HF_HOME=\$HOME/.cache/huggingface
export HF_TOKEN=\$(cat \$HF_HOME/token 2>/dev/null || true)
cd ${REPO}
${DISK_CHECK}
if [ -f \"${OUT}/${mid}/bio_hs.npy\" ]; then echo 'Already done: ${mid}'; exit 0; fi
echo '=== Extract: ${mid} ==='
for attempt in 1 2 3; do
    ${PYTHON} ${SCRIPT} --stage extract --model_id ${mid} --domains bio && break
    echo \"Attempt \$attempt failed. Retrying in 120s...\"
    sleep 120
done
echo '=== Done: ${mid} ==='"
}

submit_probe() {
    local mid="$1"; local dep="${2:-}"; local nm="${mid//-/_}"
    local dep_flag=""
    [[ -n "$dep" ]] && dep_flag="--dependency=afterok:${dep}"
    sbatch --parsable \
      --job-name="io_prb_lyr_${nm}" \
      --output="${LOGS}/prb_lyr_${nm}_%j.out" \
      --error="${LOGS}/prb_lyr_${nm}_%j.err" \
      --time=01:00:00 --mem=32G --cpus-per-task=4 \
      --partition=public ${dep_flag} \
      --wrap="set -e
${CONDA_SETUP}
export HF_HOME=\$HOME/.cache/huggingface
cd ${REPO}
if [ -f \"${OUT}/${mid}/k_scores_layer.parquet\" ]; then echo 'Already done: ${mid}'; exit 0; fi
echo '=== Probe (per-layer): ${mid} ==='
${PYTHON} ${SCRIPT} --stage probe --model_id ${mid} ${PROBE_FLAGS}
echo '=== Done: ${mid} ==='"
}


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — base
# ══════════════════════════════════════════════════════════════════════════════
echo "=== Tier 1: base ==="

if needs_probe "base"; then
    if needs_extract "base"; then
        JID_BASE_EXT=$(submit_extract "base")
        echo "  base ext: ${JID_BASE_EXT}"
        JID_BASE_PRB=$(submit_probe "base" "${JID_BASE_EXT}")
    else
        JID_BASE_PRB=$(submit_probe "base")
    fi
    echo "  base prb: ${JID_BASE_PRB}"
else
    echo "  base: already done"
fi


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — all ck8 (parallel)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Tier 2: all ck8 ==="
declare -A JID_CK8_EXT

for method in "${METHODS_ORDER[@]}"; do
    mid="${method}_ck8"

    if needs_extract "$mid"; then
        jid=$(submit_extract "$mid")
        JID_CK8_EXT[$method]=$jid
        echo "  ${method} ck8 ext: ${jid}"
    else
        JID_CK8_EXT[$method]=""
        echo "  ${method} ck8 ext: already done"
    fi

    if needs_probe "$mid"; then
        jid_prb=$(submit_probe "$mid" "${JID_CK8_EXT[$method]}")
        echo "  ${method} ck8 prb: ${jid_prb}"
    else
        echo "  ${method} ck8 prb: already done"
    fi
done


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 — ck1-7: sequential within method, parallel across methods
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Tier 3: ck1-7 ==="

for method in "${METHODS_ORDER[@]}"; do
    echo "  --- ${method} ck1-7 ---"
    prev_ext_jid="${JID_CK8_EXT[$method]}"
    jid_ext=""

    for ck in $(seq 1 7); do
        mid="${method}_ck${ck}"

        if needs_extract "$mid"; then
            if [[ -n "$prev_ext_jid" ]]; then
                jid_ext=$(submit_extract "$mid" "--dependency=afterok:${prev_ext_jid}")
            else
                jid_ext=$(submit_extract "$mid")
            fi
            echo "    ${mid} ext: ${jid_ext} (dep=${prev_ext_jid:-none})"
            prev_ext_jid=$jid_ext
        else
            echo "    ${mid} ext: already done"
            prev_ext_jid=""
            jid_ext=""
        fi

        if needs_probe "$mid"; then
            jid_prb=$(submit_probe "$mid" "$jid_ext")
            echo "    ${mid} prb: ${jid_prb}"
        else
            echo "    ${mid} prb: already done"
        fi

        jid_ext=""
    done
done

echo ""
echo "All jobs submitted. Monitor: squeue -u yuval.mandel"
echo "Logs: ${LOGS}/"
