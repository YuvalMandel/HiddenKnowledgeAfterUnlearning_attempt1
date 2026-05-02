#!/bin/bash
# submit_inside_out_v3.sh — Inside-Out pipeline (bio, LR, full layer, own probe, no CV).
#
# Parallelism / scheduling:
#   • All extracts (base + all ck8) start immediately with NO inter-dependencies.
#   • Each probe starts as soon as its own extract finishes.
#   • ck1-7 within each method run SEQUENTIALLY (ck1→ck2→…→ck7) to avoid
#     multiple simultaneous 16 GB downloads from the same method.
#   • Different methods' ck1-7 chains DO run in parallel with each other.
#   • Each ck1-7 chain starts after its method's ck8 extract finishes (GPU slot freed).
#
# Robustness:
#   • Extract jobs retry up to 3 times (HF network errors).
#   • Every extract checks for ≥20 GB free disk before running; aborts if not.
#   • Skip already-done models (checks bio_hs.npy / k_scores.parquet).
#
# Usage:  bash submit_inside_out_v3.sh

set -e

REPO="/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1"
PYTHON="/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3"
SCRIPT="$REPO/inside_out_knowledge.py"
LOGS="$REPO/inside_out_logs"
OUT="$REPO/inside_out_out"
mkdir -p "$LOGS"

PROBE_FLAGS="--domains bio --clfs LR --lcs full --no_cv --no_cross_probe --n_jobs 4"
METHODS_ORDER=("GradDiff" "PB_J" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR")

# ── Disk space check snippet (embedded in every extract job) ─────────────────
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

# ── Helpers: check if work is already done ───────────────────────────────────
needs_extract() { [[ ! -f "${OUT}/$1/bio_hs.npy" ]]; }
needs_probe()   { [[ ! -f "${OUT}/$1/k_scores.parquet" ]]; }

# ── Submit a single extract job ──────────────────────────────────────────────
# submit_extract <model_id> [--dependency=afterok:JID]
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

# ── Submit a single probe job ────────────────────────────────────────────────
# submit_probe <model_id> <dep_jid>
submit_probe() {
    local mid="$1"; local dep="$2"; local nm="${mid//-/_}"
    sbatch --parsable \
      --job-name="io_prb_${nm}" \
      --output="${LOGS}/prb_${nm}_%j.out" \
      --error="${LOGS}/prb_${nm}_%j.err" \
      --time=00:30:00 --mem=32G --cpus-per-task=4 \
      --partition=public --dependency=afterok:${dep} \
      --wrap="set -e
${CONDA_SETUP}
export HF_HOME=\$HOME/.cache/huggingface
cd ${REPO}
if [ -f \"${OUT}/${mid}/k_scores.parquet\" ]; then echo 'Already done: ${mid}'; exit 0; fi
echo '=== Probe: ${mid} ==='
${PYTHON} ${SCRIPT} --stage probe --model_id ${mid} ${PROBE_FLAGS}
echo '=== Done: ${mid} ==='"
}


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — base (no dependencies)
# ══════════════════════════════════════════════════════════════════════════════
echo "=== Tier 1+2: base and all ck8 extracts (all in parallel) ==="

if needs_extract "base"; then
    JID_BASE_EXT=$(submit_extract "base")
    echo "  base ext: ${JID_BASE_EXT}"
    JID_BASE_PRB=$(submit_probe "base" "${JID_BASE_EXT}")
    echo "  base prb: ${JID_BASE_PRB}"
else
    echo "  base: already done"
fi


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — all ck8 extracts + probes, all in parallel (no dep on base)
# ══════════════════════════════════════════════════════════════════════════════
declare -A JID_CK8_EXT

for method in "${METHODS_ORDER[@]}"; do
    mid="${method}_ck8"
    if needs_extract "$mid"; then
        jid=$(submit_extract "$mid")
        JID_CK8_EXT[$method]=$jid
        echo "  ${method} ck8 ext: ${jid}"
    else
        echo "  ${method} ck8 ext: already done"
        JID_CK8_EXT[$method]=""
    fi

    if needs_probe "$mid"; then
        if [[ -n "${JID_CK8_EXT[$method]}" ]]; then
            jid_prb=$(submit_probe "$mid" "${JID_CK8_EXT[$method]}")
            echo "  ${method} ck8 prb: ${jid_prb}"
        else
            # extract already done → probe can start immediately
            jid_prb=$(sbatch --parsable \
              --job-name="io_prb_${method//-/_}_ck8" \
              --output="${LOGS}/prb_${method//-/_}_ck8_%j.out" \
              --error="${LOGS}/prb_${method//-/_}_ck8_%j.err" \
              --time=00:30:00 --mem=32G --cpus-per-task=4 --partition=public \
              --wrap="set -e; ${CONDA_SETUP}; export HF_HOME=\$HOME/.cache/huggingface; cd ${REPO}; ${PYTHON} ${SCRIPT} --stage probe --model_id ${mid} ${PROBE_FLAGS}")
            echo "  ${method} ck8 prb: ${jid_prb} (no ext dep)"
        fi
    else
        echo "  ${method} ck8 prb: already done"
    fi
done


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 — ck1-7: sequential WITHIN each method, parallel ACROSS methods
#          Each method's chain starts after its ck8 extract (GPU slot freed).
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Tier 3: ck1-7 sequential chains (parallel across methods) ==="

for method in "${METHODS_ORDER[@]}"; do
    echo "  --- ${method} ck1-7 ---"

    # First job in this chain depends on ck8 extract finishing (or starts immediately)
    if [[ -n "${JID_CK8_EXT[$method]}" ]]; then
        prev_ext_jid="${JID_CK8_EXT[$method]}"
    else
        prev_ext_jid=""
    fi

    for ck in $(seq 1 7); do
        mid="${method}_ck${ck}"

        # --- extract ---
        if needs_extract "$mid"; then
            if [[ -n "$prev_ext_jid" ]]; then
                jid_ext=$(submit_extract "$mid" "--dependency=afterok:${prev_ext_jid}")
            else
                jid_ext=$(submit_extract "$mid")
            fi
            echo "    ${mid} ext: ${jid_ext} (dep=${prev_ext_jid:-none})"
            prev_ext_jid=$jid_ext   # next ck waits for this one
        else
            echo "    ${mid} ext: already done"
            prev_ext_jid=""   # no dep needed for next ck if this one is already done
        fi

        # --- probe ---
        if needs_probe "$mid"; then
            if [[ -n "$jid_ext" ]]; then
                jid_prb=$(submit_probe "$mid" "$jid_ext")
                echo "    ${mid} prb: ${jid_prb}"
            else
                # extract done but probe not: run immediately
                jid_prb=$(sbatch --parsable \
                  --job-name="io_prb_${method//-/_}_ck${ck}" \
                  --output="${LOGS}/prb_${method//-/_}_ck${ck}_%j.out" \
                  --error="${LOGS}/prb_${method//-/_}_ck${ck}_%j.err" \
                  --time=00:30:00 --mem=32G --cpus-per-task=4 --partition=public \
                  --wrap="set -e; ${CONDA_SETUP}; export HF_HOME=\$HOME/.cache/huggingface; cd ${REPO}; ${PYTHON} ${SCRIPT} --stage probe --model_id ${mid} ${PROBE_FLAGS}")
                echo "    ${mid} prb: ${jid_prb} (no ext dep)"
            fi
        else
            echo "    ${mid} prb: already done"
        fi

        # reset jid_ext for next iteration (will be set again if extract is needed)
        jid_ext=""
    done
done

echo ""
echo "All jobs submitted. Monitor: squeue -u yuval.mandel"
echo "Logs: ${LOGS}/"
