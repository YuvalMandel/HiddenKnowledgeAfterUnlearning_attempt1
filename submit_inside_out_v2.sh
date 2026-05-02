#!/bin/bash
# submit_inside_out_v2.sh
#
# Priority-ordered submission for the Inside-Out pipeline.
# Scope: bio domain, LR classifier, full layer config, own probe, no CV.
#
# Tier 1: base extract → base probe
# Tier 2: all 8 ck8 extracts in parallel (after base extract)
#         → all 8 ck8 probes in parallel (each after its own extract)
# Tier 3: GradDiff ck1-7 extract array → probe array
#         PB_J     ck1-7 extract array → probe array
#         RMU      ck1-7 extract array → probe array
#         RMU-LAT  ck1-7 extract array → probe array
#         RepNoise ck1-7 extract array → probe array
#         ELM      ck1-7 extract array → probe array
#         RR       ck1-7 extract array → probe array
#         TAR      ck1-7 extract array → probe array
#
# Usage: bash submit_inside_out_v2.sh

set -e

REPO="/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1"
PYTHON="/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3"
SCRIPT="$REPO/inside_out_knowledge.py"
LOGS="$REPO/inside_out_logs"
mkdir -p "$LOGS"

# Probe flags shared by every probe job
PROBE_FLAGS="--domains bio --clfs LR --lcs full --no_cv --no_cross_probe --n_jobs 4"

# Method priority order for ck1-7
METHODS_ORDER=("GradDiff" "PB_J" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR")

CONDA_INIT="source /home/yuval.mandel/miniconda3/etc/profile.d/conda.sh && conda activate unlearning"
HF_EXPORT='export HF_HOME=$HOME/.cache/huggingface; export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)'

# ── Helper: submit a single extract job ────────────────────────────────────────
submit_extract() {
    local model_id="$1"
    local dep_flag="${2:-}"       # e.g. "--dependency=afterok:12345"
    local name="${model_id//-/_}" # job name safe

    sbatch --parsable \
      --job-name="io_ext_${name}" \
      --output="${LOGS}/ext_${name}_%j.out" \
      --error="${LOGS}/ext_${name}_%j.err" \
      --time=02:00:00 \
      --mem=32G \
      --cpus-per-task=4 \
      --gres=gpu:L40:1 \
      --partition=public \
      --requeue \
      ${dep_flag} \
      --wrap="set -e
${CONDA_INIT}
${HF_EXPORT}
cd ${REPO}
echo '=== Extract: ${model_id} ==='
${PYTHON} ${SCRIPT} --stage extract --model_id ${model_id} --domains bio
echo '=== Done: ${model_id} ==='"
}

# ── Helper: submit a single probe job ──────────────────────────────────────────
submit_probe() {
    local model_id="$1"
    local dep_jid="$2"            # afterok dependency JID
    local name="${model_id//-/_}"

    sbatch --parsable \
      --job-name="io_prb_${name}" \
      --output="${LOGS}/prb_${name}_%j.out" \
      --error="${LOGS}/prb_${name}_%j.err" \
      --time=00:30:00 \
      --mem=32G \
      --cpus-per-task=4 \
      --partition=public \
      --dependency=afterok:${dep_jid} \
      --wrap="set -e
${CONDA_INIT}
${HF_EXPORT}
cd ${REPO}
echo '=== Probe: ${model_id} ==='
${PYTHON} ${SCRIPT} --stage probe --model_id ${model_id} ${PROBE_FLAGS}
echo '=== Done: ${model_id} ==='"
}

# ── Helper: submit ck1-7 extract array for a method ────────────────────────────
submit_ck17_extract_array() {
    local method="$1"
    local dep_jid="$2"   # wait until ck8 extract is done (GPU slot freed)
    local name="${method//-/_}"

    # Build model IDs for ck1-7
    local model_list=""
    for ck in $(seq 1 7); do
        model_list="${model_list} ${method}_ck${ck}"
    done
    model_list="${model_list# }"   # trim leading space

    # Map array index 0..6 → checkpoint 1..7
    sbatch --parsable \
      --job-name="io_ext_${name}_17" \
      --array=0-6 \
      --output="${LOGS}/ext_${name}_%a_%j.out" \
      --error="${LOGS}/ext_${name}_%a_%j.err" \
      --time=02:00:00 \
      --mem=32G \
      --cpus-per-task=4 \
      --gres=gpu:L40:1 \
      --partition=public \
      --requeue \
      --dependency=afterok:${dep_jid} \
      --wrap="set -e
${CONDA_INIT}
${HF_EXPORT}
cd ${REPO}
CK=\$((SLURM_ARRAY_TASK_ID + 1))
MODEL_ID=\"${method}_ck\${CK}\"
echo \"=== Extract: \${MODEL_ID} (task \${SLURM_ARRAY_TASK_ID}) ===\"
${PYTHON} ${SCRIPT} --stage extract --model_id \"\${MODEL_ID}\" --domains bio
echo \"=== Done: \${MODEL_ID} ===\""
}

# ── Helper: submit ck1-7 probe array for a method ──────────────────────────────
submit_ck17_probe_array() {
    local method="$1"
    local dep_jid="$2"   # wait until ck1-7 extract array is done
    local name="${method//-/_}"

    sbatch --parsable \
      --job-name="io_prb_${name}_17" \
      --array=0-6 \
      --output="${LOGS}/prb_${name}_%a_%j.out" \
      --error="${LOGS}/prb_${name}_%a_%j.err" \
      --time=00:30:00 \
      --mem=32G \
      --cpus-per-task=4 \
      --partition=public \
      --dependency=afterok:${dep_jid} \
      --wrap="set -e
${CONDA_INIT}
${HF_EXPORT}
cd ${REPO}
CK=\$((SLURM_ARRAY_TASK_ID + 1))
MODEL_ID=\"${method}_ck\${CK}\"
echo \"=== Probe: \${MODEL_ID} (task \${SLURM_ARRAY_TASK_ID}) ===\"
${PYTHON} ${SCRIPT} --stage probe --model_id \"\${MODEL_ID}\" ${PROBE_FLAGS}
echo \"=== Done: \${MODEL_ID} ===\""
}


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — base
# ══════════════════════════════════════════════════════════════════════════════

echo "Submitting Tier 1: base extract → base probe"

JID_BASE_EXT=$(submit_extract "base")
echo "  base extract:  ${JID_BASE_EXT}"

JID_BASE_PRB=$(submit_probe "base" "${JID_BASE_EXT}")
echo "  base probe:    ${JID_BASE_PRB}"


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — ck8 for all 8 methods (extract in parallel after base, probe after extract)
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "Submitting Tier 2: ck8 extracts (after base extract) + ck8 probes"

declare -A JID_CK8_EXT
declare -A JID_CK8_PRB

for method in "${METHODS_ORDER[@]}"; do
    jid_ext=$(submit_extract "${method}_ck8" "--dependency=afterok:${JID_BASE_EXT}")
    JID_CK8_EXT[$method]=$jid_ext

    jid_prb=$(submit_probe "${method}_ck8" "${jid_ext}")
    JID_CK8_PRB[$method]=$jid_prb

    echo "  ${method} ck8: ext=${jid_ext}  prb=${jid_prb}"
done


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 — ck1-7 for each method, in priority order
#           extract starts after that method's ck8 extract (GPU slot freed)
#           probe starts after extract array completes
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "Submitting Tier 3: ck1-7 arrays (method by method)"

for method in "${METHODS_ORDER[@]}"; do
    jid_ck8_ext="${JID_CK8_EXT[$method]}"

    jid_ext_arr=$(submit_ck17_extract_array "${method}" "${jid_ck8_ext}")
    jid_prb_arr=$(submit_ck17_probe_array   "${method}" "${jid_ext_arr}")

    echo "  ${method} ck1-7: ext_array=${jid_ext_arr}  prb_array=${jid_prb_arr}"
done

echo ""
echo "All jobs submitted."
echo "Monitor: squeue -u yuval.mandel"
echo "Logs:    ${LOGS}/"
