#!/bin/bash
# submit_inside_out.sh — Submit the full Inside-Out pipeline on Newton.
#
# Stages:
#   1. extract (GPU array, 65 jobs)  — one job per model
#   2. probe   (CPU array, 65 jobs)  — one job per model, after all extracts done
#   3. aggregate (CPU, 1 job)        — after all probes done
#
# Usage:
#   bash submit_inside_out.sh              # submit all three stages
#   bash submit_inside_out.sh extract      # submit only extract
#   bash submit_inside_out.sh probe <jid>  # submit probe depending on job <jid>

set -e

REPO="/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1"
PYTHON="/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3"
SCRIPT="$REPO/inside_out_knowledge.py"
LOGS="$REPO/inside_out_logs"
mkdir -p "$LOGS"

N_MODELS=65   # 1 base + 8 methods * 8 checkpoints
ARRAY_END=$((N_MODELS - 1))

# Helper: get model_id for a given SLURM array task index
# Used inside job scripts via: MODEL_ID=$(get_model_id $SLURM_ARRAY_TASK_ID)
GET_MODEL_ID_CMD="$PYTHON -c \"
import sys
methods=['GradDiff','RMU','RMU-LAT','RepNoise','ELM','RR','TAR','PB_J']
ids=['base']+[m+'_ck'+str(ck) for m in methods for ck in range(1,9)]
print(ids[int(sys.argv[1])])
\" \$SLURM_ARRAY_TASK_ID"

# ── Stage 1: Extract (GPU) ─────────────────────────────────────────────────

JOB_EXTRACT=$(sbatch --parsable \
  --job-name=io_extract \
  --array=1-${ARRAY_END}%5 \
  --output="${LOGS}/extract_%A_%a.out" \
  --error="${LOGS}/extract_%A_%a.err" \
  --time=04:00:00 \
  --mem=32G \
  --cpus-per-task=4 \
  --gres=gpu:L40:1 \
  --partition=public \
  --requeue \
  << 'SBATCH_EXTRACT'
#!/bin/bash
source /home/yuval.mandel/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)

REPO="/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1"
PYTHON="/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3"
SCRIPT="$REPO/inside_out_knowledge.py"

MODEL_ID=$($PYTHON -c "
import sys
methods=['GradDiff','RMU','RMU-LAT','RepNoise','ELM','RR','TAR','PB_J']
ids=['base']+[m+'_ck'+str(ck) for m in methods for ck in range(1,9)]
print(ids[int(sys.argv[1])])
" $SLURM_ARRAY_TASK_ID)

echo "=== Extract: $MODEL_ID (task $SLURM_ARRAY_TASK_ID) ==="
cd "$REPO"
$PYTHON "$SCRIPT" --stage extract --model_id "$MODEL_ID"
echo "=== Done: $MODEL_ID ==="
SBATCH_EXTRACT
)

echo "Submitted extract array: ${JOB_EXTRACT}  (jobs 1-${ARRAY_END}, base already done)"

if [[ "${1:-}" == "extract" ]]; then
  echo "extract-only mode — done."
  exit 0
fi

# ── Stage 2: Probe (CPU) ───────────────────────────────────────────────────

JOB_PROBE=$(sbatch --parsable \
  --job-name=io_probe \
  --array=0-${ARRAY_END}%20 \
  --output="${LOGS}/probe_%A_%a.out" \
  --error="${LOGS}/probe_%A_%a.err" \
  --time=06:00:00 \
  --mem=64G \
  --cpus-per-task=8 \
  --partition=public \
  --dependency=afterok:${JOB_EXTRACT} \
  << 'SBATCH_PROBE'
#!/bin/bash
source /home/yuval.mandel/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)

REPO="/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1"
PYTHON="/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3"
SCRIPT="$REPO/inside_out_knowledge.py"

MODEL_ID=$($PYTHON -c "
import sys
methods=['GradDiff','RMU','RMU-LAT','RepNoise','ELM','RR','TAR','PB_J']
ids=['base']+[m+'_ck'+str(ck) for m in methods for ck in range(1,9)]
print(ids[int(sys.argv[1])])
" $SLURM_ARRAY_TASK_ID)

echo "=== Probe: $MODEL_ID (task $SLURM_ARRAY_TASK_ID) ==="
cd "$REPO"
$PYTHON "$SCRIPT" --stage probe --model_id "$MODEL_ID"
echo "=== Done: $MODEL_ID ==="
SBATCH_PROBE
)

echo "Submitted probe array:   ${JOB_PROBE}  (depends on ${JOB_EXTRACT})"

# ── Stage 3: Aggregate (CPU) ───────────────────────────────────────────────

JOB_AGG=$(sbatch --parsable \
  --job-name=io_aggregate \
  --output="${LOGS}/aggregate_%j.out" \
  --error="${LOGS}/aggregate_%j.err" \
  --time=01:00:00 \
  --mem=32G \
  --cpus-per-task=4 \
  --partition=public \
  --dependency=afterok:${JOB_PROBE} \
  << 'SBATCH_AGG'
#!/bin/bash
source /home/yuval.mandel/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)

REPO="/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1"
PYTHON="/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3"
SCRIPT="$REPO/inside_out_knowledge.py"

echo "=== Aggregate ==="
cd "$REPO"
$PYTHON "$SCRIPT" --stage aggregate
echo "=== Done ==="
SBATCH_AGG
)

echo "Submitted aggregate:     ${JOB_AGG}  (depends on ${JOB_PROBE})"
echo ""
echo "Pipeline: ${JOB_EXTRACT} → ${JOB_PROBE} → ${JOB_AGG}"
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOGS}/"
