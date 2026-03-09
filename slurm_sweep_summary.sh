#!/bin/bash
#SBATCH --job-name=hk_sweep_sum
#SBATCH --output=logs/sweep_sum_%a_%j.out
#SBATCH --error=logs/sweep_sum_%a_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=public
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7

# Runs sweep_summary (print table + save CSV) then immediately generates the
# per-method layer-accuracy plot.  No GPU needed.
#
# Plot behaviour is controlled by env vars (set by submit_sweep.sh or by
# exporting manually before calling sbatch):
#
#   PLOT_PLOT_TYPE      line | heatmap          (default: line)
#   PLOT_METRIC         accuracy | f1 | auc …   (default: all metrics)
#   PLOT_CLF            LR | RF | AdaBoost       (default: all)
#   PLOT_PROBE_SOURCE   method | base            (default: method)
#   PLOT_DATASET        bio | cyber              (default: bio)
#   PLOT_CYBER_SUBSET   og|pattern|gibberish|both (default: none)
#   PLOT_OUT            filename template         (default: auto-generated)
#   PLOT_CHECKPOINT_DIR path to checkpoints dir  (default: checkpoints)
#   CONDA_ENV           conda environment name   (default: unlearning)

echo "Job started at $(date)"
echo "Running on node: $(hostname), array task: ${SLURM_ARRAY_TASK_ID}"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-unlearning}"

mkdir -p logs

SWEEP_METHODS=("GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB&J")
METHOD="${SWEEP_METHODS[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "=== Sweep summary for: ${METHOD} ==="

# ── Step 1: generate table + CSV ─────────────────────────────────────────────
CKPT_DIR="${PLOT_CHECKPOINT_DIR:-checkpoints}"
python hidden_knowledge_after_unlearning.py \
    --stage sweep_summary \
    --method "${METHOD}"

# ── Step 2: generate plot ─────────────────────────────────────────────────────
echo ""
echo "=== Generating plot for: ${METHOD} ==="

# Build --out argument: if PLOT_OUT is set use it as a template (append method
# name to the stem); otherwise omit it so the script auto-generates a name.
PLOT_ARGS="--mode checkpoints --method ${METHOD}"
PLOT_ARGS="${PLOT_ARGS} --checkpoint_dir ${CKPT_DIR}"

[[ -n "${PLOT_PLOT_TYPE}"    ]] && PLOT_ARGS="${PLOT_ARGS} --plot_type ${PLOT_PLOT_TYPE}"
[[ -n "${PLOT_METRIC}"       ]] && PLOT_ARGS="${PLOT_ARGS} --metric ${PLOT_METRIC}"
[[ -n "${PLOT_CLF}"          ]] && PLOT_ARGS="${PLOT_ARGS} --clf ${PLOT_CLF}"
[[ -n "${PLOT_PROBE_SOURCE}" ]] && PLOT_ARGS="${PLOT_ARGS} --probe_source ${PLOT_PROBE_SOURCE}"
[[ -n "${PLOT_DATASET}"      ]] && PLOT_ARGS="${PLOT_ARGS} --dataset ${PLOT_DATASET}"
[[ -n "${PLOT_CYBER_SUBSET}" ]] && PLOT_ARGS="${PLOT_ARGS} --cyber_subset ${PLOT_CYBER_SUBSET}"

if [[ -n "${PLOT_OUT}" ]]; then
    SAFE_METHOD="${METHOD//&/_}"
    SAFE_METHOD="${SAFE_METHOD// /_}"
    SAFE_METHOD="${SAFE_METHOD//\//_}"
    OUT_STEM="${PLOT_OUT%.*}"
    OUT_EXT="${PLOT_OUT##*.}"
    METHOD_OUT="${OUT_STEM}_${SAFE_METHOD}.${OUT_EXT}"
    PLOT_ARGS="${PLOT_ARGS} --out ${METHOD_OUT}"
    echo "Plot output: ${METHOD_OUT}"
else
    echo "Plot output: auto-generated from params"
fi

python plot_layer_accuracy.py ${PLOT_ARGS}

echo "Job finished at $(date)"
