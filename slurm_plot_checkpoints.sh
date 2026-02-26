#!/bin/bash
#SBATCH --job-name=hk_plot_ck
#SBATCH --output=logs/plot_ck_%a_%j.out
#SBATCH --error=logs/plot_ck_%a_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=public
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-7

# Task layout:  one method per array task
#   task 0 : GradDiff
#   task 1 : RMU
#   task 2 : RMU-LAT
#   task 3 : RepNoise
#   task 4 : ELM
#   task 5 : RR
#   task 6 : TAR
#   task 7 : PB&J
#
# Args are passed via environment variables set by submit_plot_checkpoints.sh:
#   PLOT_PROBE_SOURCE   (default: method)
#   PLOT_CLF            (default: all)
#   PLOT_METRIC         (default: all)
#   PLOT_OUT            (default: layer_accuracy.png)
#   PLOT_CHECKPOINT_DIR (default: checkpoints)
#   CONDA_ENV           (default: htm_keyboard_1)

echo "Job started at $(date)"
echo "Running on node: $(hostname), array task: ${SLURM_ARRAY_TASK_ID}"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-htm_keyboard_1}"

mkdir -p logs

SWEEP_METHODS=("GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB&J")
METHOD="${SWEEP_METHODS[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "Method: ${METHOD}"

# Build method-specific output filename: layer_accuracy.png -> layer_accuracy_GradDiff.png
# Mirrors plot_layer_accuracy._safe_name(): replace & / space with _
BASE_OUT="${PLOT_OUT:-layer_accuracy.png}"
SAFE_METHOD="${METHOD//&/_}"
SAFE_METHOD="${SAFE_METHOD// /_}"
SAFE_METHOD="${SAFE_METHOD//\//_}"
OUT_STEM="${BASE_OUT%.*}"
OUT_EXT="${BASE_OUT##*.}"
METHOD_OUT="${OUT_STEM}_${SAFE_METHOD}.${OUT_EXT}"

echo "Output: ${METHOD_OUT}"

# Build optional argument list from environment
EXTRA_ARGS="--out ${METHOD_OUT}"
[[ -n "${PLOT_PROBE_SOURCE}"   ]] && EXTRA_ARGS="$EXTRA_ARGS --probe_source ${PLOT_PROBE_SOURCE}"
[[ -n "${PLOT_CLF}"            ]] && EXTRA_ARGS="$EXTRA_ARGS --clf ${PLOT_CLF}"
[[ -n "${PLOT_METRIC}"         ]] && EXTRA_ARGS="$EXTRA_ARGS --metric ${PLOT_METRIC}"
[[ -n "${PLOT_CHECKPOINT_DIR}" ]] && EXTRA_ARGS="$EXTRA_ARGS --checkpoint_dir ${PLOT_CHECKPOINT_DIR}"

python plot_layer_accuracy.py \
    --mode checkpoints \
    --method "${METHOD}" \
    $EXTRA_ARGS

echo "Job finished at $(date)"
