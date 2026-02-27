#!/bin/bash
# submit_plot_checkpoints.sh
#
# Submit 8 parallel SLURM tasks, one per unlearning method, each producing
# a layer-accuracy-over-checkpoints PNG.  Each task runs in its own process
# so memory is fully isolated between methods.
#
# Usage:
#   bash submit_plot_checkpoints.sh
#   bash submit_plot_checkpoints.sh --probe_source base
#   bash submit_plot_checkpoints.sh --clf LR --metric true_accuracy
#   bash submit_plot_checkpoints.sh --out plots/ck.png --checkpoint_dir /path/to/ckpts
#
# Output files (example with default --out layer_accuracy.png):
#   layer_accuracy_GradDiff.png
#   layer_accuracy_RMU.png
#   ... (one per method)

set -e
mkdir -p logs

# ── Defaults ──────────────────────────────────────────────────────────────────
export PLOT_PROBE_SOURCE="method"
export PLOT_CLF=""
export PLOT_METRIC=""
export PLOT_PLOT_TYPE=""
export PLOT_OUT=""
export PLOT_CHECKPOINT_DIR="checkpoints"
export CONDA_ENV="htm_keyboard_1"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --probe_source)   export PLOT_PROBE_SOURCE="$2";   shift ;;
        --clf)            export PLOT_CLF="$2";            shift ;;
        --metric)         export PLOT_METRIC="$2";         shift ;;
        --plot_type)      export PLOT_PLOT_TYPE="$2";      shift ;;
        --out)            export PLOT_OUT="$2";            shift ;;
        --checkpoint_dir) export PLOT_CHECKPOINT_DIR="$2"; shift ;;
        --env)            export CONDA_ENV="$2";           shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Submit ────────────────────────────────────────────────────────────────────
echo "Submitting checkpoint-plot jobs (8 methods in parallel) ..."
echo "  probe_source : ${PLOT_PROBE_SOURCE}"
echo "  clf          : ${PLOT_CLF:-all}"
echo "  metric       : ${PLOT_METRIC:-all}"
echo "  plot_type    : ${PLOT_PLOT_TYPE:-line}"
echo "  out template : ${PLOT_OUT:-auto}"
echo "  checkpoint_dir: ${PLOT_CHECKPOINT_DIR}"
echo "  conda env    : ${CONDA_ENV}"
echo ""

JOB=$(sbatch --parsable --export=ALL --array=0-7 slurm_plot_checkpoints.sh)
echo "  Submitted job array: ${JOB}"
echo "  Monitor : squeue -u \$USER"
echo "  Logs    : logs/plot_ck_<task>_${JOB}.out"
if [[ -n "${PLOT_OUT}" ]]; then
    OUT_STEM="${PLOT_OUT%.*}"
    OUT_EXT="${PLOT_OUT##*.}"
    echo "  Output  : ${OUT_STEM}_{method}.${OUT_EXT}  (one file per method)"
else
    echo "  Output  : auto-generated per method, e.g. line_checkpoints_GradDiff_all_metrics_all_clf_${PLOT_PROBE_SOURCE}.png"
fi
