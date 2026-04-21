#!/bin/bash
#SBATCH --job-name=hk_re_dr
#SBATCH --output=logs/re_dr_%a_%j.out
#SBATCH --error=logs/re_dr_%a_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=public
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# No GPU needed — sklearn probes on pre-computed .npy hidden states.
# Array: 8 unlearning methods (0-7).
#SBATCH --array=0-7

echo "Job started at $(date)"
echo "Node: $(hostname)  array_task: ${SLURM_ARRAY_TASK_ID}"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p logs method_selection_out/re_dr

# Map array index → method name (matches safe_name() output in hidden_knowledge_after_unlearning.py)
METHODS=("GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB_J")
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"

# Use the sanitised name for PB&J (pipeline saves as PB_J_hs_*.npy)
echo ""
echo "Running RE/DR analysis for method: ${METHOD}"
python method_selection/re_dr_analysis.py \
    --method "${METHOD}" \
    --data_dir checkpoints \
    --labels_csv checkpoints/bio_labels.csv \
    --base_prefix base_hs \
    --method_suffix _hs \
    --out_root method_selection_out/re_dr \
    --eval_split test \
    --train_scheme full \
    --train_window_id full \
    --band_mode concat \
    --re_metric auc \
    --subset_mode none \
    --use_probe_cache \
    --save_band_prediction_csvs \
    --save_json

echo "Job finished at $(date)"
