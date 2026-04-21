#!/bin/bash
#SBATCH --job-name=hk_method_id
#SBATCH --output=logs/method_id_%j.out
#SBATCH --error=logs/method_id_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=public
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# CPU only — loads pre-computed .npy hidden states, trains small classifiers, plots.

echo "Job started at $(date)"
echo "Node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

METHODS="GradDiff,RMU,RMU-LAT,RepNoise,ELM,RR,TAR,PB_J"
DATA_DIR=checkpoints
LABELS_CSV=checkpoints/bio_labels.csv
OUT_ROOT=method_selection_out/method_id

mkdir -p logs "${OUT_ROOT}"

# ── Run classification for each band ─────────────────────────────────────────
for BAND in early mid late; do
    echo ""
    echo "Running method identification: band=${BAND} ..."
    python method_selection/raw_hiddenstate_method_classification_safe.py \
        --data_dir      "${DATA_DIR}" \
        --labels_csv    "${LABELS_CSV}" \
        --methods       "${METHODS}" \
        --method_suffix _hs \
        --band          "${BAND}" \
        --train_split   val \
        --test_splits   test \
        --max_rows_per_method 100 \
        --models        logistic,rf,nn \
        --pca_components 0 \
        --out_dir       "${OUT_ROOT}/${BAND}"
done

# ── Plot ──────────────────────────────────────────────────────────────────────
echo ""
echo "Generating method identification bar plot ..."
python method_selection/plot_method_identification.py \
    --results_dir "${OUT_ROOT}" \
    --out_dir     plots

echo ""
echo "Job finished at $(date)"
echo "Plot: plots/method_identification_by_band.png"
echo "CSVs: ${OUT_ROOT}/{early,mid,late}/method_classification_summary.csv"
