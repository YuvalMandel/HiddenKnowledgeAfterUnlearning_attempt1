#!/bin/bash
#SBATCH --job-name=hk_re_dr_summary
#SBATCH --output=logs/re_dr_summary_%j.out
#SBATCH --error=logs/re_dr_summary_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=public
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
# CPU only — aggregates band prediction CSVs into authoritative_row_measurements.csv.

echo "Job started at $(date)"
echo "Node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p method_selection_out

# Aggregate all 8 methods' band prediction CSVs → authoritative_row_measurements.csv
# Uses the late band (most informative for knowledge erasure)
echo ""
echo "Building authoritative_row_measurements.csv (late band) ..."
python method_selection/build_authoritative_measurements.py \
    --band late \
    --eval_split test \
    --train_scheme full \
    --train_window_id full \
    --re_dr_root method_selection_out/re_dr \
    --labels_csv checkpoints/bio_labels.csv \
    --out_csv method_selection_out/authoritative_row_measurements.csv

echo ""
echo "Job finished at $(date)"
