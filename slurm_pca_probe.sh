#!/bin/bash
#SBATCH --job-name=pca_probe
#SBATCH --output=logs/pca_probe_%j.out
#SBATCH --error=logs/pca_probe_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=public
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ============================================================
# PCA Probe Visualization — pca_probe_viz_band_report.py
#
# Single run (default: PB_J ck1, layers 12-22, pca 40):
#   sbatch slurm_pca_probe.sh
#
# Explicit single run:
#   sbatch slurm_pca_probe.sh --post_hs checkpoints/sweep_RMU/ck3/hs_train.npy
#
# Multiple methods, all checkpoints, 8 parallel workers:
#   sbatch slurm_pca_probe.sh --methods GradDiff RMU PB_J --checkpoints all --workers 8
#
# All methods, checkpoints 1-4, 8 workers:
#   sbatch slurm_pca_probe.sh --methods all --checkpoints 1-4 --workers 8
#
# All args are forwarded verbatim to the Python script.
# Increase --cpus-per-task to match --workers for full parallelism.
# ============================================================

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "CPUs available: $(nproc)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p logs

# Default to --workers matching allocated CPUs if not already in $@
EXTRA=""
if [[ "$*" != *"--workers"* ]]; then
    EXTRA="--workers $(nproc)"
fi

python pca_probe_viz_band_report.py $EXTRA "$@"

echo "Job finished at $(date)"
