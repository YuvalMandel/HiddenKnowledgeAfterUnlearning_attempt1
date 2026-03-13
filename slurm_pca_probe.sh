#!/bin/bash
#SBATCH --job-name=pca_probe
#SBATCH --output=logs/pca_probe_%j.out
#SBATCH --error=logs/pca_probe_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=public
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ============================================================
# PCA Probe Visualization — pca_probe_viz_band_report.py
#
# Usage (all args are optional; defaults match run_pca_probe_band_report.cmd):
#
#   sbatch slurm_pca_probe.sh
#   sbatch slurm_pca_probe.sh --post_hs checkpoints/RMU_hs_train.npy
#   sbatch slurm_pca_probe.sh --layers 20 --pca_components 20 --post_hs checkpoints/GradDiff_hs_train.npy
#
# All extra arguments are forwarded verbatim to the Python script.
# ============================================================

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p logs

python pca_probe_viz_band_report.py "$@"

echo "Job finished at $(date)"
