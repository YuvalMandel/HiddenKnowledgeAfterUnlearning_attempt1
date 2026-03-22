#!/bin/bash
#SBATCH --job-name=pca_viz
#SBATCH --output=logs/pca_viz_%j.out
#SBATCH --error=logs/pca_viz_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=public
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
# No GPU needed.

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p logs

python pca_probe_viz_report.py --out_dir pca_viz "$@"

echo "Job finished at $(date)"
