#!/bin/bash
#SBATCH --job-name=hk_kfold_summary
#SBATCH --output=logs/kfold_summary_%j.out
#SBATCH --error=logs/kfold_summary_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=public
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
# No GPU needed.

echo "Job started at $(date)"
echo "Node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p logs data

echo ""
echo "Running kfold summary..."
python kfold_probe.py --stage kfold_summary

echo "Job finished at $(date)"
