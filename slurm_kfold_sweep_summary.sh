#!/bin/bash
#SBATCH --job-name=hk_kfold_sweep_sum
#SBATCH --output=logs/kfold_sweep_summary_%j.out
#SBATCH --error=logs/kfold_sweep_summary_%j.err
#SBATCH --time=00:20:00
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
echo "Running kfold_sweep_summary..."
python kfold_probe.py --stage kfold_sweep_summary

echo ""
echo "Running kfold_sweep_tables (wide-format pivot for plot scripts)..."
python kfold_probe.py --stage kfold_sweep_tables

echo "Job finished at $(date)"
