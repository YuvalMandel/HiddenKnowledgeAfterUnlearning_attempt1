#!/bin/bash
#SBATCH --job-name=dual_conf_plot
#SBATCH --output=logs/dual_conf_plot_%j.out
#SBATCH --error=logs/dual_conf_plot_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=public
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# No GPU needed.

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p logs
mkdir -p dual_confidence_outputs

python hidden_knowledge_dual_confidence_package/generate_hidden_knowledge_dual_confidence_figures.py \
    --input-dir dual_confidence_inputs \
    --output-dir dual_confidence_outputs

echo "Job finished at $(date)"
