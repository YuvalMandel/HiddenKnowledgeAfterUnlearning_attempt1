#!/bin/bash
#SBATCH --job-name=probe_delta
#SBATCH --output=logs/probe_delta_%j.out
#SBATCH --error=logs/probe_delta_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=public

set -e
cd /home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1

source /home/yuval.mandel/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p plots logs

echo "=== Plot probe delta ==="
python3 method_selection/plot_probe_delta.py --out_dir plots --clf LR

echo "=== Plot suppressed/emerged ==="
python3 method_selection/plot_suppressed_emerged.py --out_dir plots

echo "DONE"
