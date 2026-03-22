#!/bin/bash
#SBATCH --job-name=dual_conf_gen
#SBATCH --output=logs/dual_conf_gen_%A_%a.out
#SBATCH --error=logs/dual_conf_gen_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=public
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-8
# No GPU needed.

METHODS=("Base" "GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB&J")
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Method: ${METHOD}"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p logs
mkdir -p dual_confidence_inputs

python generate_dual_confidence_inputs.py \
    --method "${METHOD}" \
    --out_dir dual_confidence_inputs

echo "Job finished at $(date)"
