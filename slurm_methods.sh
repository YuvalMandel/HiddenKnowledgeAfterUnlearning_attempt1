#!/bin/bash
#SBATCH --job-name=hk_method
#SBATCH --output=logs/method_%a_%j.out
#SBATCH --error=logs/method_%a_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7

echo "Job started at $(date)"
echo "Running on node: $(hostname), array task: ${SLURM_ARRAY_TASK_ID}"
echo "GPU info:"
nvidia-smi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)

mkdir -p logs

# Map array index to method name.
# Order must match UNLEARNED_MODELS in hidden_knowledge_after_unlearning.py.
METHODS=("GradDiff" "RMU" "RMU-LAT" "RepNoise" "ELM" "RR" "TAR" "PB&J")
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "Processing method: ${METHOD}"
python hidden_knowledge_after_unlearning.py --stage method --method "${METHOD}"

echo "Job finished at $(date)"
