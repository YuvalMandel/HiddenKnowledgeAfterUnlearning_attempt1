#!/bin/bash
#SBATCH --job-name=hk_gen
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-64

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate insideout_unlearn_4

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)

mkdir -p logs

# Resolve model_id from task index using the script's own list
MODEL_ID=$(python inside_out_knowledge.py --list_models \
           | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

echo "Model ID: ${MODEL_ID}"

python inside_out_knowledge.py \
    --stage gen \
    --model_id "${MODEL_ID}" \
    --domains bio \
    --max_new_tokens 64

echo "Job finished at $(date)"
