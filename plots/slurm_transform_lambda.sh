#!/bin/bash
#SBATCH --job-name=tlambda
#SBATCH --output=logs/tlambda_%A_%a.out
#SBATCH --error=logs/tlambda_%A_%a.err
#SBATCH --time=00:40:00
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-10
echo "start $(date) on $(hostname) task $SLURM_ARRAY_TASK_ID"; nvidia-smi
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p logs
cd $HOME/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
METHOD=${METHOD:-RepNoise}
GAMMAS=(0.003 0.01 0.03 0.1 0.3 1.0 3.0 10.0 100.0 1000.0 inf)
G=${GAMMAS[$SLURM_ARRAY_TASK_ID]}
echo "##### METHOD=$METHOD gamma=$G #####"
python plots/causal_recover_transform_lambda.py $METHOD $G
echo "done $(date)"
