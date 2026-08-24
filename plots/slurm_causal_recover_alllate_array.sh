#!/bin/bash
#SBATCH --job-name=recover_alllate
#SBATCH --output=logs/recover_alllate_%A_%a.out
#SBATCH --error=logs/recover_alllate_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-15
# Answers "why does the fixed injection grid stop at layer 15 of 32?".
#   all  : inject at every layer 1..32 -- no layer selection to defend.
#   late : {18,21,24,27,30}, the fixed grid mirrored into the second half, which
#          separates "more injection sites" from "later injection sites".
# Both retain the matched-norm random control, so steered-vs-random stays valid
# even if absolute K_ext degrades under a heavier perturbation.
echo "start $(date) on $(hostname) task $SLURM_ARRAY_TASK_ID"; nvidia-smi
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p logs
cd $HOME/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
METHODS=(GradDiff RMU RMU-LAT RepNoise ELM RR TAR PB_J)
M=${METHODS[$(( SLURM_ARRAY_TASK_ID % 8 ))]}
if [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then MODE=all; else MODE=late; fi
echo "##### $M $MODE #####"
python plots/causal_recover.py $M $MODE 5 correct
echo "done $(date)"
