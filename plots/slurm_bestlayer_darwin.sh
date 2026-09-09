#!/bin/bash
#SBATCH --job-name=bestlayer
#SBATCH --partition=public
#SBATCH --array=0-8
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs_bestlayer/darwin_%A_%a.log
# CPU-only, so this belongs on DARWIN (ssh yuval.mandel@132.68.38.100), not
# Newton -- Newton's public partition rejects jobs without --gres=gpu.
# Same home directory, same conda env, 515 GB nodes.
cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
METHODS=(base GradDiff RMU RMU-LAT RepNoise ELM RR TAR PB_J)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
echo "=== $M on $(hostname) at $(date)"
~/miniconda3/envs/insideout_unlearn_4/bin/python plots/bestlayer_skip.py "$M" "$@"
echo "=== $M done $(date)"
