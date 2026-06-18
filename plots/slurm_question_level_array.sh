#!/bin/bash
#SBATCH --job-name=qlvl_arr
#SBATCH --output=qlvl_%A_%a.out
#SBATCH --error=qlvl_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-7
cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
M=(GradDiff PB_J RMU RMU-LAT RepNoise ELM RR TAR)
~/miniconda3/envs/unlearning/bin/python plots/activation_question_level.py ${M[$SLURM_ARRAY_TASK_ID]}
