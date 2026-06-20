#!/bin/bash
#SBATCH --job-name=distract_agg
#SBATCH --output=distract_agg_%j.out
#SBATCH --error=distract_agg_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
~/miniconda3/envs/unlearning/bin/python plots/aggregate_distractor_structure.py
