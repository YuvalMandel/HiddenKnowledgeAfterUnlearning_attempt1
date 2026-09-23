#!/bin/bash
#SBATCH --job-name=chem_cross
#SBATCH --partition=public
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=inside_out_logs/chemx_%A_%a.out
#SBATCH --error=inside_out_logs/chemx_%A_%a.err
#
# Chem-fitted probes read into the two forget domains, DARWIN (CPU).
#
# WMDP-Chem is the one WMDP domain RMU did NOT unlearn (their S4: "we focus on
# unlearning hazardous knowledge in biosecurity and cybersecurity, but not in
# chemistry"; Table 2 chem moves 0.0/-0.5/-3.2 pts). A probe fitted there has
# therefore never seen anything the method was trained to remove.
#
# One array task per (model, direction) so both directions run at once, each
# with its own 16-way layer sweep.
set -euo pipefail

PYTHON=/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3
REPO=/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1

MODELS=(zephyr_base zephyr_base zephyr_rmu zephyr_rmu)
PAIRS=(chem:bio     chem:cyber   chem:bio   chem:cyber)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
PAIR=${PAIRS[$SLURM_ARRAY_TASK_ID]}
SUFFIX="${PAIR/:/_to_}"

cd "$REPO"
echo "host=$(hostname) model=$MODEL pair=$PAIR start=$(date -Is)"
"$PYTHON" inside_out_knowledge.py --stage cross_domain --model_id "$MODEL" \
    --pairs "$PAIR" --out_suffix "$SUFFIX" --n_jobs 16
echo "model=$MODEL pair=$PAIR end=$(date -Is)"
ls -l "inside_out_out/$MODEL/k_scores_$SUFFIX.parquet"
