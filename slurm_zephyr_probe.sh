#!/bin/bash
#SBATCH --job-name=zephyr_probe
#SBATCH --partition=public
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
# 24G not 64G: measured peak is ~13.3 GB per probe (5 concurrent on socrates,
# 66.7 GB total), so 64 GB was ~5x over and memory, not CPU, was capping
# concurrency -- 380 of 515 GB was *requested* while ~135 GB sat idle.
#SBATCH --time=03:00:00
#SBATCH --output=inside_out_logs/zephyr_probe_%A_%a.out
#SBATCH --error=inside_out_logs/zephyr_probe_%A_%a.err
#
# Probe stage for the Zephyr pair. DARWIN, not Newton: this is CPU-only and
# Newton rejects CPU-only jobs; the Newton login node also OOM-kills it
# (bio_hs.npy is 1.4 GB and the login cap is ~3 GB). Darwin shares the home
# directory, so the hidden states written by the Newton GPU job are already here.
#
# Submit:  ssh yuval.mandel@132.68.38.100
#          cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
#          /usr/local/bin/sbatch --array=0-1 slurm_zephyr_probe.sh
set -euo pipefail
REPO=/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
PYTHON=/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3
# Index must match --array. A previous version of this line was written by a
# heredoc that mangled its line continuations into a literal backslash-n, which
# bash split into a bare array element "n" at 4 and 9, shifting every model
# after it by one.
MODELS=(zephyr_base zephyr_rmu mixtral_base mixtral_rmu
        zephyr_graddiff zephyr_graddiff_sam zephyr_npo zephyr_npo_cr
        zephyr_npo_gp zephyr_npo_rs zephyr_npo_sam zephyr_npo_wa
        zephyr_simnpo
        l3_dpo l3_graddiff l3_idkap l3_ilurmu l3_npoilu
        l3_nposam l3_npo l3_simnpo l3_undial)
#        0            1           2             3
#        4                5                    6           7
#        8              9              10              11
#        12
#        13     14          15        16        17
#        18        19     20         21

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"; mkdir -p inside_out_logs
echo "host=$(hostname) model=$MODEL start=$(date -Is)"
"$PYTHON" inside_out_knowledge.py --stage probe --model_id "$MODEL" --domains bio \
    --clfs LR --lcs best_layer --n_jobs 16 --no_cross_probe
echo "model=$MODEL end=$(date -Is)"
