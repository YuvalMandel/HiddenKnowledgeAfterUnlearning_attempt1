#!/bin/bash
#SBATCH --job-name=optml_l3
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-8%3
#SBATCH --output=inside_out_logs/optml_l3_%A_%a.out
#SBATCH --error=inside_out_logs/optml_l3_%A_%a.err
#
# OPTML-Group's WMDP suite on Llama-3-8B-Instruct -- nine unlearning methods on
# THE SAME BASE THIS PAPER ALREADY PUBLISHES, so inside_out_out/base is the
# shared reference and there is no base run to do.
#
# Verified drop-in before launching (plots/check_optml_llama_tokens.py):
#   * all 9 give legacy True/False ids 3082/3641, same as our base
#   * all 9 render byte-identical prompts (99 tokens)
#   * 7 append a [pad] token at 128256, strictly after the stock vocabulary,
#     so no existing id shifts; LEGACY_TF_MODELS pins pad back to eos anyway
#   * 32 blocks -> 33 hidden states, hidden 4096 -- N_LAYERS unchanged
#
# Each task: download -> extract -> DELETE THE WEIGHTS. We keep the hidden
# states (1.4 GB); the weights (~16 GB) are re-downloadable. %3 caps peak disk
# at ~48 GB against the ~131 GB left under the quota.
#
# Extraction also writes bio_ext_alt.npy: the SAME logits scored with the
# derived ids 2575/4139. Free here (the logits are in hand) and exact, so if we
# ever switch rules we do not have to reconstruct it from fp16 hidden states.
#
# Probe afterwards on DARWIN: Newton rejects CPU-only jobs and its login node
# OOM-kills the probe stage.
#
# Submit:  ssh newton
#          cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
#          /usr/local/bin/sbatch slurm_optml_llama.sh

set -euo pipefail

REPO=/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
PYTHON=/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3
export HUGGINGFACE_HUB_CACHE=/home/yuval.mandel/hf_kint/hub

IDS=(l3_dpo l3_graddiff l3_idkap l3_ilurmu l3_npoilu
     l3_nposam l3_npo l3_simnpo l3_undial)
REPOS=(OPTML-Group/DPO-WMDP-llama3-8b-instruct
       OPTML-Group/GradDiff-WMDP-llama3-8b-instruct
       OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct
       OPTML-Group/ILU-RMU-WMDP-llama3-8b-instruct
       OPTML-Group/NPO-ILU-WMDP-llama3-8b-instruct
       OPTML-Group/NPO-SAM-WMDP-llama3-8b-instruct
       OPTML-Group/NPO-WMDP-llama3-8b-instruct
       OPTML-Group/SimNPO-WMDP-llama3-8b-instruct
       OPTML-Group/UNDIAL-WMDP-llama3-8b-instruct)

MODEL=${IDS[$SLURM_ARRAY_TASK_ID]}
HFREPO=${REPOS[$SLURM_ARRAY_TASK_ID]}
CACHEDIR="$HUGGINGFACE_HUB_CACHE/models--${HFREPO//\//--}"

cd "$REPO"
mkdir -p inside_out_logs
echo "host=$(hostname) model=$MODEL repo=$HFREPO start=$(date -Is)"

# skip if already extracted
if [ -f "inside_out_out/$MODEL/bio_hs.npy" ]; then
    echo "already extracted, nothing to do"; exit 0
fi

# 0. disk guard.
#    df is USELESS here: /home is a shared 137 TB VAST mount and reports ~49 TB
#    free no matter how full our quota is. There is no quota(1) or lfs(1) on
#    this cluster, and a full "du -s $HOME" takes 2 minutes. So: du the two
#    trees that actually grow (569 of 637 GB on 2026-09-20, 4 s to measure) and
#    add a constant for the static remainder.
HOME_QUOTA_GB=770     # writes began failing at ~772 GB used
OTHER_GB=70           # rest of $HOME outside the two trees below, measured 2026-09-20
NEED_GB=40            # ~16 GB weights + 1.4 GB hidden states + slack

USED_GB=$(du -s --block-size=1G "$HOME/hf_kint" "$HOME/LLMSecurity" 2>/dev/null | awk '{s+=$1} END{print s+0}')
USED_GB=$((USED_GB + OTHER_GB))
FREE_GB=$((HOME_QUOTA_GB - USED_GB))
echo "quota: ${USED_GB}/${HOME_QUOTA_GB} GB used, ${FREE_GB} GB free (need ${NEED_GB})"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "ABORT: only ${FREE_GB} GB left under the ${HOME_QUOTA_GB} GB quota" >&2
    exit 1
fi

# 1. download, safetensors only, with retries (Newton DNS is flaky)
"$PYTHON" - <<PY
from huggingface_hub import snapshot_download
import os, sys, time
for a in range(5):
    try:
        p = snapshot_download("$HFREPO", cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
                              allow_patterns=["*.safetensors","*.json","*.model","*.txt"],
                              max_workers=6, etag_timeout=30)
        print("downloaded", p, flush=True); break
    except Exception as e:
        print("retry", a, repr(e)[:120], flush=True); time.sleep(20)
else:
    sys.exit("download failed after 5 attempts")
PY

# 2. extract
"$PYTHON" inside_out_knowledge.py --stage extract --model_id "$MODEL" --domains bio

# 3. delete the weights, but ONLY on success and ONLY an OPTML-Group directory.
#    The guard matters: a wrong CACHEDIR here would wipe the base model or the
#    LLM-GAT checkpoints, which are not as cheap to replace.
if [ -f "inside_out_out/$MODEL/bio_hs.npy" ] && [ -f "inside_out_out/$MODEL/bio_ext.npy" ]; then
    case "$CACHEDIR" in
        *"models--OPTML-Group--"*)
            du -sh "$CACHEDIR" 2>/dev/null || true
            rm -rf "$CACHEDIR"
            echo "deleted weights: $CACHEDIR"
            ;;
        *)
            echo "REFUSING to delete unexpected path: $CACHEDIR" >&2
            ;;
    esac
else
    echo "extraction produced no output -- keeping weights for a retry" >&2
    exit 1
fi

echo "model=$MODEL end=$(date -Is)"
ls -la "inside_out_out/$MODEL/"
