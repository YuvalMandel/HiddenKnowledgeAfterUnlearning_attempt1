#!/bin/bash
#SBATCH --job-name=yi_sp
#SBATCH --partition=public
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=0-1%1
#SBATCH --output=inside_out_logs/yi_%A_%a.out
#SBATCH --error=inside_out_logs/yi_%A_%a.err
#
# Suppression Profile on WMDP's Yi-34B-Chat RMU checkpoint -- the FOURTH model
# family and the last of WMDP's three published RMU base architectures
# (Zephyr-7B and Mixtral-8x7B are done, KNOWN_ISSUES #36).
#
# WHY THIS ONE NEEDED CODE CHANGES. Every model so far had 33 hidden states and
# hidden_dim 4096 -- Llama-3-8B, Mistral-7B and Mixtral-8x7B all coincide. Yi is
# 60 blocks -> 61 hidden states, hidden 7168, vocab 64000. N_LAYERS is now set
# per-model by set_n_layers(): from model.config at extract, from the saved
# array's shape at probe. A cross-probe between different depths is refused
# rather than silently mismatching layers.
#
# WHY A PRO6000 (96 GB). Yi-34B in fp16 is ~68.8 GB of weights; with 61 layers
# of hidden states plus 64k-wide logits at BATCH_SIZE=8 the forward pass peaks
# around 73-75 GB. That does NOT leave safe headroom on an A100-80. The
# PRO6000 has 96 GB on one device and several were idle.
#   Fallbacks, both fine (device_map="auto", hidden states copied to CPU):
#     --gres=gpu:H200:1      (141 GB, galileo4 / nlp-h200-1)
#     --gres=gpu:A100:2
#
# TOKEN IDS: Yi is a genuinely new family, so it is NOT in LEGACY_TF_MODELS and
# takes tf_token_ids()'s derived ids -- the legacy " True"/" False" rule picks
# tokens a sentencepiece model never emits after its template. Extraction also
# writes bio_ext_alt.npy with the other rule, free, for comparison.
#
# DISK. 68.8 GB per model, safetensors only (no .bin duplicates). %1 means one
# at a time and the weights are deleted on success, so peak is one model's
# weights + its 4.2 GB hidden states + a 4.2 GB partial checkpoint = ~78 GB.
# The guard below refuses to start without 85 GB under the real quota.
#
# Probe afterwards on DARWIN -- Newton rejects CPU-only jobs:
#     sbatch --mem=64G --time=24:00:00 --array=22-23 slurm_zephyr_probe.sh
#
# Submit:  ssh newton
#          cd ~/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
#          /usr/local/bin/sbatch slurm_yi.sh

set -euo pipefail

REPO=/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1
PYTHON=/home/yuval.mandel/miniconda3/envs/unlearning/bin/python3
export HUGGINGFACE_HUB_CACHE=/home/yuval.mandel/hf_kint/hub

IDS=(yi_base yi_rmu)
REPOS=(01-ai/Yi-34B-Chat cais/Yi-34B-Chat_RMU)

MODEL=${IDS[$SLURM_ARRAY_TASK_ID]}
HFREPO=${REPOS[$SLURM_ARRAY_TASK_ID]}
CACHEDIR="$HUGGINGFACE_HUB_CACHE/models--${HFREPO//\//--}"

cd "$REPO"
mkdir -p inside_out_logs
echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES model=$MODEL repo=$HFREPO start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ -f "inside_out_out/$MODEL/bio_hs.npy" ]; then
    echo "already extracted, nothing to do"; exit 0
fi

# 0. disk guard. df is useless here (shared 137 TB VAST mount reports ~49 TB
#    free regardless of our quota) and there is no quota(1)/lfs(1) on this
#    cluster, so du the two trees that actually grow and add a constant for the
#    static remainder. Writes began failing at ~772 GB used.
HOME_QUOTA_GB=770
OTHER_GB=70
NEED_GB=85            # 68.8 weights + 4.2 hidden states + 4.2 partial + slack

USED_GB=$(du -s --block-size=1G "$HOME/hf_kint" "$HOME/LLMSecurity" 2>/dev/null | awk '{s+=$1} END{print s+0}')
USED_GB=$((USED_GB + OTHER_GB))
FREE_GB=$((HOME_QUOTA_GB - USED_GB))
echo "quota: ${USED_GB}/${HOME_QUOTA_GB} GB used, ${FREE_GB} GB free (need ${NEED_GB})"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "ABORT: only ${FREE_GB} GB left under the ${HOME_QUOTA_GB} GB quota" >&2
    echo "Free space first -- e.g. delete an extracted model's weights under hf_kint." >&2
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

# 3. delete the weights, but ONLY on success and ONLY a Yi directory. The guard
#    matters: a wrong CACHEDIR would wipe the base model or the LLM-GAT
#    checkpoints, which are not as cheap to replace.
if [ -f "inside_out_out/$MODEL/bio_hs.npy" ] && [ -f "inside_out_out/$MODEL/bio_ext.npy" ]; then
    case "$CACHEDIR" in
        *"models--01-ai--Yi-34B"*|*"models--cais--Yi-34B"*)
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
