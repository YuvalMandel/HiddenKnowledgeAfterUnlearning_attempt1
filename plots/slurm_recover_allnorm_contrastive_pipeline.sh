#!/bin/bash
# allnorm x contrastive causal recovery, all 8 methods, PIPELINED.
#
# Why a pipeline and not a job array: Newton's HF cache was purged (blobs/ is
# empty, every snapshot symlink dangles), so all 8 checkpoints must be
# re-downloaded ~16 GB each. An 8-task array would need all 128 GB resident at
# once. This holds ONE GPU and overlaps the download of model i+1 with the
# compute of model i, deleting model i as soon as it is done. Peak disk is two
# checkpoints (~32 GB) instead of eight.
#
#   fetch 0
#   for i: [ fetch i+1 in background ] || run i ; drop i ; wait
#
# Vector mode is "contrastive": centroid = correct - mean(wrong options), so
#   d_S = [correct-mean(wrong)]_base - [correct-mean(wrong)]_ck8   (same for d_F)
# Layer mode is "allnorm": inject at every layer 1..32, with d_S and d_F rescaled
# so their TOTAL injected L2 norm equals the fixed {3,6,9,12,15} grid's budget.
# The random controls d_R / d_RF are built AFTER that rescale and normalised to
# ||d_S[L]|| / ||d_F[L]||, so all four arms sit at one matched norm budget.
# Note BUDGET is computed under the active VECTOR_MODE, so this run is matched to
# the CONTRASTIVE fixed-grid norm, not the correct-only one.
#
# Outputs: plots/activation_vectors/causal_recover_<METHOD>_allnorm_contrastive.csv
#
#SBATCH --job-name=recover_ac_pipe
#SBATCH --output=logs/recover_ac_pipe_%j.out
#SBATCH --error=logs/recover_ac_pipe_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -u
echo "start $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unlearning
export HF_HOME="$HOME/.cache/huggingface"
cd "$HOME/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1" || exit 1
mkdir -p logs plots/activation_vectors

METHODS=(RepNoise GradDiff PB_J   TAR  RR  ELM  RMU  RMU-LAT)
SLUGS=(  repnoise graddiff pbj    tar  rr  elm  rmu  rmu-lat)
N=${#METHODS[@]}

repo_of() { echo "LLM-GAT/llama-3-8b-instruct-$1-checkpoint-8"; }
dir_of()  { echo "$HF_HOME/hub/models--LLM-GAT--llama-3-8b-instruct-$1-checkpoint-8"; }

fetch() {   # $1 = slug ; retries, because Newton's DNS is flaky
  local slug=$1 repo attempt
  repo=$(repo_of "$slug")
  for attempt in 1 2 3; do
    echo "[fetch] $slug attempt $attempt $(date +%H:%M:%S)"
    if huggingface-cli download "$repo" --max-workers 8 >"logs/fetch_${slug}.log" 2>&1; then
      echo "[fetch] $slug OK $(date +%H:%M:%S) ($(du -sh "$(dir_of "$slug")" 2>/dev/null | cut -f1))"
      return 0
    fi
    echo "[fetch] $slug FAILED attempt $attempt; see logs/fetch_${slug}.log"
    sleep 30
  done
  return 1
}

drop() {    # $1 = slug -- free the ~16 GB immediately after compute
  local d
  d=$(dir_of "$1")
  [ -d "$d" ] && rm -rf "$d" && echo "[drop ] $1 removed $(date +%H:%M:%S)"
}

echo "===== prefetch ${SLUGS[0]} ====="
fetch "${SLUGS[0]}" || { echo "FATAL: first download failed"; exit 1; }

FAILED=()
for i in $(seq 0 $((N - 1))); do
  M=${METHODS[$i]}
  S=${SLUGS[$i]}
  NEXT=$((i + 1))

  FETCH_PID=""
  if [ "$NEXT" -lt "$N" ]; then
    fetch "${SLUGS[$NEXT]}" &
    FETCH_PID=$!
  fi

  echo "===== [$((i + 1))/$N] $M  allnorm contrastive  $(date +%H:%M:%S) ====="
  T0=$SECONDS
  # offline only for the compute step, so it cannot stall on the network;
  # the background download deliberately runs WITHOUT these set.
  if TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       python plots/causal_recover.py "$M" allnorm 5 contrastive; then
    echo "[run  ] $M OK in $((SECONDS - T0))s"
  else
    echo "[run  ] $M FAILED after $((SECONDS - T0))s"
    FAILED+=("$M")
  fi

  drop "$S"
  df -h "$HF_HOME" | tail -1

  if [ -n "$FETCH_PID" ]; then
    wait "$FETCH_PID" || echo "[fetch] background download for ${SLUGS[$NEXT]} failed"
  fi
done

echo "===== done $(date) ====="
echo "outputs:"
ls -la plots/activation_vectors/causal_recover_*_allnorm_contrastive.csv 2>/dev/null
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED METHODS: ${FAILED[*]}"
  exit 1
fi
echo "all $N methods completed"
