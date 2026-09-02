#!/bin/bash
# topknorm: top-k layers with the total injected norm matched to the fixed grid.
#
# WHY: bottomk had a norm-matched variant and topk did not, so every
# top-vs-bottom comparison confounded layer choice with injected magnitude. The
# confound is live -- norm-matching the BOTTOM layers lifts them +0.042 -> +0.100,
# level with unnormalised topk (+0.103). topknorm vs bottomknorm is the only
# clean test of whether layer choice matters once the budget is held equal.
#
# Both populations: Q (all 1,273) for the headline, Q* for continuity with the
# existing Q* table. 16 runs over 8 downloads, pipelined and deleting as it goes.
#
# NOTE: RMU and RMU-LAT have a non-significant suppressed_vs_retained profile
# (perm_q >= 0.05), so their selected layers are close to noise; the script
# prints its own warning. Read the top-vs-bottom contrast on the other six.
#
#SBATCH --job-name=recover_topknorm
#SBATCH --output=logs/recover_topknorm_%j.out
#SBATCH --error=logs/recover_topknorm_%j.err
#SBATCH --time=05:00:00
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

dir_of() { echo "$HF_HOME/hub/models--LLM-GAT--llama-3-8b-instruct-$1-checkpoint-8"; }
fetch() {
  local slug=$1 a
  for a in 1 2 3; do
    echo "[fetch] $slug attempt $a $(date +%H:%M:%S)"
    huggingface-cli download "LLM-GAT/llama-3-8b-instruct-$slug-checkpoint-8" \
      --max-workers 8 >"logs/fetch_${slug}.log" 2>&1 && \
      { echo "[fetch] $slug OK $(date +%H:%M:%S)"; return 0; }
    echo "[fetch] $slug FAILED attempt $a"; sleep 30
  done
  return 1
}
drop() { local d; d=$(dir_of "$1"); [ -d "$d" ] && rm -rf "$d" && echo "[drop ] $1"; }
run() {
  local m=$1 pop=$2 t0=$SECONDS
  echo "----- $m / topknorm / pop=$pop  $(date +%H:%M:%S) -----"
  if RECOVER_POP="$pop" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       python plots/causal_recover.py "$m" topknorm 5 correct; then
    echo "[run  ] $m topknorm $pop OK in $((SECONDS - t0))s"
  else
    echo "[run  ] $m topknorm $pop FAILED after $((SECONDS - t0))s"
    FAILED+=("$m/$pop")
  fi
}

echo "===== prefetch ${SLUGS[0]} ====="
fetch "${SLUGS[0]}" || { echo "FATAL: first download failed"; exit 1; }

FAILED=()
for i in $(seq 0 $((N - 1))); do
  M=${METHODS[$i]}; S=${SLUGS[$i]}; NEXT=$((i + 1))
  FETCH_PID=""
  if [ "$NEXT" -lt "$N" ]; then fetch "${SLUGS[$NEXT]}" & FETCH_PID=$!; fi
  echo "########## [$((i + 1))/$N] $M ##########"
  run "$M" q        # headline population first
  run "$M" qstar
  drop "$S"
  df -h "$HF_HOME" | tail -1
  [ -n "$FETCH_PID" ] && { wait "$FETCH_PID" || echo "[fetch] prefetch failed"; }
done

echo "===== done $(date) ====="
ls plots/activation_vectors/causal_recover_*_topknorm*.csv 2>/dev/null | wc -l
echo "expected 16 (8 methods x {q, qstar})"
[ ${#FAILED[@]} -gt 0 ] && { echo "FAILED: ${FAILED[*]}"; exit 1; }
echo "all runs completed"
