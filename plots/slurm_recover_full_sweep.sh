#!/bin/bash
# Two jobs in one pipeline, every layer-selection rule.
#
# PART A -- Q* backfill. topk5 and band5 are missing the forg_random control for
#   6 of 8 methods each, so their arm-separation column rests on one or two
#   methods. Re-running emits all four conditions and completes the comparison.
#   Run for all 8 methods, not just the 12 gaps, so the whole column is produced
#   by one script version.
#
# PART B -- population Q. Every rule re-run with RECOVER_POP=q, i.e. the
#   suppressed/forgotten sets drawn from all 1,273 questions instead of the 701
#   of Q*. This is the run that matters: on Q* the forgotten cell is 21-68
#   questions, so only 9-28 are held out and the suppressed-vs-forgotten test is
#   underpowered under EVERY rule (best Wilcoxon p = 0.109). On Q the forgotten
#   cell is 138-193. It also puts 5.3 on the same population as 5.1 and 5.2.
#   Outputs carry a _q suffix, so nothing overwrites the Q* results.
#
# Pipelined exactly as job 1354174: download model i+1 while computing model i,
# delete model i the moment its runs finish. Peak disk is two checkpoints.
#
#SBATCH --job-name=recover_sweep
#SBATCH --output=logs/recover_sweep_%j.out
#SBATCH --error=logs/recover_sweep_%j.err
#SBATCH --time=14:00:00
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

# mode name + K passed to causal_recover.py; K is ignored by the non-k rules
RULES=(fixed topk band late all allnorm latenorm bottomk bottomknorm)

dir_of() { echo "$HF_HOME/hub/models--LLM-GAT--llama-3-8b-instruct-$1-checkpoint-8"; }

fetch() {
  local slug=$1 attempt
  for attempt in 1 2 3; do
    echo "[fetch] $slug attempt $attempt $(date +%H:%M:%S)"
    if huggingface-cli download "LLM-GAT/llama-3-8b-instruct-$slug-checkpoint-8" \
         --max-workers 8 >"logs/fetch_${slug}.log" 2>&1; then
      echo "[fetch] $slug OK $(date +%H:%M:%S)"
      return 0
    fi
    echo "[fetch] $slug FAILED attempt $attempt"
    sleep 30
  done
  return 1
}

drop() { local d; d=$(dir_of "$1"); [ -d "$d" ] && rm -rf "$d" && echo "[drop ] $1 $(date +%H:%M:%S)"; }

run() {   # $1 method  $2 rule  $3 population
  local m=$1 rule=$2 pop=$3 t0=$SECONDS
  echo "----- $m / $rule / pop=$pop  $(date +%H:%M:%S) -----"
  if RECOVER_POP="$pop" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       python plots/causal_recover.py "$m" "$rule" 5 correct; then
    echo "[run  ] $m $rule $pop OK in $((SECONDS - t0))s"
  else
    echo "[run  ] $m $rule $pop FAILED after $((SECONDS - t0))s"
    FAILED+=("$m/$rule/$pop")
  fi
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

  echo "########## [$((i + 1))/$N] $M ##########"
  # PART B first -- it is the run that matters, so it completes even if the job
  # hits the wall clock partway through the last model.
  for r in "${RULES[@]}"; do
    run "$M" "$r" q
  done
  # PART A: the two rules whose Q* forgotten control is missing
  run "$M" topk qstar
  run "$M" band qstar

  drop "$S"
  df -h "$HF_HOME" | tail -1
  [ -n "$FETCH_PID" ] && { wait "$FETCH_PID" || echo "[fetch] prefetch failed"; }
done

echo "===== done $(date) ====="
echo "Q-population outputs:"
ls plots/activation_vectors/causal_recover_*_q.csv 2>/dev/null | wc -l
echo "expected 72 (8 methods x 9 rules; the fixed rule writes causal_recover_<M>_q.csv)"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
echo "all runs completed"
