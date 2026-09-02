#!/bin/bash
# Split-half at matched budget: topknorm16 vs bottomknorm16.
#
# At k=16 the two sets PARTITION the 32 layers -- 16 each, overlap 0 for six of
# eight methods (1 for TAR and RMU, from AUC ties), and together they are
# exactly "all". Same layer count, same injected norm budget, complementary
# halves, every layer used once. This is the clean test of whether layer choice
# matters once magnitude is held equal.
#
# Motivation: norm-matching the BOTTOM five layers lifted them +0.042 -> +0.100,
# level with unnormalised topk5 (+0.103), and allnorm -- which selects nothing --
# has the largest suppressed effect of any rule (+0.131). All of which suggests
# the injected budget, not the targeting, is doing the work. If topknorm16 and
# bottomknorm16 come out level, that is close to decisive.
#
# NO OVERWRITE: causal_recover.py now records K in the filename whenever K != 5,
# so these write _topknorm16 / _bottomknorm16 and every existing k=5 output keeps
# its name. Verified: 0 of the 64 planned files already exist.
#
# NOTE: RMU and RMU-LAT have a non-significant suppressed_vs_retained profile
# (perm_q >= 0.05), so their layer ordering is close to noise; read the contrast
# on the other six methods.
#
#SBATCH --job-name=recover_split16
#SBATCH --output=logs/recover_split16_%j.out
#SBATCH --error=logs/recover_split16_%j.err
#SBATCH --time=06:00:00
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
RULES=(topknorm bottomknorm)

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
  local m=$1 rule=$2 pop=$3 t0=$SECONDS out
  out="plots/activation_vectors/causal_recover_${m}_${rule}16"
  [ "$pop" = "q" ] && out="${out}_q"
  if [ -f "${out}.csv" ]; then
    echo "[skip ] ${out}.csv already exists -- refusing to overwrite"
    return
  fi
  echo "----- $m / ${rule}16 / pop=$pop  $(date +%H:%M:%S) -----"
  if RECOVER_POP="$pop" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       python plots/causal_recover.py "$m" "$rule" 16 correct; then
    echo "[run  ] $m ${rule}16 $pop OK in $((SECONDS - t0))s"
  else
    echo "[run  ] $m ${rule}16 $pop FAILED after $((SECONDS - t0))s"
    FAILED+=("$m/${rule}16/$pop")
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
  for r in "${RULES[@]}"; do
    run "$M" "$r" q          # headline population first
  done
  for r in "${RULES[@]}"; do
    run "$M" "$r" qstar
  done
  drop "$S"
  df -h "$HF_HOME" | tail -1
  [ -n "$FETCH_PID" ] && { wait "$FETCH_PID" || echo "[fetch] prefetch failed"; }
done

echo "===== done $(date) ====="
echo "outputs: $(ls plots/activation_vectors/causal_recover_*norm16*.csv 2>/dev/null | wc -l) (expected 32)"
echo "k=5 outputs still intact: $(ls plots/activation_vectors/causal_recover_*_topknorm.csv plots/activation_vectors/causal_recover_*_bottomknorm.csv 2>/dev/null | wc -l)"
[ ${#FAILED[@]} -gt 0 ] && { echo "FAILED: ${FAILED[*]}"; exit 1; }
echo "all runs completed"
