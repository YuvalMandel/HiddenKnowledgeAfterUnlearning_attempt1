#!/bin/bash
# submit_70b_smart.sh — detect best available GPUs, submit N parallel 70B workers.
#
# Usage:
#   bash submit_70b_smart.sh                  # auto-detect, 2 workers
#   bash submit_70b_smart.sh --workers 4      # submit 4 workers
#   bash submit_70b_smart.sh --init           # (re)initialise work queue first
#   bash submit_70b_smart.sh --merge-only     # merge completed results, no submission
#   bash submit_70b_smart.sh --status         # print queue progress and exit
#   bash submit_70b_smart.sh --force          # resubmit even if results look complete
#
# GPU type names used in --gres match what SLURM reports in `sinfo -o "%N %G"`.
# If a type name is wrong (job stays pending indefinitely) check with:
#   sinfo -o "%N %G" | grep -i <keyword>

set -euo pipefail
cd "$(dirname "$0")"          # always run from the project root
mkdir -p logs checkpoints

# Activate the conda environment so python3 has torch/datasets/etc.
# Safe to call even if already activated.
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || true
conda activate unlearning 2>/dev/null || true

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WORKERS=2
DO_INIT=0
MERGE_ONLY=0
FORCE=0
STATUS_ONLY=0
PARSABLE=0
DEPENDENCY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)    WORKERS="$2";       shift 2 ;;
        --init)       DO_INIT=1;          shift   ;;
        --merge-only) MERGE_ONLY=1;       shift   ;;
        --status)     STATUS_ONLY=1;      shift   ;;
        --force)      FORCE=1;            shift   ;;
        --parsable)   PARSABLE=1;         shift   ;;
        --dependency) DEPENDENCY="$2";    shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# When --parsable: all verbose output goes to stderr; only job IDs go to stdout.
if [[ $PARSABLE -eq 1 ]]; then
    exec 3>&1 1>&2
fi

# ---------------------------------------------------------------------------
# GPU configs, in priority order: "gres_spec  mem  walltime  label  vram_gb"
#
# Llama-3-70B in bfloat16 needs ~140 GB of VRAM.
#
# Tier 1 — model fits entirely in GPU VRAM (fast, no CPU offloading):
#   ≥160 GB VRAM.  Ordered by speed (H200 > A100 > Ada/Pro > A40/L40/A6000).
#
# Tier 2 — moderate CPU offloading (~40–60 GB spills to RAM, slower):
#   80–140 GB VRAM.  Usable but generation takes 2–3× longer.
#
# GPU type names that are confirmed on this cluster (from sinfo output):
#   A40, H200  — verified
# Type names that follow standard SLURM convention but are UNVERIFIED (*):
#   A100, L40S, L40, A6000, RTX6000Ada, PRO6000, A4000, L4
# Run `sinfo -o "%N %G"` and adjust names if a job stays pending forever.
# ---------------------------------------------------------------------------
GPU_CONFIGS=(
    # ── Tier 1: model fully in VRAM ──────────────────────────────────────────
    # H200 (143 GB each) — nlp-h200-1 (8 GPUs), galileo4 (8 GPUs)
    "gpu:H200:2    80G   24:00:00  2×H200(286GB)"
    "gpu:H200:1    80G   24:00:00  1×H200(143GB)"   # barely fits; worth trying first

    # A100 (80 GB each) — chuck1/2, entropy1/2 (8 GPUs each)  [*name unverified]
    "gpu:A100:2    80G   24:00:00  2×A100(160GB)"
    "gpu:A100:4    80G   24:00:00  4×A100(320GB)"

    # RTX PRO 6000 BW (96 GB each) — nlp-pro6000-1, bruno5, galileo5 (8 each)
    # [*SLURM name unverified — try PRO6000, RTX6000BW, or check sinfo]
#    "gpu:PRO6000:2  80G   36:00:00  2×PRO6000(192GB)"

    # RTX A40 (49 GB each) — nlp-a40-1, newton3/4, galileo1/2, dym-lab/2 (8–10 each)
    "gpu:A40:4    100G   48:00:00  4×A40(196GB)"   # ✓ confirmed working

    # L40S (49 GB each) — newton5, nlp-L40-1/2, dym-lab3, bruno3/4, euler2,
    #                      tdk-bm4, clair1, houdini (4–8 each)  [*name unverified]
    "gpu:L40S:4   100G   48:00:00  4×L40S(196GB)"

    # L40 (49 GB each) — bruno1/2, euler1 (8 each)  [*name unverified]
    "gpu:L40:4    100G   48:00:00  4×L40(196GB)"

    # RTX A6000 (49 GB each) — newton2 (7 GPUs)  [*name unverified]
    "gpu:A6000:4  100G   48:00:00  4×A6000(196GB)"

    # RTX 6000 Ada Generation (49 GB each) — nlp-ada-1/2 (8 each)  [*name unverified]
    "gpu:RTX6000Ada:4  100G  48:00:00  4×RTX6000Ada(196GB)"

    # ── Tier 2: CPU offloading (~40–60 GB spills to RAM, slower) ─────────────
    # Single A100 (80 GB) — 60 GB offloaded to CPU RAM
    "gpu:A100:1   200G   48:00:00  1×A100(80GB,+CPU)"

    # 3× A40/L40S/L40 (147 GB) — model just barely fits; ~0 GB offload
    "gpu:A40:3    100G   72:00:00  3×A40(147GB)"
    "gpu:L40S:3   100G   72:00:00  3×L40S(147GB)"
    "gpu:L40:3    100G   72:00:00  3×L40(147GB)"

#    # Single PRO6000 (96 GB) — 44 GB offloaded
#    "gpu:PRO6000:1  200G  72:00:00  1×PRO6000(96GB,+CPU)"
#
#    # 2× A40/L40S (96 GB) — 44 GB offloaded; NOTE: OOMed in practice with
#    # batch_size=4 during generation.  Reduce LLAMA70B_GEN_BATCH_SIZE to 1
#    # in hidden_knowledge_after_unlearning.py if you use these configs.
#    "gpu:A40:2    200G   96:00:00  2×A40(96GB,+CPU)"
#    "gpu:L40S:2   200G   96:00:00  2×L40S(96GB,+CPU)"
#    "gpu:L40:2    200G   96:00:00  2×L40(96GB,+CPU)"
)

# ---------------------------------------------------------------------------
# Helper: count free GPUs of a given type using sinfo + python
# ---------------------------------------------------------------------------
free_gpus_of_type() {
    local gtype="$1"
    python3 - "$gtype" <<'PYEOF' 2>/dev/null || echo 0
import sys, subprocess, re
gtype = sys.argv[1]
try:
    out = subprocess.check_output(
        ["sinfo", "--noheader", "-O", "gres:100,gresused:100,statelong:20"],
        text=True, stderr=subprocess.DEVNULL,
    )
except Exception:
    print(0); sys.exit()

total_free = 0
for line in out.splitlines():
    parts = line.split()
    if len(parts) < 3:
        continue
    state = parts[2]
    if any(s in state for s in ("drain", "down", "offline", "resv", "maint")):
        continue
    gres     = parts[0]
    gresused = parts[1]
    m = re.search(rf"gpu:{re.escape(gtype)}:(\d+)", gres)
    if not m:
        continue
    total = int(m.group(1))
    m2    = re.search(rf"gpu:{re.escape(gtype)}:(\d+)", gresused)
    used  = int(m2.group(1)) if m2 else 0
    total_free += max(0, total - used)
print(total_free)
PYEOF
}

# ---------------------------------------------------------------------------
# Helper: check whether llama70b_results.json is already complete
# ---------------------------------------------------------------------------
results_complete() {
    python3 - <<'PYEOF' 2>/dev/null
import json, pathlib, sys
p = pathlib.Path("checkpoints/llama70b_results.json")
if not p.exists():
    sys.exit(1)
d = json.load(open(p))
needed = ("bio_gen_stats", "bio_logit_stats", "bio_mcq_stats",
          "cyber_gen_stats", "cyber_logit_stats", "cyber_mcq_stats",
          "bio_mcq_logit_stats", "cyber_mcq_logit_stats")
sys.exit(0 if all(k in d for k in needed) else 1)
PYEOF
}

# ---------------------------------------------------------------------------
# --status
# ---------------------------------------------------------------------------
if [[ $STATUS_ONLY -eq 1 ]]; then
    python3 run_70b_distributed.py --status
    exit $?
fi

# ---------------------------------------------------------------------------
# Short-circuit if already complete
# ---------------------------------------------------------------------------
if results_complete && [[ $FORCE -eq 0 ]] && [[ $MERGE_ONLY -eq 0 ]]; then
    echo "[smart] llama70b_results.json already complete. Nothing to do."
    echo "        (Use --force to resubmit anyway.)"
    exit 0
fi

# ---------------------------------------------------------------------------
# --merge-only: just merge and exit
# ---------------------------------------------------------------------------
if [[ $MERGE_ONLY -eq 1 ]]; then
    echo "[smart] Running merge..."
    python3 run_70b_distributed.py --merge
    exit $?
fi

# ---------------------------------------------------------------------------
# Initialise (or reinitialise) the work queue
# ---------------------------------------------------------------------------
if [[ $DO_INIT -eq 1 ]] || [[ ! -f checkpoints/llama70b_dist/queue.json ]]; then
    echo "[smart] Initialising work queue..."
    python3 run_70b_distributed.py --init
fi

# Check whether all chunks are already done
if python3 run_70b_distributed.py --status --quiet 2>/dev/null; then
    echo "[smart] All chunks done. Merging..."
    python3 run_70b_distributed.py --merge
    exit 0
fi

# ---------------------------------------------------------------------------
# Find best available GPU config
# ---------------------------------------------------------------------------
CHOSEN_GRES=""
CHOSEN_MEM=""
CHOSEN_TIME=""
CHOSEN_LABEL=""

echo "[smart] Checking GPU availability..."
echo "  (SLURM type names marked [*] are unverified — check with: sinfo -o \"%N %G\")"
echo ""
for config in "${GPU_CONFIGS[@]}"; do
    read -r gres mem walltime label <<< "$config"
    gtype=$(echo "$gres" | cut -d: -f2)
    n_req=$(echo "$gres"  | cut -d: -f3)
    free=$(free_gpus_of_type "$gtype")
    if [[ "$free" -ge "$n_req" ]]; then
        avail="✓ available"
    else
        avail="✗ $free free (need $n_req)"
    fi
    printf "  %-28s %s\n" "$label" "$avail"
    if [[ "$free" -ge "$n_req" && -z "$CHOSEN_GRES" ]]; then
        CHOSEN_GRES="$gres"
        CHOSEN_MEM="$mem"
        CHOSEN_TIME="$walltime"
        CHOSEN_LABEL="$label"
    fi
done

if [[ -z "$CHOSEN_GRES" ]]; then
    echo ""
    echo "[smart] Nothing free right now — queuing for 4×A40 (will start when slots open)."
    CHOSEN_GRES="gpu:A40:4"
    CHOSEN_MEM="100G"
    CHOSEN_TIME="48:00:00"
    CHOSEN_LABEL="4×A40(queued)"
fi

echo ""
echo "[smart] Selected: $CHOSEN_LABEL"
echo "        gres=$CHOSEN_GRES  mem=$CHOSEN_MEM  time=$CHOSEN_TIME"
echo "[smart] Submitting $WORKERS worker(s)..."

# ---------------------------------------------------------------------------
# Submit workers
# ---------------------------------------------------------------------------
JOB_IDS=()
for i in $(seq 1 "$WORKERS"); do
    JOBID=$(sbatch --parsable \
        --job-name="hk_70b_w${i}" \
        --output="logs/70b_w${i}_%j.out" \
        --error="logs/70b_w${i}_%j.err" \
        --partition=public \
        --gres="$CHOSEN_GRES" \
        --mem="$CHOSEN_MEM" \
        --time="$CHOSEN_TIME" \
        --cpus-per-task=8 \
        --export=ALL,SMART_RESUBMIT=1,SUBMIT_DIR="$(pwd)" \
        ${DEPENDENCY:+--dependency=$DEPENDENCY} \
        slurm_70b_auto.sh)
    JOB_IDS+=("$JOBID")
    echo "  Worker $i → job $JOBID ($CHOSEN_LABEL)"
done

echo ""
echo "[smart] Monitor:  squeue -u \$USER"
echo "        Status:   bash submit_70b_smart.sh --status"
echo "        Logs:     tail -f logs/70b_w1_<JOBID>.out"

# Output colon-separated job IDs to the original stdout (for --parsable callers).
if [[ $PARSABLE -eq 1 ]]; then
    ( IFS=':'; echo "${JOB_IDS[*]}" >&3 )
fi
