# Hidden Knowledge After Unlearning

**Research question:** When a large language model is unlearned (i.e., a method like GradDiff or RMU suppresses certain knowledge from its outputs), does the knowledge disappear from the model's internal representations — or does it remain hidden in the hidden states even though the model stops verbalising it?

---

## What the code does

The experiment uses the [WMDP](https://huggingface.co/datasets/cais/wmdp) biosecurity benchmark as the **forget set** (knowledge that unlearning methods try to erase) and [WikiText-103](https://huggingface.co/datasets/wikitext) passages as the **retain set** (general knowledge that should be preserved).

### Pipeline overview

```
Stage 1: base      ──► Stage 2: methods (×8, parallel) ──► Stage 3: summary
```

#### Stage 1 — Base model (`--stage base`)
1. Load and split WMDP-bio questions into **train / val / test** (500 / 200 / rest).
2. Each question becomes two yes/no prompts: one with the *correct* answer (→ "Yes") and one with a *wrong* answer (→ "No").  WikiText passages are similarly turned into correct/wrong continuation pairs.
3. Extract **hidden states** (all transformer layers, last token) for train/val/test pairs.
4. Train a **linear probe** (logistic regression) per layer; pick the best layer on the validation set.
5. Run **generation** on test and retain sets; record the first word (Yes/No).
6. Run a **logit-based metric**: a forward pass at the last input token records the max logit over Yes-tokens vs No-tokens — no decoding required.
7. Save everything to `checkpoints/`.

#### Stage 2 — Unlearned models (`--stage method --method <NAME>`)
Runs independently for each of the 8 LLM-GAT unlearning methods (all in parallel via SLURM job array):

1. Extract hidden states of **train / val / test** using the *unlearned* model.
2. Train a **method-specific probe** on those hidden states (same train set, different model).  Find its best layer on the unlearned model's val hidden states.
3. Run generation and logit scoring on test and retain sets.
4. Evaluate with **two probe sets**:
   - **Base probes** (trained on base-model hs) — do the base-model's learned directions transfer?
   - **Method probes** (trained on unlearned-model hs) — is a *new* direction still detectable?
5. Save to `checkpoints/`.

#### Stage 3 — Summary (`--stage summary`)
Loads all checkpoints (no GPU needed) and prints:
- Per-method sample Q&A comparisons
- Three summary tables (generation + base probe + logit, method-specific probes, retain set)

### Metrics

| Metric | What it measures |
|---|---|
| **Generation accuracy** | Does the model *say* the right Yes/No? |
| **Base probe accuracy** | Do directions learned from the base model's hidden states still classify Yes/No in the unlearned model? |
| **Method probe accuracy** | Does a probe trained on the *unlearned* model's own hidden states still find the knowledge? |
| **Logit accuracy** | Is the logit for the correct Yes/No token higher than the wrong one, without any generation? |

A gap between generation ↓ and probe/logit accuracy ↑ is evidence of **residual hidden knowledge** after unlearning.

---

## File structure

```
hidden_knowledge_after_unlearning.py   Main Python script (all stages)
submit_pipeline.sh                     Submit all jobs with SLURM dependencies
run_hidden_knowledge.sh                Convenience alias for submit_pipeline.sh
slurm_base.sh                          SLURM script for stage 1 (base model)
slurm_methods.sh                       SLURM job array for stage 2 (8 methods)
slurm_summary.sh                       SLURM script for stage 3 (summary)
checkpoints/                           Auto-created; holds .npy, .pkl, .json
logs/                                  Auto-created; SLURM stdout/stderr
```

---

## Setup

### Requirements

- Python 3.10+
- PyTorch with CUDA (tested on A40 48 GB)
- `transformers`, `datasets`, `scikit-learn`, `numpy`

```bash
conda create -n unlearning python=3.10
conda activate unlearning
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets scikit-learn numpy
```

You need a Hugging Face account with access to `meta-llama/Meta-Llama-3-8B-Instruct`:

```bash
huggingface-cli login
```

---

## Running the code

### On a SLURM cluster (recommended)

```bash
# Submit all three stages with automatic job dependencies:
bash submit_pipeline.sh

# Monitor:
squeue -u $USER

# Logs:
tail -f logs/base_<JOBID>.out
tail -f logs/method_<TASK>_<JOBID>.out
tail -f logs/summary_<JOBID>.out
```

The three stages run as:
- `slurm_base.sh` — 1 × A40, up to 8 h
- `slurm_methods.sh` — 8 × A40 in parallel (job array), up to 10 h each
- `slurm_summary.sh` — CPU-only, 30 min

Stage 2 starts automatically once stage 1 succeeds; stage 3 starts once all stage-2 tasks succeed.

### Running a single method manually

```bash
# Prerequisite: stage 1 must have completed first.
python hidden_knowledge_after_unlearning.py --stage method --method GradDiff
```

### Running locally (no SLURM)

```bash
python hidden_knowledge_after_unlearning.py --stage base
python hidden_knowledge_after_unlearning.py --stage method --method GradDiff
# ... repeat for each method ...
python hidden_knowledge_after_unlearning.py --stage summary
```

### Resubmitting after preemption

Checkpoints are saved after each heavy operation (hidden-state extraction arrays, generation answers, logit scores).  Simply resubmit the failed job — it will resume from where it left off:

```bash
sbatch slurm_methods.sh   # or the full pipeline again
```

To start completely from scratch, delete the `checkpoints/` directory.

---

## Configuration

Key constants at the top of `hidden_knowledge_after_unlearning.py`:

| Constant | Default | Description |
|---|---|---|
| `BASE_MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | Base (un-unlearned) model |
| `UNLEARNED_MODELS` | 8 LLM-GAT checkpoints | Dict of method name → HF model ID |
| `FORGET_SUBSET` | `wmdp-bio` | WMDP subset to treat as forget set |
| `TRAIN_SIZE` | 500 | Questions used to train probes |
| `VAL_SIZE` | 200 | Questions used to select best probe layer |
| `N_RETAIN_PASSAGES` | 3 | Number of WikiText passage pairs |
| `GENERATION_BATCH_SIZE` | 8 | Batch size for text generation |
| `HIDDEN_STATE_BATCH_SIZE` | 8 | Batch size for hidden-state extraction |
| `LOGIT_BATCH_SIZE` | 16 | Batch size for logit-score computation |
| `MAX_NEW_TOKENS` | 64 | Max tokens generated per prompt |

---

## Output

The summary stage prints three tables:

```
SUMMARY — FORGET SET (test) — Generation / Base-model Probe / Logit
═══════════════════════════════════════════════════════════════════════
Method        GenAcc  GYes   GNo   Gib  BaseProbe  BPYes  BPNo  LogitAcc  LYes   LNo
──────────────────────────────────────────────────────────────────────
Base           0.XXX 0.XXX 0.XXX 0.XXX      0.XXX  0.XXX 0.XXX     0.XXX 0.XXX 0.XXX
GradDiff       ...
...

SUMMARY — FORGET SET (test) — Method-Specific Probes
  (probes trained on the UNLEARNED model's own hidden states)
═══════════════════════════════
Method       BestLyr  MProbe  MPYes   MPNo
...

SUMMARY — RETAIN SET — Generation / Logit  (should stay near 1.0)
═══════════════════════════════════════════════════════════
Method       RetainAcc   RYes   RNo   RLogit  RLYes   RLNo
...
```

**Column guide:**
- `Gen*` — generation accuracy (first word of model output is Yes/No)
- `BaseProbe` / `BP*` — base-model probe applied to unlearned model hidden states
- `Logit*` — max(Yes-token logits) vs max(No-token logits) comparison
- `MProbe` / `MP*` — probe trained on the *unlearned* model's own hidden states
- `Retain*` — same metrics on the WikiText retain set (should remain high)
- `Gibberish` — fraction of outputs that contain neither "Yes" nor "No"
