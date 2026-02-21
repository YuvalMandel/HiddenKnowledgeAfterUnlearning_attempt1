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
3. Extract **hidden states** (all transformer layers, last non-pad token via `attention_mask`) for train/val/test pairs.  The chat template is applied *without* the generation-prompt suffix so the probed token is the final user-message token, not an assistant-turn marker.
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

### Probe types

For **every model** (base + all 8 unlearned), two families of probes are trained on the **train set** hidden states, validated on the **val set**, and evaluated on the **test set**:

| Family | Input | Classifiers |
|---|---|---|
| **Per-layer** | Single layer's hidden state vector (4096-d) | LR · RF (PCA→64 first) · AdaBoost (PCA→64) |
| **Multi-layer** | All layers concatenated, then PCA→256 | LR · RF · AdaBoost |

- **Per-layer**: one classifier per transformer layer per type; best layer selected independently for each classifier type on the val set.
- **Multi-layer**: single classifier sees information from every layer simultaneously; PCA is applied first to keep training tractable.

The four combinations of (training hs source) × (test hs source) form a 2×2 matrix:

```
Train \ Test  │  Base model hs     │  Unlearned model hs
──────────────┼────────────────────┼─────────────────────
Base probes   │  Table 2 diagonal  │  Table 2
Method probes │  Table 5  ★        │  Table 3
```

- **Table 2 diagonal** (train pre → test pre): baseline — probes trained and tested on the base model.
- **Table 2** (train pre → test post): do the base model's learned directions still work in the unlearned model?
- **Table 3** (train post → test post): does a fresh probe on the unlearned model still find the knowledge?
- **Table 5 ★** (train post → test pre): do the unlearned model's directions transfer back to the base model? If yes, the representations are similar; if no, unlearning shifted them significantly.

### Metrics

| Metric | What it measures |
|---|---|
| **Generation accuracy** | Does the model *say* the right Yes/No? |
| **Base probe (LR/RF/AdaBoost)** | Do directions learned from the base model still classify Yes/No in the unlearned model's hidden states? |
| **Method probe (LR/RF/AdaBoost)** | Does a probe trained on the *unlearned* model's own hidden states still find the knowledge? |
| **Logit accuracy** | Is the logit for the correct Yes/No token higher than the wrong one, without generation?  The generation prompt *is* included so the last token position predicts the first output token. |

A gap between generation ↓ and probe/logit accuracy ↑ is evidence of **residual hidden knowledge** after unlearning.
Tree-based probes (RF, AdaBoost) can detect non-linear residual structure that LR would miss.

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
| `PCA_DIMS_PER_LAYER` | 64 | PCA components before RF/AdaBoost per-layer probes |
| `PCA_DIMS_MULTI` | 256 | PCA components for all multi-layer probes |

---

## Output

The summary stage prints four tables:

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
- `Gen*` — generation accuracy (first word of model output is Yes/No)
- `Logit*` — max(Yes-token logits) vs max(No-token logits) at the last input position
- `PL-{clf}` — per-layer probe at that classifier's independently chosen best validation layer
- `ML-{clf}` — multi-layer probe (all layers concatenated, then PCA-256)
- `Ret*` — retain-set metrics (should stay high)
- `Gibberish` — fraction of outputs containing neither "Yes" nor "No"

**Tables 2, 3, 5** all share the same column layout — per-layer (PL) and multi-layer (ML) results for LR, RF, and AdaBoost. The distinction is which probes are applied to which hidden states (see the 2×2 matrix above).

**Note on old checkpoints:** If a method checkpoint was created before Table 5 was added, the method stage will automatically compute the missing cross-probe stats from the saved probes and `base_hs_test.npy` without a full rerun.
