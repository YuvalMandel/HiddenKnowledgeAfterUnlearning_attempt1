# Hidden Knowledge After Unlearning

**Research question:** When a large language model is unlearned (i.e., a method like GradDiff or RMU suppresses certain knowledge from its outputs), does the knowledge disappear from the model's internal representations — or does it remain hidden in the hidden states even though the model stops verbalising it?

---

## What the code does

The experiment uses the [WMDP](https://huggingface.co/datasets/cais/wmdp) biosecurity benchmark as the **forget set** (knowledge that unlearning methods try to erase) and [WikiText-103](https://huggingface.co/datasets/wikitext) passages as the **retain set** (general knowledge that should be preserved).

### Pipeline overview

```
Stage 1: base      ──► Stage 2: methods (×9, parallel) ──► Stage 3: summary
```

#### Stage 1 — Base model (`--stage base`)
1. Load and split WMDP-bio questions into **train / val / test** (500 / 200 / rest).
2. Each question becomes two True/False prompts: one with the *correct* answer (→ "True") and one with a *wrong* answer (→ "False").  The prompt is a semantic statement ("The answer to the question '…' is '…'.") rather than a surface question, making the label a truth-value rather than a surface token.  WikiText passages are similarly turned into correct/wrong continuation pairs.  All three splits are saved to `checkpoints/wmdp_tf_pairs.csv` on first run; subsequent runs load from the CSV to skip re-downloading and ensure reproducible splits.
3. Extract **hidden states** (all transformer layers, last non-pad token via `attention_mask`) for train/val/test pairs.  The chat template is applied *without* the generation-prompt suffix so the probed token is the final user-message token, not an assistant-turn marker.
4. Train a **linear probe** (logistic regression) per layer; pick the best layer on the validation set.
5. Run **generation** on test and retain sets; record the first word (True/False).
6. Run a **logit-based metric**: a forward pass at the last input token records the max logit over True-tokens vs False-tokens — no decoding required.
7. Run **MCQ direct evaluation**: give the model each original multiple-choice question with all four answer options and record the A/B/C/D letter response (Table 6).
8. Save everything to `checkpoints/`.

#### Stage 2 — Unlearned models (`--stage method --method <NAME>`)
Runs independently for each of the 8 LLM-GAT unlearning methods plus the raw `Llama3-8B` reference (9 jobs total, all in parallel via SLURM job array):

1. Load forget pairs from `checkpoints/wmdp_tf_pairs.csv` (created by stage 1).
2. Extract hidden states of **train / val / test** using the *unlearned* model.
3. Train a **method-specific probe** on those hidden states (same train set, different model).  Find its best layer on the unlearned model's val hidden states.
4. Run generation and logit scoring on test and retain sets.
5. Run **MCQ direct evaluation** on the test questions (Table 6).
6. Evaluate with **two probe sets**:
   - **Base probes** (trained on base-model hs) — do the base-model's learned directions transfer?
   - **Method probes** (trained on unlearned-model hs) — is a *new* direction still detectable?
7. Save to `checkpoints/`.

#### Stage 3 — Summary (`--stage summary`)
Loads all checkpoints (no GPU needed) and prints:
- Per-method sample Q&A comparisons
- Six summary tables (see Output section)

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
| **Generation accuracy** | Does the model *say* the right True/False? |
| **Base probe (LR/RF/AdaBoost)** | Do directions learned from the base model still classify True/False in the unlearned model's hidden states? |
| **Method probe (LR/RF/AdaBoost)** | Does a probe trained on the *unlearned* model's own hidden states still find the knowledge? |
| **Logit accuracy** | Is the logit for the correct True/False token higher than the wrong one, without generation?  The generation prompt *is* included so the last token position predicts the first output token. |
| **MCQ direct accuracy** | Given the original multiple-choice question with all four options, does the model pick the correct letter (A/B/C/D)? |

A gap between generation ↓ and probe/logit accuracy ↑ is evidence of **residual hidden knowledge** after unlearning.
Tree-based probes (RF, AdaBoost) can detect non-linear residual structure that LR would miss.

### Prompt design: True/False over Yes/No

Prompts are framed as truth-value judgements rather than direct questions:

```
Statement: The answer to the question '{question}' is '{proposed_answer}'.
Respond with only 'True' or 'False'.
```

**Why True/False instead of Yes/No:**  "Yes" and "No" are surface-level response tokens that the model has been trained to produce in many superficially different contexts.  "True" and "False" encode a *semantic property* — the truth of an explicit propositional statement — which more cleanly isolates whether the model retains factual knowledge, regardless of whether it has been fine-tuned to refuse or redirect surface answers.

---

## File structure

```
hidden_knowledge_after_unlearning.py       Main Python script (all stages)
submit_pipeline.sh                         Submit all jobs with SLURM dependencies
run_hidden_knowledge.sh                    Convenience alias for submit_pipeline.sh
slurm_base.sh                              SLURM script for stage 1 (base model)
slurm_methods.sh                           SLURM job array for stage 2 (8 methods)
slurm_summary.sh                           SLURM script for stage 3 (summary)
checkpoints/                                 Auto-created; holds .npy, .pkl, .json
checkpoints/wmdp_tf_pairs.csv               Cached WMDP train/val/test pairs (created on first run)
checkpoints/summary_table1_gen_logit.csv    Table 1 CSV (generation + logit)
checkpoints/summary_table2_base_probes.csv  Table 2 CSV (base probes)
checkpoints/summary_table3_method_probes.csv Table 3 CSV (method-specific probes)
checkpoints/summary_table4_retain.csv       Table 4 CSV (retain set)
checkpoints/summary_table5_cross_probes.csv Table 5 CSV (cross-probe quadrant)
checkpoints/summary_table6_mcq.csv          Table 6 CSV (MCQ direct A/B/C/D)
logs/                                        Auto-created; SLURM stdout/stderr
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
- `slurm_methods.sh` — 9 × A40 in parallel (job array), up to 10 h each
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
| `WMDP_CSV_PATH` | `checkpoints/wmdp_tf_pairs.csv` | Cached WMDP pairs; delete to force rebuild |
| `UNLEARNED_MODELS` | 8 LLM-GAT checkpoints + `Llama3-8B` | Dict of method name → HF model ID; includes raw `meta-llama/Meta-Llama-3-8B` as a reference |
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

The summary stage prints six tables and saves each as a CSV under `checkpoints/`:

```
TABLE 1 — FORGET SET (test): Generation accuracy + Logit-based metric
Method        GenAcc  GTrue  GFalse   Gib  LogitAcc  LTrue  LFalse
────────────────────────────────────────────────────────────────────
Base           0.XXX  0.XXX   0.XXX 0.XXX     0.XXX  0.XXX   0.XXX
GradDiff       ...
...

TABLE 2 — FORGET SET (test): BASE-model probes applied to test hidden states
Method   PL-LR Acc True Fals Lyr  PL-RF ...  ML-LR ...
...

TABLE 3 — FORGET SET (test): METHOD-SPECIFIC probes
  (probes trained on the unlearned model's own hidden states)
Method   PL-LR Acc True Fals Lyr  ...
...

TABLE 4 — RETAIN SET: Generation + Logit  (should stay near 1.0)
Method       RetAcc  RTrue  RFalse   RLogit  RLTrue  RLFalse
...

TABLE 5 — FORGET SET (test): METHOD probes applied to BASE-model hidden states
  (cross-quadrant: do unlearned-model directions transfer back to the base model?)
Method   PL-LR Acc True Fals Lyr  ...
...

TABLE 6 — MCQ DIRECT: Original multiple-choice questions (A/B/C/D)
Method         Acc   Acc_A  Acc_B  Acc_C  Acc_D    Gib
...
```

**Column guide:**
- `Gen*` — generation accuracy (first word of model output is True/False)
- `Logit*` — max(True-token logits) vs max(False-token logits) at the last input position
- `PL-{clf}` — per-layer probe at that classifier's independently chosen best validation layer
- `ML-{clf}` — multi-layer probe (all layers concatenated, then PCA-256)
- `Ret*` — retain-set metrics (should stay high)
- `Gibberish` — fraction of outputs containing neither "True"/"False" (Tables 1–5) or a valid letter A–D (Table 6)
- `Acc_A/B/C/D` — per-answer-letter accuracy for MCQ questions whose correct answer is that letter

**Tables 2, 3, 5** all share the same column layout — per-layer (PL) and multi-layer (ML) results for LR, RF, and AdaBoost. The distinction is which probes are applied to which hidden states (see the 2×2 matrix above).

**Note on old checkpoints:** If a method checkpoint was created before Table 5 was added, the method stage will automatically compute the missing cross-probe stats from the saved probes and `base_hs_test.npy` without a full rerun.
