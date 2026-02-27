# Hidden Knowledge After Unlearning

**Research question:** When a large language model is unlearned (i.e., a method like GradDiff or RMU suppresses certain knowledge from its outputs), does the knowledge disappear from the model's internal representations — or does it remain hidden in the hidden states even though the model stops verbalising it?

---

## What the code does

The experiment uses the [WMDP](https://huggingface.co/datasets/cais/wmdp) biosecurity benchmark as the **forget set** (knowledge that unlearning methods try to erase) and WMDP **cyber** questions (from `data/wmdp_cyber_true_false_balanced.csv`) as the **cyber set** — a parallel domain used to measure how well knowledge is retained in a different but similarly structured hazard domain.

### Pipeline overview

```
Stage 1: base      ──► Stage 2: methods (×9, parallel) ──► Stage 3: summary
                   └── Stage 4: sweep  (×8, parallel)
```

Stage 4 (sweep) is independent of stages 2–3 and only requires stage 1 to have finished.

#### Stage 1 — Base model (`--stage base`)
1. Load and split WMDP-bio questions into **train / val / test** (500 / 200 / rest).
2. Each WMDP-bio question becomes two True/False prompts (correct answer → "True", wrong answer → "False"). The WMDP-cyber CSV (`data/wmdp_cyber_true_false_balanced.csv`) is loaded the same way — each `question_id` has one True and one False row, and both sets use the **identical prompt format**: `make_tf_prompt(question, choice)` → `"Claim: The answer to '{question}' is '{choice}'."`. All bio pairs are saved to `data/wmdp_tf_pairs.csv` on first run.
3. Extract **hidden states** (all transformer layers, last non-pad token via `attention_mask`) for **bio train/val/test** and **cyber train/val/test** pairs.
4. Train separate probe sets (LR/RF/AdaBoost per-layer + multi-layer) for **bio** and **cyber**.
5. Run **generation** and **logit scoring** on bio test and cyber test sets.
6. Run **MCQ direct evaluation** on bio test questions (Table 6).
7. Save everything to `checkpoints/`.

#### Stage 2 — Unlearned models (`--stage method --method <NAME>`)
Runs independently for each of the 8 LLM-GAT unlearning methods plus the raw `Llama3-8B` reference (9 jobs total, all in parallel via SLURM job array):

1. Load bio forget pairs from `data/wmdp_tf_pairs.csv` and cyber pairs from `data/wmdp_cyber_true_false_balanced.csv`.
2. Extract hidden states of **bio train/val/test** and **cyber train/val/test** using the *unlearned* model.
3. Train **method-specific probe sets** for bio and cyber separately. Find best layer on unlearned-model val hidden states.
4. Run generation and logit scoring on bio test and cyber test sets.
5. Run **MCQ direct evaluation** on bio test questions (Table 6).
6. Evaluate with **four probe sets** per domain (bio & cyber):
   - **Base probes on unlearned hs** — do base-model directions transfer?
   - **Method probes on unlearned hs** — is a fresh direction still detectable?
   - **Method probes on base hs** — cross-quadrant: do unlearned directions transfer back?
7. Save to `checkpoints/`.

#### Stage 3 — Summary (`--stage summary`)
Loads all checkpoints (no GPU needed) and prints:
- Per-method sample Q&A comparisons
- Six summary tables (see Output section)

#### Stage 4 — Checkpoint sweep

Three sweep sub-stages:

| Sub-stage | CLI flag | GPU | Purpose |
|---|---|---|---|
| Single checkpoint | `--stage sweep --method M --checkpoint N` | Yes | Run one (method, checkpoint) pair — used by the parallel SLURM array |
| Sequential sweep | `--stage sweep --method M` | Yes | Run all 8 checkpoints in one job (local / fallback) |
| Summary | `--stage sweep_summary --method M` | No | Load cached results and print table / save CSV |

For each checkpoint the sweep:
1. Extracts **bio and cyber train/val/test hidden states** from that checkpoint.
2. Runs **generation**, **logit scoring**, and **MCQ** on the bio test set; generation and logit scoring on the cyber test set.
3. Evaluates four probe sets:
   - **Bio base probes** — trained on the base instruct model (loaded from stage 1, no retraining).
   - **Bio per-checkpoint probes** — trained on this checkpoint's own bio hidden states.
   - **Cyber base probes** — base model's cyber probes applied to this checkpoint's cyber hidden states.
   - **Cyber per-checkpoint probes** — trained on this checkpoint's own cyber hidden states.
4. Caches all arrays and results under `checkpoints/sweep_{method}/ck{N}/` for safe resumption.

Output (after `sweep_summary`):
- **Printed SWEEP TABLE** — rows = checkpoints 1–8, columns = generation accuracy, logit accuracy, MCQ accuracy, base-probe and method-probe accuracies.
- **`checkpoints/sweep_{method}/{method}_sweep.csv`** — same data as a CSV for plotting.

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
submit_sweep.sh                            Submit checkpoint-sweep jobs (stage 4)
run_hidden_knowledge.sh                    Convenience alias for submit_pipeline.sh
slurm_base.sh                              SLURM script for stage 1 (base model)
slurm_methods.sh                           SLURM job array for stage 2 (8 methods)
slurm_summary.sh                           SLURM script for stage 3 (summary)
slurm_sweep.sh                             SLURM job array for stage 4 (64 tasks: 8 methods × 8 checkpoints)
slurm_sweep_summary.sh                     SLURM job array for sweep_summary (8 tasks, CPU-only)
checkpoints/                                 Auto-created; holds .npy, .pkl, .json model caches
checkpoints/sweep_METHOD/                    Sweep results for one method
checkpoints/sweep_METHOD/ckN/               Per-checkpoint cache: hs_{train,val,test}.npy, cyber_hs_{train,val,test}.npy, probes.pkl, cyber_probes.pkl, partial.json, results.json
data/                                        Auto-created; holds all CSV outputs
data/wmdp_tf_pairs.csv                      Cached WMDP bio train/val/test pairs (created on first run)
data/wmdp_cyber_true_false_balanced.csv     WMDP cyber True/False pairs (must be present before running)
data/summary_table1_gen_logit.csv           Table 1 CSV (bio generation + logit)
data/summary_table2_base_probes.csv         Table 2 CSV (bio base probes)
data/summary_table3_method_probes.csv       Table 3 CSV (bio method-specific probes)
data/summary_table4_cyber_gen_logit.csv     Table 4 CSV (cyber generation + logit)
data/summary_table4b_cyber_base_probes.csv  Table 4b CSV (cyber base probes)
data/summary_table4c_cyber_method_probes.csv Table 4c CSV (cyber method probes)
data/summary_table4d_cyber_cross_probes.csv  Table 4d CSV (cyber cross-probe quadrant)
data/summary_table5_cross_probes.csv        Table 5 CSV (bio cross-probe quadrant)
data/summary_table6_mcq.csv                 Table 6 CSV (MCQ direct A/B/C/D)
data/sweep_METHOD/METHOD_sweep.csv          Time-series CSV (one row per checkpoint, written by sweep_summary)
logs/                                        Auto-created; SLURM stdout/stderr
```

---

## Setup

### Requirements

- Python 3.10+
- PyTorch with CUDA (tested on L40 48 GB)
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
# Submit all three main stages with automatic job dependencies:
bash submit_pipeline.sh

# Monitor:
squeue -u $USER

# Logs:
tail -f logs/base_<JOBID>.out
tail -f logs/method_<TASK>_<JOBID>.out
tail -f logs/summary_<JOBID>.out
```

The three stages run as:
- `slurm_base.sh` — 1 × L40, up to 8 h
- `slurm_methods.sh` — 9 × L40 in parallel (job array), up to 10 h each
- `slurm_summary.sh` — CPU-only, 30 min

Stage 2 starts automatically once stage 1 succeeds; stage 3 starts once all stage-2 tasks succeed.

### Running the checkpoint sweep

```bash
# After stage 1 (base) has completed:

# Submit all 64 parallel jobs (8 methods × 8 checkpoints):
bash submit_sweep.sh

# Submit sweep + auto-submit summary+plot jobs after all 64 complete:
bash submit_sweep.sh --summary

# Submit sweep + summary, with plot options:
bash submit_sweep.sh --summary --plot_type heatmap --metric f1
bash submit_sweep.sh --summary --metric auc --probe_source base
bash submit_sweep.sh --summary --clf LR --metric accuracy --plot_type line

# One method only (8 parallel jobs):
bash submit_sweep.sh --method GradDiff

# Single (method, checkpoint) job:
bash submit_sweep.sh --method GradDiff --checkpoint 3

# Submit summary+plot array directly (after sweep jobs have finished):
sbatch --export=ALL --array=0-7 slurm_sweep_summary.sh

# Logs:
tail -f logs/sweep_<TASK>_<JOBID>.out
tail -f logs/sweep_sum_<TASK>_<JOBID>.out
```

The sweep runs as a **64-task SLURM array** (`slurm_sweep.sh`, tasks 0–63).
Each task handles one (method, checkpoint) pair independently — all 64 can run in parallel.

`slurm_sweep_summary.sh` runs two steps per method in one job:
1. Print the SWEEP TABLE and save `{method}_sweep.csv` (calls `--stage sweep_summary`)
2. Generate the layer-accuracy plot by calling `plot_layer_accuracy.py`

Plot options are forwarded via environment variables:

| `submit_sweep.sh` flag | Env var passed to SLURM | Default |
|---|---|---|
| `--plot_type line\|heatmap` | `PLOT_PLOT_TYPE` | `line` |
| `--metric accuracy\|f1\|…` | `PLOT_METRIC` | all metrics |
| `--clf LR\|RF\|AdaBoost` | `PLOT_CLF` | all |
| `--probe_source method\|base` | `PLOT_PROBE_SOURCE` | `method` |
| `--out filename` | `PLOT_OUT` | auto-generated |

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

# Checkpoint sweep (stage 4) — run after stage 1:
# Sequential (all 8 checkpoints in one process):
python hidden_knowledge_after_unlearning.py --stage sweep --method GradDiff
# Single checkpoint only:
python hidden_knowledge_after_unlearning.py --stage sweep --method GradDiff --checkpoint 3
# Print table from cached results (no GPU):
python hidden_knowledge_after_unlearning.py --stage sweep_summary --method GradDiff
```

### Resubmitting after preemption

Checkpoints are saved after each heavy operation (hidden-state extraction arrays, generation answers, logit scores).  Simply resubmit the failed job — it will resume from where it left off:

```bash
sbatch slurm_methods.sh   # or the full pipeline again
```

To start completely from scratch, delete the `checkpoints/` directory and `data/wmdp_tf_pairs.csv`.

To recompute only the **cyber** set (e.g. after changing `CYBER_TRAIN_SIZE`), delete:
- `checkpoints/base_cyber_hs_{train,val,test}.npy` and `checkpoints/base_cyber_probes.pkl`
- The `cyber_test_answers` and `cyber_logit_scores` keys from `checkpoints/base_partial.json`
- For each method: `checkpoints/{sn}_cyber_hs_{train,val,test}.npy` and `checkpoints/{sn}_cyber_probes.pkl`
Then rerun `--stage base` followed by `--stage method` for each method.

All bio hidden states, bio probes, and generation caches are preserved and reused automatically.

---

## Configuration

Key constants at the top of `hidden_knowledge_after_unlearning.py`:

| Constant | Default | Description |
|---|---|---|
| `BASE_MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | Base (un-unlearned) model |
| `WMDP_CSV_PATH` | `data/wmdp_tf_pairs.csv` | Cached WMDP bio pairs; delete to force rebuild |
| `CYBER_CSV_PATH` | `data/wmdp_cyber_true_false_balanced.csv` | WMDP cyber True/False pairs CSV |
| `UNLEARNED_MODELS` | 8 LLM-GAT checkpoints + `Llama3-8B` | Dict of method name → HF model ID; includes raw `meta-llama/Meta-Llama-3-8B` as a reference |
| `FORGET_SUBSET` | `wmdp-bio` | WMDP subset to treat as forget set |
| `TRAIN_SIZE` | 500 | Bio questions used to train probes |
| `VAL_SIZE` | 200 | Bio questions used to select best probe layer |
| `CYBER_TRAIN_SIZE` | 500 | Cyber question pairs used to train probes |
| `CYBER_VAL_SIZE` | 200 | Cyber question pairs used to select best probe layer |
| `GENERATION_BATCH_SIZE` | 8 | Batch size for text generation |
| `HIDDEN_STATE_BATCH_SIZE` | 8 | Batch size for hidden-state extraction |
| `LOGIT_BATCH_SIZE` | 16 | Batch size for logit-score computation |
| `MAX_NEW_TOKENS` | 64 | Max tokens generated per prompt |
| `PCA_DIMS_PER_LAYER` | 64 | PCA components before RF/AdaBoost per-layer probes |
| `PCA_DIMS_MULTI` | 256 | PCA components for all multi-layer probes |
| `N_SWEEP_CHECKPOINTS` | 8 | Number of training checkpoints evaluated in `--stage sweep` |

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

TABLE 4 — CYBER SET (test): Generation + Logit
Method       CyAcc  CyTrue  CyFalse   Gib  CyLogit  CLTrue  CLFalse
...

TABLE 4b — CYBER SET (test): BASE-model probes applied to cyber hidden states
TABLE 4c — CYBER SET (test): METHOD-SPECIFIC probes on cyber hidden states
TABLE 4d — CYBER SET (test): METHOD probes applied to BASE-model cyber hidden states
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
- `Cy*` / `CL*` — cyber-set generation and logit metrics
- `Gibberish` — fraction of outputs containing neither "True"/"False" (Tables 1–5) or a valid letter A–D (Table 6)
- `Acc_A/B/C/D` — per-answer-letter accuracy for MCQ questions whose correct answer is that letter

**Tables 2, 3, 5** all share the same column layout — per-layer (PL) and multi-layer (ML) results for LR, RF, and AdaBoost. The distinction is which probes are applied to which hidden states (see the 2×2 matrix above).

**Note on old checkpoints:** If a method checkpoint was created before Table 5 was added, the method stage will automatically compute the missing cross-probe stats from the saved probes and `base_hs_test.npy` without a full rerun.

### Stage 4 — Sweep output

```
SWEEP TABLE — GradDiff: unlearning evolution over 8 checkpoints
Ck  GenAcc  GTru  GFal   Gib  LogAcc  LTru  LFal  MCQAcc  MGib  BP-LR Acc True Fals  BP-RF ...
─────────────────────────────────────────────────────────────────────────────────────────────
 1   0.XXX  0.XXX 0.XXX 0.XXX  0.XXX 0.XXX 0.XXX  0.XXX 0.XXX  0.XXX 0.XXX 0.XXX  ...
 2   ...
 ...
 8   ...
```

**Column groups:**
- `GenAcc / GTru / GFal / Gib` — generation accuracy (overall / true-label / false-label / gibberish)
- `LogAcc / LTru / LFal` — logit-based accuracy (no decoding)
- `MCQAcc / MGib` — multiple-choice (A/B/C/D) accuracy and gibberish rate
- `BP-LR/RF/Ada` — base-probe per-layer accuracy for each classifier at its best validation layer
- `MP-LR/RF/Ada` — method-probe per-layer accuracy (only when `--method_probes` is passed)

The CSV (`checkpoints/sweep_<method>/<method>_sweep.csv`) also includes multi-layer, vote-ensemble, and avg-ensemble columns for each classifier and probe family.

---

## Generating plots

Plots are produced by `plot_layer_accuracy.py` (no GPU required — reads from saved `.npy` and `.pkl` files).

### Plot modes (`--mode`)

| Value | Description |
|---|---|
| `methods` (default) | Per-layer probe metric for all 10 models at their final checkpoint. One curve/row per model. |
| `checkpoints` | Per-layer probe metric for **one** unlearning method across all 8 training checkpoints, with Base (Instruct) as a reference. `--method METHOD` or `--method all` (8 files). |

### Plot types (`--plot_type`)

| Value | Description |
|---|---|
| `line` (default) | One curve per model/checkpoint, X axis = layer. |
| `heatmap` | 2-D grid: X axis = layer, Y axis = checkpoint or model, colour = metric value (red=low → green=high). One subplot per classifier. |

### Metrics (`--metric`)

If omitted, all metrics are plotted as separate rows in one figure.

| Value | Description |
|---|---|
| `accuracy` | Overall True/False classification accuracy |
| `true_accuracy` | Accuracy on True-labelled examples only |
| `false_accuracy` | Accuracy on False-labelled examples only |
| `precision` | Precision (positive class = True) |
| `recall` | Recall (positive class = True) |
| `f1` | F1 score |
| `auc` | AUC-ROC (from `predict_proba` or `decision_function`) |

### Probe source (`--probe_source`)

| Value | Description |
|---|---|
| `method` (default) | Each model/checkpoint evaluated with its **own** trained probes (Table 3) |
| `base` | Base model's probes applied to every model's hidden states (Table 2) |

### Output filenames

When `--out` is not given, a descriptive name is auto-generated:

```
{plot_type}_{mode}[_{method}]_{metric}_{clf}_{probe_source}.png
```

Examples:
```
line_methods_all_metrics_all_clf_method.png
line_checkpoints_GradDiff_f1_LR_method.png
heatmap_checkpoints_GradDiff_auc_all_clf_base.png
heatmap_methods_accuracy_all_clf_method.png
```

When `--out` is given with `--method all`, it is used as a template and the method name is appended to the stem.

### Examples

```bash
# ── Line plots ──────────────────────────────────────────────────────────────

# All models, all metrics (default):
python plot_layer_accuracy.py

# All models, F1 only, LR classifier:
python plot_layer_accuracy.py --metric f1 --clf LR

# All models, AUC-ROC, base probes:
python plot_layer_accuracy.py --metric auc --probe_source base

# One method across 8 checkpoints:
python plot_layer_accuracy.py --mode checkpoints --method GradDiff --metric f1

# All 8 methods, one PNG each (auto-named):
python plot_layer_accuracy.py --mode checkpoints --method all --metric accuracy

# ── Heatmaps ────────────────────────────────────────────────────────────────

# All models × all layers, coloured by accuracy:
python plot_layer_accuracy.py --plot_type heatmap --metric accuracy

# All models × all layers, F1, LR only:
python plot_layer_accuracy.py --plot_type heatmap --metric f1 --clf LR

# Checkpoints × layers heatmap, one method:
python plot_layer_accuracy.py --mode checkpoints --method GradDiff --plot_type heatmap --metric f1

# Checkpoints × layers, AUC-ROC, base probes:
python plot_layer_accuracy.py --mode checkpoints --method RMU --plot_type heatmap --metric auc --probe_source base

# All 8 methods, heatmap, one PNG each (auto-named):
python plot_layer_accuracy.py --mode checkpoints --method all --plot_type heatmap --metric f1
```

### SLURM checkpoint plots

`submit_plot_checkpoints.sh` submits 8 parallel CPU jobs (one per unlearning method) via `slurm_plot_checkpoints.sh`.

```bash
# Default (line, method probes, all metrics — filenames auto-generated):
bash submit_plot_checkpoints.sh

# Heatmap, F1 only:
bash submit_plot_checkpoints.sh --plot_type heatmap --metric f1

# Base probes, AUC, custom output template:
bash submit_plot_checkpoints.sh --probe_source base --metric auc --out plots/ck.png

# LR only, line plot:
bash submit_plot_checkpoints.sh --clf LR --metric accuracy
```

When `--out` is omitted, filenames are auto-generated per method:
```
line_checkpoints_GradDiff_f1_all_clf_method.png
line_checkpoints_RMU_f1_all_clf_method.png
...
```

When `--out plots/ck.png` is given, output is:
```
plots/ck_GradDiff.png
plots/ck_RMU.png
...
```
