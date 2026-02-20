# Hidden Knowledge After Unlearning

**Research question:** When a large language model is unlearned (i.e., a method like GradDiff or RMU suppresses certain knowledge from its outputs), does the knowledge disappear from the model's internal representations — or does it remain hidden in the hidden states even though the model stops verbalising it?

---

## What the code does

The experiment uses the [WMDP](https://huggingface.co/datasets/cais/wmdp) biosecurity benchmark as the **forget set** (knowledge that unlearning methods try to erase) and [WikiText-103](https://huggingface.co/datasets/wikitext) passages as the **retain set** (general knowledge that should be preserved).

### Pipeline overview

1. **Dataset preparation**
   - The WMDP-bio multiple-choice questions are split into train / val / test (500 / 200 / rest).
   - Each question is turned into two yes/no prompts: one with the *correct* answer ("Is this correct? → Yes") and one with a *wrong* answer ("Is this correct? → No").
   - For the retain set, WikiText passages are split into prefix + correct continuation, paired with a wrong continuation from a different passage.

2. **Base model evaluation** (`meta-llama/Meta-Llama-3-8B-Instruct`)
   - Extract hidden states (all transformer layers, last token) for the train / val / test forget pairs.
   - Train a **linear probe** (logistic regression) per layer to classify yes/no labels from the hidden states.
   - Pick the best layer on the validation set.
   - Run generation on the test forget set and the retain set; record yes/no accuracy.

3. **Unlearned model evaluation** (8 LLM-GAT checkpoints: GradDiff, RMU, RMU-LAT, RepNoise, ELM, RR, TAR, PB&J)
   - For each unlearned model, repeat generation and hidden-state extraction on the test set.
   - Apply the **base-model probes** (trained in step 2) to the unlearned model's hidden states.
   - If probe accuracy stays high even though generation accuracy drops, the model still internally represents the "forgotten" knowledge.

4. **Summary table** comparing generation accuracy, probe accuracy, and retain-set accuracy across all methods.

### Key insight

- **Generation accuracy** measures what the model *says* it knows.
- **Probe accuracy** measures what the model's hidden states *encode*.
- A gap between them (low generation, high probe) would be evidence of hidden residual knowledge after unlearning.

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

You will also need a Hugging Face account with access to `meta-llama/Meta-Llama-3-8B-Instruct`. Set your token:

```bash
huggingface-cli login
```

---

## Running the code

### Directly (local machine or interactive node)

```bash
python hidden_knowledge_after_unlearning.py
```

### On a SLURM cluster

```bash
sbatch run_hidden_knowledge.sh
```

The script requests 1 × A40 GPU (48 GB), 48 GB RAM, 8 CPUs, and a 12-hour time limit.
Output is written to `hidden_knowledge_<jobid>.out`.

### Checkpointing

Intermediate results are saved in `checkpoints/` after each model is processed.
If the job is interrupted and resubmitted, already-computed models are loaded from disk and skipped — only missing models are rerun.
Delete the `checkpoints/` directory to start fresh.

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
| `MAX_NEW_TOKENS` | 64 | Max tokens generated per prompt |

---

## Output

The script prints a **per-method breakdown** and a final **summary table**:

```
SUMMARY TABLE — all methods vs base  (test forget set)
================================================================================
Method        Gen Acc  Gen Yes   Gen No  Gibberish   Probe Acc   P-Yes    P-No
--------------------------------------------------------------------------------
Base            0.XXX    0.XXX    0.XXX       0.XXX       0.XXX   0.XXX   0.XXX
GradDiff        0.XXX    0.XXX    0.XXX       0.XXX       0.XXX   0.XXX   0.XXX
...
================================================================================

RETAIN SET — generation accuracy (both models should stay near 1.0)
```

- **Gen Acc**: fraction of test prompts where the model's first word (Yes/No) matches the ground truth.
- **Probe Acc**: fraction correctly classified by the linear probe from hidden states.
- **Gibberish rate**: fraction of outputs that contain neither "Yes" nor "No".
- **Retain Acc**: same as Gen Acc but on WikiText continuation prompts (should remain high for all methods).
