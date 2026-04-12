#!/usr/bin/env python3
"""
plot_layer_accuracy.py

Two plot modes (--mode):

  methods    (default)
             Per-layer probe metric for all 10 models at their final-checkpoint
             state.  X axis = layer, one curve/row per model.

  checkpoints
             Per-layer probe metric for ONE unlearning method across all 8
             training checkpoints, with the Base (Instruct) model as reference.
             X axis = layer, one curve/row per checkpoint + one for base.

             --method METHOD   which method to show (required, or "all")
             --method all      produce one graph per method (8 files)

Two plot types (--plot_type):

  line      (default) — one curve per model/checkpoint, X = layer
  heatmap   — 2-D grid: X = layer, Y = checkpoint or model, colour = metric

Two probe-source modes (--probe_source):
  method  (default / Table 3)
          Each model/checkpoint is evaluated with its own per-layer probes.

  base    (Table 2)
          The base model's probes are applied to every model's hidden states.
          Exception: Llama3-8B (methods mode) and Base (Instruct) (both modes)
          always use their own probes.

Metrics (--metric):
  accuracy | true_accuracy | false_accuracy | precision | recall | f1 | auc
  If omitted all metrics are shown as separate rows.

Cyber subset filtering (--cyber_subset, only with --dataset cyber):
  og        — full test set (no filter)
  pattern   — exclude computational/code-execution questions
  gibberish — exclude questions where the base model produced gibberish
  both      — exclude both (recommended for cleaner comparison with bio)

  The "both" and "pattern" subsets remove questions that cannot be meaningfully
  answered in True/False format, making heatmap comparisons against the Base
  (Instruct) model more meaningful.  Subset filtering requires hidden-state
  .npy files to exist (pre-computed per-layer stats in results.json are for the
  full set and cannot be re-filtered without the original arrays).

Output filenames are auto-generated from parameters when --out is not given:
  {plot_type}_{mode}[_{method}]_{metric}_{clf}_{probe_source}[_{subset}].png
  e.g.  heatmap_checkpoints_GradDiff_f1_LR_method_cyber_both.png
        line_methods_all_metrics_all_clf_base_cyber_both_relative.png

Usage:
    # --- methods / line ---
    python plot_layer_accuracy.py
    python plot_layer_accuracy.py --metric f1 --clf LR
    python plot_layer_accuracy.py --probe_source base --metric auc

    # --- methods / heatmap ---
    python plot_layer_accuracy.py --plot_type heatmap --metric accuracy
    python plot_layer_accuracy.py --plot_type heatmap --metric f1 --clf LR

    # --- methods / heatmap / cyber subset ---
    python plot_layer_accuracy.py --plot_type heatmap --dataset cyber --cyber_subset both --relative
    python plot_layer_accuracy.py --plot_type heatmap --dataset cyber --cyber_subset pattern --metric auc

    # --- checkpoints / line ---
    python plot_layer_accuracy.py --mode checkpoints --method GradDiff
    python plot_layer_accuracy.py --mode checkpoints --method RMU --probe_source base
    python plot_layer_accuracy.py --mode checkpoints --method all

    # --- checkpoints / heatmap ---
    python plot_layer_accuracy.py --mode checkpoints --method GradDiff --plot_type heatmap --metric f1
    python plot_layer_accuracy.py --mode checkpoints --method all --plot_type heatmap --metric auc

    # --- checkpoints / heatmap / cyber subset with normalization ---
    python plot_layer_accuracy.py --mode checkpoints --method GradDiff \\
        --plot_type heatmap --dataset cyber --cyber_subset both --relative
"""

import argparse
import csv
import gc
import json
import pickle
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")          # headless; switch to "TkAgg" / "Qt5Agg" for interactive
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — mirrors hidden_knowledge_after_unlearning.py
# ---------------------------------------------------------------------------

ALL_MODELS = {
    "Base (Instruct)": "base",
    "GradDiff":        "GradDiff",
    "RMU":             "RMU",
    "RMU-LAT":         "RMU-LAT",
    "RepNoise":        "RepNoise",
    "ELM":             "ELM",
    "RR":              "RR",
    "TAR":             "TAR",
    "PB&J":            "PB_J",
}

# Unlearning methods that have checkpoint sweeps (in the same order as SLURM layout)
SWEEP_METHODS   = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]
N_CHECKPOINTS   = 8

CLF_NAMES          = ["LR", "RF", "AdaBoost"]
METRIC_NAMES       = ["accuracy", "true_accuracy", "false_accuracy",
                      "precision", "recall", "f1", "auc"]
PROBE_SOURCE_NAMES = ["method", "base"]

# In base mode (methods), these models always use their own probes.
OWN_PROBE_MODELS = {"Llama3-8B"}

THICK_MODELS  = {"Base (Instruct)", "Llama3-8B"}
DASHED_MODELS = {"Llama3-8B"}

_PALETTE = [
    "#1f77b4",   # blue         — Base (Instruct)
    "#ff7f0e",   # orange       — GradDiff
    "#2ca02c",   # green        — RMU
    "#d62728",   # red          — RMU-LAT
    "#9467bd",   # purple       — RepNoise
    "#8c564b",   # brown        — ELM
    "#e377c2",   # pink         — RR
    "#7f7f7f",   # grey         — TAR
    "#bcbd22",   # yellow-green — PB&J
    "#17becf",   # cyan         — Llama3-8B
]

MODEL_COLORS = {name: _PALETTE[i] for i, name in enumerate(ALL_MODELS)}

# Checkpoint-mode colours: base = same blue as "Base (Instruct)"; ck1-ck8 = plasma gradient
_CK_PLASMA   = cm.plasma(np.linspace(0.15, 0.85, N_CHECKPOINTS))
CK_COLORS    = {"Base (Instruct)": MODEL_COLORS["Base (Instruct)"]}
CK_COLORS.update({f"ck{n}": tuple(_CK_PLASMA[n - 1]) for n in range(1, N_CHECKPOINTS + 1)})
# Ordered label list for checkpoint plots (base first, then ck1..ck8)
CK_LABELS    = ["Base (Instruct)"] + [f"ck{n}" for n in range(1, N_CHECKPOINTS + 1)]

DATA_DIR         = Path("data")    # where CSVs (wmdp_tf_pairs.csv, summary tables) live
CYBER_CSV_PATH   = DATA_DIR / "wmdp_cyber_true_false_balanced.csv"
CYBER_TRAIN_SIZE = 500
CYBER_VAL_SIZE   = 200
RANDOM_SEED      = 42

# Cyber subsets (mirrors hidden_knowledge_after_unlearning.py)
CYBER_SUBSETS = ["og", "pattern", "gibberish", "both"]

# Questions requiring code execution / computation — excluded from "pattern" and "both" subsets
import re as _re
_COMP_PATTERNS = [_re.compile(p) for p in [
    r'Which arguments will cause',
    r'What is the output of the following',
    r'extern "C"',
    r'following.*assembly function',
    r'following ve assembly',
]]

def _is_computational(question: str) -> bool:
    return any(p.search(question) for p in _COMP_PATTERNS)

Y_MIN    = 0.45   # fixed lower bound of y-axis
Y_PAD    = 0.05   # padding fraction above the highest point
N_LAYERS = 32     # transformer layers (indices 1-32; layer 0 = embedding, skipped)

METRIC_LABELS = {
    "accuracy":       "Accuracy (overall)",
    "true_accuracy":  "True-label accuracy",
    "false_accuracy": "False-label accuracy",
    "precision":      "Precision",
    "recall":         "Recall",
    "f1":             "F1 Score",
    "auc":            "AUC-ROC",
}

PROBE_SOURCE_TITLE_TEMPLATE = {
    "method": "Per-Layer Probe {metric} — each model's own probes",
    "base":   "Per-Layer Probe {metric} — base probes applied to all models",
}

def probe_source_title(probe_source: str, metrics: list) -> str:
    metric_label = METRIC_LABELS[metrics[0]] if len(metrics) == 1 else "Metrics"
    return PROBE_SOURCE_TITLE_TEMPLATE[probe_source].format(metric=metric_label)

# ---------------------------------------------------------------------------
# External-logit reference line config
# ---------------------------------------------------------------------------

# Llama-3-70B-Instruct appears only as a dotted reference line (no probe curve).
LLAMA70B_LABEL = "Llama-3-70B-Instruct"
LLAMA70B_COLOR = "#000000"   # black — distinct from all curve colours

# Legend section 1: solid probe curves (Llama3-70B has no curve, so excluded)
LEGEND_CURVE_ORDER = [
    "Base (Instruct)", "Llama3-8B",
    "GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J",
]

# Legend section 2: dotted external-logit lines (70B inserted between 8B models)
LEGEND_LOGIT_ORDER = [
    "Base (Instruct)", LLAMA70B_LABEL, "Llama3-8B",
    "GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J",
]

# Checkpoint-mode logit legend: base, 70B reference, then ck1..ck8
CK_LOGIT_ORDER = ["Base (Instruct)", LLAMA70B_LABEL] + [
    f"ck{n}" for n in range(1, N_CHECKPOINTS + 1)
]


def _logit_color(name: str) -> str:
    """Return the colour to use for an external-logit reference line."""
    if name == LLAMA70B_LABEL:
        return LLAMA70B_COLOR
    if name in CK_COLORS:          # checkpoint-mode labels (ck1..ck8, Base)
        return CK_COLORS[name]
    return MODEL_COLORS.get(name, LLAMA70B_COLOR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Mirror of hidden_knowledge_after_unlearning.safe_name()."""
    return name.replace("&", "_").replace("/", "_").replace(" ", "_")


def _ck_dir(method_name: str, ck_num: int, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"sweep_{_safe_name(method_name)}" / f"ck{ck_num}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_y_test(csv_path: Path) -> np.ndarray:
    labels = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                labels.append(1 if row["label"] == "True" else 0)
    if not labels:
        raise RuntimeError(f"No test-split rows found in {csv_path}")
    return np.array(labels, dtype=np.int32)


def load_cyber_y_test_methods(checkpoint_dir: Path) -> np.ndarray:
    """Load cyber y_test for methods mode.

    Prefers base_cyber_y_test.npy (written by run_base()).  Falls back to
    deriving labels from the CSV with the same RNG seed — produces identical
    results because both load_cyber_tf_pairs(rng) and load_cyber_test_pairs_sweep()
    use random.Random(RANDOM_SEED=42).
    """
    path = checkpoint_dir / "base_cyber_y_test.npy"
    if path.exists():
        return np.load(path).astype(np.int32)
    print("  [warn] base_cyber_y_test.npy not found; deriving labels from CSV (same RNG seed).")
    return load_cyber_y_test_sweep()


def load_cyber_test_pairs_sweep() -> list:
    """Return cyber test pairs in the exact shuffled order used by _sweep_load_shared().

    Replicates load_cyber_tf_pairs(rng) with a fresh random.Random(RANDOM_SEED):
      1. Shuffle question_ids
      2. Advance rng past train-pair shuffle and val-pair shuffle
      3. Shuffle test pairs
    Returns list of dicts: {question_id, question, label (1=True/0=False)}.
    """
    if not CYBER_CSV_PATH.exists():
        raise FileNotFoundError(f"Cyber CSV not found at {CYBER_CSV_PATH}")
    by_qid: dict = {}
    with open(CYBER_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row["question_id"]
            if qid not in by_qid:
                by_qid[qid] = {}
            by_qid[qid][row["label"]] = row

    question_ids = list(by_qid.keys())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(question_ids)
    train_ids = question_ids[:CYBER_TRAIN_SIZE]
    val_ids   = question_ids[CYBER_TRAIN_SIZE:CYBER_TRAIN_SIZE + CYBER_VAL_SIZE]
    test_ids  = question_ids[CYBER_TRAIN_SIZE + CYBER_VAL_SIZE:]

    def _make(ids):
        pairs = []
        for qid in ids:
            g = by_qid[qid]
            if "True" not in g or "False" not in g:
                continue
            question = g["True"]["question"]
            pairs.append({"question_id": qid, "question": question, "label": 1})
            pairs.append({"question_id": qid, "question": question, "label": 0})
        rng.shuffle(pairs)
        return pairs

    _make(train_ids)   # advance rng past train-pair shuffle
    _make(val_ids)     # advance rng past val-pair shuffle
    return _make(test_ids)


def load_cyber_y_test_sweep() -> np.ndarray:
    """Derive cyber y_test for checkpoints/sweep mode (correctly applies pair shuffle)."""
    pairs = load_cyber_test_pairs_sweep()
    return np.array([p["label"] for p in pairs], dtype=np.int32)


def load_cyber_test_meta_methods(checkpoint_dir: Path) -> list | None:
    """Load per-pair metadata saved by run_base() for methods-mode subset filtering.

    Returns list of {question_id, is_comp} dicts in the same order as
    base_cyber_hs_test.npy, or None if the file hasn't been written yet
    (run --stage base to generate it).
    """
    path = checkpoint_dir / "base_cyber_test_meta.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_base_cyber_gib_qids(checkpoint_dir: Path) -> list:
    """Load cyber gibberish question-IDs from base_results.json."""
    path = checkpoint_dir / "base_results.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("cyber_gibberish_qids") or []
    except Exception:
        return []


def compute_cyber_subset_mask(subset_name: str, pairs_meta: list,
                               gib_qids) -> np.ndarray:
    """Return boolean mask for the named cyber subset.

    pairs_meta: list of dicts with keys 'question_id' and optionally 'is_comp'.
    gib_qids:   iterable of question_ids where the base model gave gibberish.

    Subsets:
      og        — all pairs (no filter)
      pattern   — exclude computational questions
      gibberish — exclude questions where base model gave gibberish
      both      — exclude computational AND base-gibberish questions
    """
    gib_set   = set(gib_qids or [])
    comp_excl = subset_name in ("pattern", "both")
    gib_excl  = subset_name in ("gibberish", "both")

    mask = []
    for m in pairs_meta:
        exclude = False
        if comp_excl and m.get("is_comp", False):
            exclude = True
        if gib_excl and m["question_id"] in gib_set:
            exclude = True
        mask.append(not exclude)
    return np.array(mask, dtype=bool)


def load_hs(sn: str, checkpoint_dir: Path, dataset: str = "bio"):
    fname = f"{sn}_cyber_hs_test.npy" if dataset == "cyber" else f"{sn}_hs_test.npy"
    path = checkpoint_dir / fname
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    return np.load(path)


def load_probe_set(sn: str, checkpoint_dir: Path, dataset: str = "bio"):
    fname = f"{sn}_cyber_probes.pkl" if dataset == "cyber" else f"{sn}_probes.pkl"
    path = checkpoint_dir / fname
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    with open(path, "rb") as f:
        ps = pickle.load(f)
    if not isinstance(ps, dict) or "per_layer" not in ps:
        print(f"  [skip] {path.name}: old probe format")
        return None
    return ps


def load_hs_ck(method_name: str, ck_num: int, checkpoint_dir: Path,
               dataset: str = "bio"):
    """Load hidden states for a sweep checkpoint."""
    fname = "cyber_hs_test.npy" if dataset == "cyber" else "hs_test.npy"
    path = _ck_dir(method_name, ck_num, checkpoint_dir) / fname
    if not path.exists():
        print(f"  [skip] sweep/{_safe_name(method_name)}/ck{ck_num}/{fname} not found")
        return None
    return np.load(path)


def load_probe_set_ck(method_name: str, ck_num: int, checkpoint_dir: Path,
                      dataset: str = "bio"):
    """Load probe set for a sweep checkpoint."""
    fname = "cyber_probes.pkl" if dataset == "cyber" else "probes.pkl"
    path = _ck_dir(method_name, ck_num, checkpoint_dir) / fname
    if not path.exists():
        print(f"  [skip] sweep/{_safe_name(method_name)}/ck{ck_num}/{fname} not found")
        return None
    with open(path, "rb") as f:
        ps = pickle.load(f)
    if not isinstance(ps, dict) or "per_layer" not in ps:
        print(f"  [skip] sweep/ck{ck_num}/{fname}: old probe format")
        return None
    return ps


# ---------------------------------------------------------------------------
# External logit value loading
# ---------------------------------------------------------------------------

# Map each model display name → (results filename, logit_stats key in that JSON)
# Bio: base uses "logit_stats"; method checkpoints use "logit"; 70B uses "bio_logit_stats".
_LOGIT_SOURCES_BIO = {
    "Base (Instruct)": ("base_results.json",      "logit_stats"),
    "GradDiff":        ("GradDiff_results.json",   "logit"),
    "RMU":             ("RMU_results.json",         "logit"),
    "RMU-LAT":         ("RMU-LAT_results.json",     "logit"),
    "RepNoise":        ("RepNoise_results.json",     "logit"),
    "ELM":             ("ELM_results.json",          "logit"),
    "RR":              ("RR_results.json",           "logit"),
    "TAR":             ("TAR_results.json",          "logit"),
    "PB&J":            ("PB_J_results.json",         "logit"),
    "Llama3-8B":       ("Llama3-8B_results.json",   "logit"),
    LLAMA70B_LABEL:    ("llama70b_results.json",     "bio_logit_stats"),
}

# Cyber: only 70B has cyber_logit_stats; base/methods do not store it pre-computed.
_LOGIT_SOURCES_CYBER = {
    LLAMA70B_LABEL:    ("llama70b_results.json",     "cyber_logit_stats"),
}


def load_external_logit(checkpoint_dir: Path, metrics: list,
                        dataset: str = "bio") -> dict:
    """
    Return ext[metric][model_name] = float for all available (metric, model) pairs.
    Reads *_results.json; missing files / keys are silently skipped.
    """
    sources = _LOGIT_SOURCES_CYBER if dataset == "cyber" else _LOGIT_SOURCES_BIO
    ext = {m: {} for m in metrics}
    for name, (fname, stats_key) in sources.items():
        path = checkpoint_dir / fname
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            stats = data.get(stats_key) or {}
            for m in metrics:
                val = stats.get(m)
                if val is not None:
                    ext[m][name] = float(val)
        except Exception:
            pass
    return ext


def load_external_logit_checkpoints(checkpoint_dir: Path,
                                     method_name: str,
                                     metrics: list,
                                     dataset: str = "bio") -> dict:
    """
    Return ext[metric][label] = float for checkpoints mode.
    Labels: "Base (Instruct)", LLAMA70B_LABEL, "ck1".."ck8".
    """
    ext = {m: {} for m in metrics}

    def _read(path, stats_key):
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f).get(stats_key) or {}
        except Exception:
            return {}

    if dataset == "cyber":
        # Base (Instruct) — no pre-computed cyber_logit_stats in base_results.json; skip.
        # Llama-3-70B-Instruct — constant reference line
        for m in metrics:
            val = _read(checkpoint_dir / "llama70b_results.json", "cyber_logit_stats").get(m)
            if val is not None:
                ext[m][LLAMA70B_LABEL] = float(val)
        # Sweep checkpoints — key is "cyber_logit"
        for ck_num in range(1, N_CHECKPOINTS + 1):
            label   = f"ck{ck_num}"
            ck_path = _ck_dir(method_name, ck_num, checkpoint_dir) / "results.json"
            stats   = _read(ck_path, "cyber_logit")
            for m in metrics:
                val = stats.get(m)
                if val is not None:
                    ext[m][label] = float(val)
    else:
        # Base (Instruct)
        for m in metrics:
            val = _read(checkpoint_dir / "base_results.json", "logit_stats").get(m)
            if val is not None:
                ext[m]["Base (Instruct)"] = float(val)
        # Llama-3-70B-Instruct — constant reference line
        for m in metrics:
            val = _read(checkpoint_dir / "llama70b_results.json", "bio_logit_stats").get(m)
            if val is not None:
                ext[m][LLAMA70B_LABEL] = float(val)
        # Sweep checkpoints — key is "logit" (not "logit_stats")
        for ck_num in range(1, N_CHECKPOINTS + 1):
            label   = f"ck{ck_num}"
            ck_path = _ck_dir(method_name, ck_num, checkpoint_dir) / "results.json"
            stats   = _read(ck_path, "logit")
            for m in metrics:
                val = stats.get(m)
                if val is not None:
                    ext[m][label] = float(val)

    return ext


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def per_layer_metric(probe_set: dict, hs_test: np.ndarray,
                     y_test: np.ndarray, clf_name: str,
                     metric: str, layer_start: int = 1) -> list:
    """
    Return [(layer_idx, value), ...] for layers layer_start .. n_layers-1.
    probe_set and hs_test may come from different models.
    Supported metrics: accuracy, true_accuracy, false_accuracy,
                       precision, recall, f1, auc.
    """
    n_layers     = hs_test.shape[1]
    layer_probes = probe_set["per_layer"].get(clf_name, {})
    true_mask    = y_test == 1
    false_mask   = y_test == 0
    result = []
    for l in range(layer_start, n_layers):
        pipe = layer_probes.get(l)
        if pipe is None:
            continue
        X = hs_test[:, l, :]
        preds = pipe.predict(X)
        if metric == "accuracy":
            val = float((preds == y_test).mean())
        elif metric == "true_accuracy":
            val = float((preds[true_mask] == 1).mean()) if true_mask.any() else 0.0
        elif metric == "false_accuracy":
            val = float((preds[false_mask] == 0).mean()) if false_mask.any() else 0.0
        elif metric == "precision":
            val = float(precision_score(y_test, preds, zero_division=0))
        elif metric == "recall":
            val = float(recall_score(y_test, preds, zero_division=0))
        elif metric == "f1":
            val = float(f1_score(y_test, preds, zero_division=0))
        elif metric == "auc":
            try:
                if hasattr(pipe, "predict_proba"):
                    scores = pipe.predict_proba(X)[:, 1]
                else:
                    scores = pipe.decision_function(X)
                val = float(roc_auc_score(y_test, scores))
            except Exception:
                val = 0.0
        else:
            val = 0.0
        result.append((l, val))
    return result


# ---------------------------------------------------------------------------
# Data collection — methods mode (loads each model's hs once for all metrics)
# ---------------------------------------------------------------------------

def collect_data(checkpoint_dir: Path, clfs: list,
                 y_test: np.ndarray, metrics: list,
                 probe_source: str, dataset: str = "bio",
                 mask: np.ndarray | None = None) -> dict:
    """
    Returns data[metric][clf_name][model_name] = (layers_list, values_list).

    probe_source == "method":
        Each model's probes on its own hidden states.
    probe_source == "base":
        Base model's probes on every model's hidden states, EXCEPT models in
        OWN_PROBE_MODELS (currently Llama3-8B) which use their own probes.

    mask: optional boolean array of shape (n_test,).  When supplied, only the
        masked-True examples are evaluated (used for cyber subset filtering).
    """
    data = {m: {clf: {} for clf in clfs} for m in metrics}

    base_probe_set = None
    if probe_source == "base":
        base_probe_set = load_probe_set("base", checkpoint_dir, dataset)
        if base_probe_set is None:
            probe_label = "base_cyber_probes.pkl" if dataset == "cyber" else "base_probes.pkl"
            raise RuntimeError(f"{probe_label} not found — run --stage base first.")
        print("  Base probe set loaded.")

    for model_name, sn in ALL_MODELS.items():
        print(f"  Loading {model_name} ({sn}) ...")

        hs = load_hs(sn, checkpoint_dir, dataset)
        if hs is None:
            continue

        # Apply subset mask when requested.
        hs_eff = hs[mask] if mask is not None else hs
        y_eff  = y_test[mask] if mask is not None else y_test

        # Determine which probe set to use for this model.
        if probe_source == "method" or model_name in OWN_PROBE_MODELS:
            probe_set = load_probe_set(sn, checkpoint_dir, dataset)
            if probe_set is None:
                continue
        else:
            probe_set = base_probe_set   # already loaded above

        for metric in metrics:
            for clf_name in clfs:
                pairs = per_layer_metric(probe_set, hs_eff, y_eff,
                                         clf_name, metric=metric, layer_start=1)
                if pairs:
                    layers, vals = zip(*pairs)
                    data[metric][clf_name][model_name] = (list(layers), list(vals))

        del hs
        gc.collect()

    return data


# ---------------------------------------------------------------------------
# Data collection — checkpoints mode
# ---------------------------------------------------------------------------

def collect_data_checkpoints(checkpoint_dir: Path, method_name: str, clfs: list,
                              y_test: np.ndarray, metrics: list,
                              probe_source: str, dataset: str = "bio",
                              mask: np.ndarray | None = None) -> dict:
    """
    Returns data[metric][clf_name][label] = (layers_list, values_list)
    where label is "Base (Instruct)", "ck1", ..., "ck8".

    probe_source == "method":
        Base model uses base_probes; each checkpoint uses its own probes.pkl.
    probe_source == "base":
        All checkpoints are evaluated with the base model's probes.
        Base (Instruct) still uses its own probes.

    mask: optional boolean array of shape (n_test,).  When supplied, only the
        masked-True examples are evaluated (used for cyber subset filtering).
        Note: when mask is active the fallback (pre-computed per-layer stats in
        results.json) is skipped because those stats were computed on the full set.
    """
    data = {m: {clf: {} for clf in clfs} for m in metrics}

    # Always load base probe set (needed for Base (Instruct) curve and probe_source=base)
    base_probe_set = load_probe_set("base", checkpoint_dir, dataset)
    if base_probe_set is None:
        probe_label = "base_cyber_probes.pkl" if dataset == "cyber" else "base_probes.pkl"
        raise RuntimeError(f"{probe_label} not found — run --stage base first.")

    # --- Base (Instruct) — always own probes ---
    # For cyber, base_cyber_hs_test.npy was saved by run_base() with its own
    # rng state, which may differ from the sweep rng used for y_test here.
    # Load base_cyber_y_test.npy directly so labels align with the .npy file.
    print("  Loading Base (Instruct) ...")
    base_hs = load_hs("base", checkpoint_dir, dataset)
    if base_hs is not None:
        if dataset == "cyber":
            base_y_test = load_cyber_y_test_methods(checkpoint_dir)
        else:
            base_y_test = y_test
        base_hs_eff = base_hs[mask] if mask is not None else base_hs
        y_eff       = base_y_test[mask] if mask is not None else base_y_test
        for metric in metrics:
            for clf_name in clfs:
                pairs = per_layer_metric(base_probe_set, base_hs_eff, y_eff,
                                         clf_name, metric=metric, layer_start=1)
                if pairs:
                    layers, vals = zip(*pairs)
                    data[metric][clf_name]["Base (Instruct)"] = (list(layers), list(vals))
        del base_hs
        gc.collect()

    # --- Sweep checkpoints — one at a time to keep peak memory low ---
    ck_probe_set = base_probe_set if probe_source == "base" else None

    if dataset == "cyber":
        json_base_key   = "cyber_all_layers_base_probe_stats"
        json_method_key = "cyber_all_layers_method_probe_stats"
    else:
        json_base_key   = "all_layers_base_probe_stats"
        json_method_key = "all_layers_method_probe_stats"

    for ck_num in range(1, N_CHECKPOINTS + 1):
        label    = f"ck{ck_num}"
        json_key = json_base_key if probe_source == "base" else json_method_key
        print(f"  Loading {method_name} {label} ...")

        hs = load_hs_ck(method_name, ck_num, checkpoint_dir, dataset)

        if hs is None:
            if mask is not None:
                # Pre-computed stats were computed on the full set — can't apply mask.
                print(f"    [skip] No hs file for {label}; subset mask requires hs.")
                continue
            # Fall back to pre-computed per-layer stats stored in results.json.
            ck_path = _ck_dir(method_name, ck_num, checkpoint_dir) / "results.json"
            if ck_path.exists():
                try:
                    with open(ck_path, encoding="utf-8") as f:
                        ck_r = json.load(f)
                    all_layers = ck_r.get(json_key, {})
                    for metric in metrics:
                        for clf_name in clfs:
                            clf_layers = all_layers.get(clf_name, {})
                            if not clf_layers:
                                continue
                            pairs_sorted = sorted((int(l), v.get(metric, 0.0))
                                                   for l, v in clf_layers.items()
                                                   if int(l) >= 1)
                            if pairs_sorted:
                                ls, vs = zip(*pairs_sorted)
                                data[metric][clf_name][label] = (list(ls), list(vs))
                    print(f"    [fallback] loaded per-layer data from {ck_path.name}")
                except Exception as e:
                    print(f"    [fallback] failed to load {ck_path}: {e}")
            continue

        if probe_source == "method":
            ck_probe_set = load_probe_set_ck(method_name, ck_num, checkpoint_dir, dataset)
            if ck_probe_set is None:
                del hs
                gc.collect()
                continue

        hs_eff = hs[mask] if mask is not None else hs
        y_eff  = y_test[mask] if mask is not None else y_test

        for metric in metrics:
            for clf_name in clfs:
                pairs = per_layer_metric(ck_probe_set, hs_eff, y_eff,
                                         clf_name, metric=metric, layer_start=1)
                if pairs:
                    layers, vals = zip(*pairs)
                    data[metric][clf_name][label] = (list(layers), list(vals))

        # Free the large hidden-state array immediately — only scalars are kept.
        del hs
        gc.collect()

    return data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_ax(ax, row_idx, ax_idx, n_rows, clf_name, metric):
    if row_idx == 0 and clf_name is not None:
        ax.set_title(clf_name, fontsize=9)
    if ax_idx == 0:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=8)
    if row_idx == n_rows - 1:
        ax.set_xlabel("Layer", fontsize=8)
    ax.set_xlim(0.5, N_LAYERS + 0.5)
    ax.set_xticks(range(1, N_LAYERS + 1))
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, zorder=1)


def _finalize_figure(fig, axes, legend_handles, legend_labels, n_clfs, out_path,
                     legend_outside=False):
    legend_handles.append(
        mlines.Line2D([], [], color="gray", linewidth=1.2,
                      linestyle=":", label="Chance (0.5)")
    )
    legend_labels.append("Chance (0.5)")

    if legend_outside:
        fig.subplots_adjust(
            left=0.07, right=0.97,
            bottom=0.20, top=0.92,
            hspace=0.35,
        )
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.07),
            ncol=4,
            fontsize=7,
            framealpha=0.9,
            title="Model",
            title_fontsize=7,
        )
    else:
        fig.subplots_adjust(
            left=0.07, right=0.97,
            bottom=0.07, top=0.94,
            hspace=0.35,
        )
        last_ax = axes[-1][0]
        last_ax.legend(
            legend_handles, legend_labels,
            loc="lower left",
            ncol=4,
            fontsize=7,
            framealpha=0.9,
            title="Model",
            title_fontsize=7,
        )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved -> {out_path}")


# ---------------------------------------------------------------------------
# Auto filename
# ---------------------------------------------------------------------------

def _auto_out_path(plot_type: str, mode: str, method: str | None,
                   metric: str | None, clf: str | None,
                   probe_source: str, dataset: str = "bio",
                   normalize: bool = False,
                   cyber_subset: str | None = None) -> Path:
    """
    Build a descriptive output filename from the run parameters.
    Example: heatmap_checkpoints_GradDiff_f1_LR_method_cyber_both.png
    """
    parts = [plot_type, mode]
    if mode == "checkpoints" and method and method != "all":
        parts.append(_safe_name(method))
    parts.append(_safe_name(metric) if metric else "all_metrics")
    parts.append(clf.lower() if clf else "all_clf")
    parts.append(probe_source)
    parts.append(dataset)
    if cyber_subset and dataset == "cyber":
        parts.append(cyber_subset)
    if normalize:
        parts.append("relative")
    return Path("_".join(parts) + ".png")


# ---------------------------------------------------------------------------
# Heatmap renderer (shared by both modes)
# ---------------------------------------------------------------------------

def _data_range(data: dict, metrics: list, clfs: list):
    """Return (vmin, vmax) of all non-NaN values in a data dict."""
    all_vals = []
    for m in metrics:
        for clf in clfs:
            for _label, (_layers, vals) in data[m].get(clf, {}).items():
                all_vals.extend(v for v in vals if not np.isnan(v))
    if not all_vals:
        return 0.0, 1.0
    return float(np.min(all_vals)), float(np.max(all_vals))


def _relative_data_range(data: dict, metrics: list, clfs: list):
    """Return (-max_abs, max_abs) of delta values (value - base) for relative heatmap."""
    # Gather base (Instruct) values per (metric, clf, layer)
    base_vals = {}
    for m in metrics:
        for clf in clfs:
            clf_data = data[m].get(clf, {})
            if "Base (Instruct)" in clf_data:
                layers, vals = clf_data["Base (Instruct)"]
                for l, v in zip(layers, vals):
                    base_vals[(m, clf, l)] = v

    deltas = []
    for m in metrics:
        for clf in clfs:
            clf_data = data[m].get(clf, {})
            for label, (layers, vals) in clf_data.items():
                for l, v in zip(layers, vals):
                    bv = base_vals.get((m, clf, l))
                    if bv is not None and not np.isnan(bv):
                        deltas.append(v - bv)

    if not deltas:
        return -0.1, 0.1
    max_abs = max(abs(min(deltas)), abs(max(deltas)))
    return (-max_abs or -0.01), (max_abs or 0.01)


def _render_heatmap(data: dict, metrics_to_plot: list, clfs: list,
                    row_labels: list, out_path: Path, title: str,
                    vmin: float | None = None, vmax: float | None = None,
                    normalize: bool = False):
    """
    Render a heatmap figure.

    Layout: rows = metrics_to_plot, cols = clfs.
    Each cell: X = layer (1..N_LAYERS), Y = row_labels (checkpoints or models),
               colour = metric value (RdYlGn).

    vmin / vmax: colour scale bounds.  If None, computed from this figure's data.
    Pass explicit values when multiple figures must share a common scale
    (e.g. --method all).
    """
    n_rows_fig = len(metrics_to_plot)
    n_clfs     = len(clfs)

    # Compute colour scale if not supplied
    if vmin is None or vmax is None:
        _vmin, _vmax = _data_range(data, metrics_to_plot, clfs)
        if vmin is None:
            vmin = _vmin
        if vmax is None:
            vmax = _vmax

    fig, axes = plt.subplots(
        n_rows_fig, n_clfs,
        figsize=(max(13, 6 * n_clfs), max(5.5, 3.5 * n_rows_fig)),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=11, x=0.5, ha="center")

    for row_idx, metric in enumerate(metrics_to_plot):
        for ax_idx, clf_name in enumerate(clfs):
            ax = axes[row_idx][ax_idx]
            clf_data = data[metric].get(clf_name, {})

            # Rows present for this clf (preserve declared order)
            present = [lb for lb in row_labels if lb in clf_data]
            mat = np.full((len(present), N_LAYERS), np.nan)
            for ri, label in enumerate(present):
                layers, vals = clf_data[label]
                for l, v in zip(layers, vals):
                    if 1 <= l <= N_LAYERS:
                        mat[ri, l - 1] = v

            if normalize:
                # Subtract Base (Instruct) row to show delta from base
                base_row = None
                if "Base (Instruct)" in present:
                    base_idx = present.index("Base (Instruct)")
                    base_row = mat[base_idx, :].copy()
                if base_row is not None:
                    mat = mat - base_row[np.newaxis, :]
                cmap_use = "RdYlGn"
                # Use symmetric colour scale around 0
                local_max = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 0.1
                _vmin = -(local_max or 0.01) if vmin is None else vmin
                _vmax =  (local_max or 0.01) if vmax is None else vmax
            else:
                cmap_use = "RdYlGn"
                _vmin, _vmax = vmin, vmax

            im = ax.imshow(
                mat, aspect="auto", origin="upper",
                vmin=_vmin, vmax=_vmax, cmap=cmap_use,
                interpolation="nearest",
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Titles / labels
            if row_idx == 0:
                ax.set_title(clf_name, fontsize=11)
            if ax_idx == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            if row_idx == n_rows_fig - 1:
                ax.set_xlabel("Layer", fontsize=10)
            tick_pos = list(range(0, N_LAYERS))
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([t + 1 for t in tick_pos], fontsize=5, rotation=90)

            ax.set_yticks(range(len(present)))
            display_labels = [lb.replace("Base (Instruct)", "Base") for lb in present]
            ax.set_yticklabels(display_labels, fontsize=8)

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.10, top=0.92, hspace=0.3, wspace=0.25)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot — methods mode (original)
# ---------------------------------------------------------------------------

def make_plot(checkpoint_dir: Path, out_path: Path,
              clf_filter: list, metric: str | None, probe_source: str,
              data_dir: Path = DATA_DIR, dataset: str = "bio",
              cyber_subset: str | None = None):

    if dataset == "cyber":
        print("Loading cyber y_test ...")
        y_test = load_cyber_y_test_methods(checkpoint_dir)
    else:
        csv_path = data_dir / "wmdp_tf_pairs.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"WMDP CSV not found at {csv_path}. "
                "Run --stage base first to generate it."
            )
        print("Loading y_test from CSV ...")
        y_test = load_y_test(csv_path)
    print(f"  Test set size: {len(y_test)}  "
          f"(pos={y_test.sum()}  neg={(y_test==0).sum()})")

    mask = None
    if dataset == "cyber" and cyber_subset is not None:
        meta = load_cyber_test_meta_methods(checkpoint_dir)
        if meta is None:
            print("  [warn] base_cyber_test_meta.json not found; ignoring --cyber_subset."
                  " Re-run --stage base to generate it.")
        else:
            gib_qids = load_base_cyber_gib_qids(checkpoint_dir)
            mask = compute_cyber_subset_mask(cyber_subset, meta, gib_qids)
            n_kept = mask.sum()
            print(f"  Cyber subset '{cyber_subset}': {n_kept}/{len(mask)} pairs kept.")

    clfs            = clf_filter if clf_filter else CLF_NAMES
    metrics_to_plot = [metric] if metric else METRIC_NAMES

    data      = collect_data(checkpoint_dir, clfs, y_test, metrics_to_plot,
                             probe_source, dataset, mask=mask)
    ext_logit = load_external_logit(checkpoint_dir, metrics_to_plot, dataset)

    row_ymax = {}
    for m in metrics_to_plot:
        mx = Y_MIN
        for clf_data in data[m].values():
            for _, vals in clf_data.values():
                mx = max(mx, max(vals))
        row_ymax[m] = min(mx * (1 + Y_PAD), 1.0)

    n_rows = len(metrics_to_plot)
    n_clfs = len(clfs)

    fig, axes = plt.subplots(
        n_rows, n_clfs,
        figsize=(max(13, 7 * n_clfs), max(5.5, 4.5 * n_rows)),
        sharey="row",
        squeeze=False,
    )
    base_title = probe_source_title(probe_source, metrics_to_plot)
    if n_clfs == 1:
        fig.suptitle(f"{base_title} ({clfs[0]})", fontsize=12, x=0.5, ha="center")
    else:
        fig.suptitle(base_title, fontsize=11, x=0.5, ha="center")

    for row_idx, m in enumerate(metrics_to_plot):
        y_max = row_ymax[m]

        for ax_idx, clf_name in enumerate(clfs):
            ax = axes[row_idx][ax_idx]
            _setup_ax(ax, row_idx, ax_idx, n_rows, clf_name if n_clfs > 1 else None, m)
            ax.set_ylim(Y_MIN, y_max)

            clf_data    = data[m].get(clf_name, {})
            any_plotted = False

            for model_name in ALL_MODELS:      # consistent z-ordering
                if model_name not in clf_data:
                    continue
                layers, vals = clf_data[model_name]
                color  = MODEL_COLORS[model_name]
                lw     = 2.4 if model_name in THICK_MODELS else 1.2
                ls     = "--" if model_name in DASHED_MODELS else "-"
                zorder = 4 if model_name in THICK_MODELS else 2
                ax.plot(layers, vals,
                        color=color, linewidth=lw, linestyle=ls, zorder=zorder)
                any_plotted = True

            # External logit dotted reference lines — disabled

            if not any_plotted:
                ax.text(0.5, 0.5, "No data",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)

    # ── Legend: section 1 — solid probe curves ────────────────────────────────
    legend_handles = []
    legend_labels  = []

    models_with_data = set()
    for m in metrics_to_plot:
        for clf in clfs:
            models_with_data.update(data[m].get(clf, {}).keys())

    for model_name in LEGEND_CURVE_ORDER:
        if model_name not in models_with_data:
            continue
        color = MODEL_COLORS[model_name]
        lw    = 2.4 if model_name in THICK_MODELS else 1.2
        ls    = "--" if model_name in DASHED_MODELS else "-"
        legend_handles.append(
            mlines.Line2D([], [], color=color, linewidth=lw, linestyle=ls,
                          label=model_name)
        )
        legend_labels.append(model_name)

    # Legend: section 2 — dotted external-logit lines — disabled

    _finalize_figure(fig, axes, legend_handles, legend_labels, n_clfs, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot — checkpoints mode (one method across training checkpoints)
# ---------------------------------------------------------------------------

def make_plot_checkpoints(checkpoint_dir: Path, out_path: Path,
                           method_name: str,
                           clf_filter: list, metric: str | None,
                           probe_source: str,
                           data_dir: Path = DATA_DIR, dataset: str = "bio",
                           cyber_subset: str | None = None):

    print(f"\n=== Checkpoints mode: {method_name} ({dataset}) ===")
    if dataset == "cyber":
        print("Loading cyber y_test ...")
        y_test = load_cyber_y_test_sweep()
    else:
        csv_path = data_dir / "wmdp_tf_pairs.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"WMDP CSV not found at {csv_path}. "
                "Run --stage base first to generate it."
            )
        print("Loading y_test from CSV ...")
        y_test = load_y_test(csv_path)
    print(f"  Test set size: {len(y_test)}  "
          f"(pos={y_test.sum()}  neg={(y_test==0).sum()})")

    mask = None
    if dataset == "cyber" and cyber_subset is not None:
        pairs_meta = load_cyber_test_pairs_sweep()
        gib_qids   = load_base_cyber_gib_qids(checkpoint_dir)
        mask = compute_cyber_subset_mask(cyber_subset, pairs_meta, gib_qids)
        print(f"  Cyber subset '{cyber_subset}': {mask.sum()}/{len(mask)} pairs kept.")

    clfs            = clf_filter if clf_filter else CLF_NAMES
    metrics_to_plot = [metric] if metric else METRIC_NAMES

    data = collect_data_checkpoints(
        checkpoint_dir, method_name, clfs, y_test, metrics_to_plot,
        probe_source, dataset, mask=mask
    )

    row_ymax = {}
    for m in metrics_to_plot:
        mx = Y_MIN
        for clf_data in data[m].values():
            for _, vals in clf_data.values():
                mx = max(mx, max(vals))
        row_ymax[m] = min(mx * (1 + Y_PAD), 1.0)

    n_rows = len(metrics_to_plot)
    n_clfs = len(clfs)

    probe_lbl = "Method Probes" if probe_source == "method" else "Base Probes"
    fig, axes = plt.subplots(
        n_rows, n_clfs,
        figsize=(7 * n_clfs, 4.5 * n_rows),
        sharey="row",
        squeeze=False,
    )
    fig.suptitle(
        f"{method_name} — Probe Accuracy Over Training Checkpoints  [{probe_lbl}]",
        fontsize=11, y=1.01,
    )

    legend_handles = []
    legend_labels  = []
    legend_built   = False

    for row_idx, m in enumerate(metrics_to_plot):
        y_max = row_ymax[m]

        for ax_idx, clf_name in enumerate(clfs):
            ax = axes[row_idx][ax_idx]
            _setup_ax(ax, row_idx, ax_idx, n_rows, clf_name, m)
            ax.set_ylim(Y_MIN, y_max)

            clf_data    = data[m].get(clf_name, {})
            any_plotted = False

            for label in CK_LABELS:     # Base first, then ck1..ck8
                if label not in clf_data:
                    continue
                layers, vals = clf_data[label]
                color  = CK_COLORS[label]
                is_base = (label == "Base (Instruct)")
                lw     = 2.4 if is_base else 1.4
                ls     = "-"
                zorder = 4 if is_base else 2

                ax.plot(layers, vals,
                        color=color, linewidth=lw, linestyle=ls, zorder=zorder)
                any_plotted = True

                if not legend_built and ax_idx == 0 and row_idx == 0:
                    legend_handles.append(
                        mlines.Line2D([], [], color=color, linewidth=lw,
                                      linestyle=ls, label=label)
                    )
                    legend_labels.append(label)

            if not any_plotted:
                ax.text(0.5, 0.5, "No data",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)

        legend_built = True

    _finalize_figure(fig, axes, legend_handles, legend_labels, n_clfs, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Heatmap — methods mode  (Y = model, X = layer)
# ---------------------------------------------------------------------------

def make_heatmap_methods(checkpoint_dir: Path, out_path: Path,
                         clf_filter: list, metric: str | None,
                         probe_source: str,
                         data_dir: Path = DATA_DIR,
                         vmin: float | None = None, vmax: float | None = None,
                         dataset: str = "bio",
                         normalize: bool = False,
                         cyber_subset: str | None = None):
    if dataset == "cyber":
        print("Loading cyber y_test ...")
        y_test = load_cyber_y_test_methods(checkpoint_dir)
    else:
        csv_path = data_dir / "wmdp_tf_pairs.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"WMDP CSV not found at {csv_path}.")
        print("Loading y_test from CSV ...")
        y_test = load_y_test(csv_path)

    mask = None
    if dataset == "cyber" and cyber_subset is not None:
        meta = load_cyber_test_meta_methods(checkpoint_dir)
        if meta is None:
            print("  [warn] base_cyber_test_meta.json not found; ignoring --cyber_subset."
                  " Re-run --stage base to generate it.")
        else:
            gib_qids = load_base_cyber_gib_qids(checkpoint_dir)
            mask = compute_cyber_subset_mask(cyber_subset, meta, gib_qids)
            print(f"  Cyber subset '{cyber_subset}': {mask.sum()}/{len(mask)} pairs kept.")

    clfs            = clf_filter if clf_filter else CLF_NAMES
    metrics_to_plot = [metric] if metric else METRIC_NAMES

    data = collect_data(checkpoint_dir, clfs, y_test, metrics_to_plot,
                        probe_source, dataset, mask=mask)

    subset_lbl  = f" [{cyber_subset}]" if cyber_subset and dataset == "cyber" else ""
    row_labels  = list(ALL_MODELS.keys())   # model display names in palette order
    title = (
        f"Probe Heatmap — all models  "
        f"[{probe_source_title(probe_source, [metric]).split(' —')[0]}]{subset_lbl}"
    )
    if normalize:
        vmin, vmax = _relative_data_range(data, metrics_to_plot, clfs)
    _render_heatmap(data, metrics_to_plot, clfs, row_labels, out_path, title,
                    vmin=vmin, vmax=vmax, normalize=normalize)


# ---------------------------------------------------------------------------
# Heatmap — checkpoints mode  (Y = checkpoint, X = layer)
# ---------------------------------------------------------------------------

def make_heatmap_checkpoints(checkpoint_dir: Path, out_path: Path,
                              method_name: str,
                              clf_filter: list, metric: str | None,
                              probe_source: str,
                              data_dir: Path = DATA_DIR,
                              vmin: float | None = None, vmax: float | None = None,
                              dataset: str = "bio",
                              normalize: bool = False,
                              cyber_subset: str | None = None):
    print(f"\n=== Heatmap checkpoints mode: {method_name} ({dataset}) ===")
    if dataset == "cyber":
        y_test = load_cyber_y_test_sweep()
    else:
        csv_path = data_dir / "wmdp_tf_pairs.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"WMDP CSV not found at {csv_path}.")
        y_test = load_y_test(csv_path)

    mask = None
    if dataset == "cyber" and cyber_subset is not None:
        pairs_meta = load_cyber_test_pairs_sweep()
        gib_qids   = load_base_cyber_gib_qids(checkpoint_dir)
        mask = compute_cyber_subset_mask(cyber_subset, pairs_meta, gib_qids)
        print(f"  Cyber subset '{cyber_subset}': {mask.sum()}/{len(mask)} pairs kept.")

    clfs            = clf_filter if clf_filter else CLF_NAMES
    metrics_to_plot = [metric] if metric else METRIC_NAMES

    data = collect_data_checkpoints(
        checkpoint_dir, method_name, clfs, y_test, metrics_to_plot,
        probe_source, dataset, mask=mask
    )

    probe_lbl  = "Method Probes" if probe_source == "method" else "Base Probes"
    subset_lbl = f" [{cyber_subset}]" if cyber_subset and dataset == "cyber" else ""
    title = (
        f"{method_name} — Probe Heatmap Over Training Checkpoints"
        f"  [{probe_lbl}]{subset_lbl}"
    )
    if normalize:
        vmin, vmax = _relative_data_range(data, metrics_to_plot, clfs)
    _render_heatmap(data, metrics_to_plot, clfs, CK_LABELS, out_path, title,
                    vmin=vmin, vmax=vmax, normalize=normalize)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-layer probe metrics.\n"
            "  methods mode     — all models at final checkpoint\n"
            "  checkpoints mode — one method across training checkpoints\n"
            "  plot_type line   — curves (default)\n"
            "  plot_type heatmap — 2-D grid (X=layer, Y=checkpoint/model, colour=metric)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["methods", "checkpoints"], default="methods",
        help="Plot mode (default: methods).",
    )
    parser.add_argument(
        "--plot_type", choices=["line", "heatmap"], default="line",
        help="Plot type: line curves or 2-D heatmap (default: line).",
    )
    parser.add_argument(
        "--method", default=None,
        help=(
            "For --mode checkpoints: method name or 'all' (one file per method).\n"
            f"Available: {', '.join(SWEEP_METHODS + ['all'])}"
        ),
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints",
        help="Checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--data_dir", default=str(DATA_DIR),
        help=f"Data directory containing CSVs (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--out", default=None,
        help=(
            "Output PNG path. If omitted a descriptive name is auto-generated:\n"
            "  {plot_type}_{mode}[_{method}]_{metric}_{clf}_{probe_source}.png\n"
            "When --method all, this becomes a filename template and the method\n"
            "name is appended to the stem for each output file."
        )
    )
    parser.add_argument(
        "--clf", choices=CLF_NAMES, default=None,
        help="Only plot one classifier (default: all three)"
    )
    parser.add_argument(
        "--metric", choices=METRIC_NAMES, default=None,
        help=(
            "Metric to plot (default: all as separate rows).\n"
            "Options: accuracy | true_accuracy | false_accuracy | "
            "precision | recall | f1 | auc"
        ),
    )
    parser.add_argument(
        "--probe_source", choices=PROBE_SOURCE_NAMES, default="method",
        help=(
            "Which probes to use (default: method).\n"
            "  method — each model/checkpoint's own probes (Table 3)\n"
            "  base   — base probes on all models"
        ),
    )
    parser.add_argument(
        "--dataset", choices=["bio", "cyber"], default="bio",
        help=(
            "Which dataset to plot probes for (default: bio).\n"
            "  bio   — WMDP-bio forget set\n"
            "  cyber — WMDP-cyber set\n"
            "Note: --dataset cyber (methods mode) requires base_cyber_y_test.npy\n"
            "      in the checkpoint dir (saved automatically by --stage base)."
        ),
    )
    parser.add_argument(
        "--relative", action="store_true", default=False,
        help=(
            "Normalize heatmap values relative to Base (Instruct).\n"
            "Each cell shows (value - base_value) so the base row is always 0.\n"
            "Green = above base, red = below base. Only affects --plot_type heatmap."
        ),
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="MODEL",
        help=(
            "Restrict kfold plot to these model display names "
            "(e.g. --models 'Base (Instruct)' GradDiff RMU). "
            "Colors are preserved from the full palette. "
            "Only used with --kfold."
        ),
    )
    parser.add_argument(
        "--kfold", action="store_true", default=False,
        help=(
            "Use 5-fold aggregated data instead of single-fold hidden states.\n"
            "Reads kfold_per_layer_table.csv; draws mean line + ±CI95 shaded band.\n"
            "Only valid with --mode methods --plot_type line --dataset bio."
        ),
    )
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=None,
        metavar=("W", "H"),
        help="Figure size in inches, e.g. --figsize 5.5 3.5",
    )
    parser.add_argument(
        "--cyber_subset", choices=CYBER_SUBSETS, default=None,
        help=(
            "Filter the cyber test set to a named subset before evaluating probes.\n"
            "Only active when --dataset cyber is set.\n"
            "  og        — full test set (no filter)\n"
            "  pattern   — exclude computational/code-execution questions\n"
            "  gibberish — exclude questions where the base model produced gibberish\n"
            "  both      — exclude both computational AND base-gibberish questions\n"
            "Subset filtering requires hidden-state .npy files to be present for each\n"
            "checkpoint (pre-computed stats in results.json are computed on the full set\n"
            "and cannot be subset-filtered without the original hidden states).\n"
            "For methods mode, also requires base_cyber_test_meta.json (saved by\n"
            "--stage base). For checkpoints/sweep mode, derived from the CSV directly."
        ),
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir       = Path(args.data_dir)
    clf_filter     = [args.clf] if args.clf else []
    dataset        = args.dataset
    normalize      = args.relative
    cyber_subset   = args.cyber_subset if dataset == "cyber" else None

    # ── k-fold aggregate mode (bio-only, line or heatmap) ────────────────────
    if args.kfold:
        if dataset != "bio":
            parser.error("--kfold only supports --dataset bio (kfold is bio-only)")

        # ── kfold sweep (checkpoints mode) ───────────────────────────────────
        if args.mode == "checkpoints":
            if args.method is None or args.method == "all":
                parser.error("--kfold --mode checkpoints requires --method METHOD "
                             f"(not 'all'). Choose from: {', '.join(SWEEP_METHODS)}")
            if args.method not in SWEEP_METHODS:
                parser.error(f"Unknown method '{args.method}'. "
                             f"Choose from: {', '.join(SWEEP_METHODS)}")
            if args.plot_type == "line":
                parser.error("--kfold --mode checkpoints only supports --plot_type heatmap "
                             "(line plot not yet implemented for kfold sweep).")

            clfs            = clf_filter if clf_filter else CLF_NAMES
            metrics_to_plot = [args.metric] if args.metric else METRIC_NAMES
            metric_tag      = args.metric or "all_metrics"
            clf_tag         = args.clf.lower() if args.clf else "all_clf"
            sn              = _re.sub(r"[^a-zA-Z0-9_-]", "_", args.method)

            kfold_sweep_out = Path(args.out) if args.out else Path(
                f"kfold_sweep_heatmap_{sn}_{metric_tag}_{clf_tag}.png"
            )
            ck_row_labels = [f"ck{n}" for n in range(1, N_CHECKPOINTS + 1)]

            sweep_data = collect_data_kfold_sweep(
                data_dir, args.method, clfs, metrics_to_plot
            )
            _render_heatmap(
                sweep_data, metrics_to_plot, clfs,
                row_labels=ck_row_labels,
                out_path=kfold_sweep_out,
                title=(f"{args.method} — Per-Layer Probe Accuracy  "
                       f"5-Fold CV Mean  [Bio WMDP]\n"
                       f"Y = training checkpoint,  X = layer"),
                normalize=normalize,
            )
            print(f"Saved → {kfold_sweep_out}")
            return

        if args.mode != "methods":
            parser.error("--kfold supports --mode methods or --mode checkpoints.")

        clfs            = clf_filter if clf_filter else CLF_NAMES
        metrics_to_plot = [args.metric] if args.metric else METRIC_NAMES
        models_filter   = args.models

        # Build display-name row order (respects --models filter, preserves palette order)
        models_to_show = [m for m in ALL_MODELS
                          if models_filter is None or m in models_filter]

        # Load kfold per-layer data; convert 3-tuple → 2-tuple for heatmap renderer
        raw_kf = collect_data_kfold(data_dir, clfs, metrics_to_plot)
        # data_means[metric][clf][model] = (layers, means)  — used by _render_heatmap
        data_means = {
            m: {
                clf: {
                    mdl: (lmc[0], lmc[1])
                    for mdl, lmc in raw_kf[m].get(clf, {}).items()
                    if models_filter is None or mdl in models_filter
                }
                for clf in clfs
            }
            for m in metrics_to_plot
        }

        models_tag = "_".join(m.replace(" ", "").replace("(", "").replace(")", "")
                              for m in models_to_show) if models_filter else "all"
        metric_tag = args.metric or "all_metrics"
        clf_tag    = args.clf.lower() if args.clf else "all_clf"

        if args.plot_type == "heatmap":
            kfold_out = Path(args.out) if args.out else Path(
                f"kfold_heatmap_methods_{metric_tag}_{clf_tag}_{models_tag}.png"
            )
            _render_heatmap(
                data_means, metrics_to_plot, clfs,
                row_labels=models_to_show,
                out_path=kfold_out,
                title=(
                    f"Per-Layer Probe Accuracy — 5-Fold CV Mean  [Bio WMDP]\n"
                    f"Models: {', '.join(models_to_show)}"
                ),
                normalize=normalize,
            )
        else:
            kfold_out = Path(args.out) if args.out else Path(
                f"kfold_line_methods_{metric_tag}_{clf_tag}_{models_tag}.png"
            )
            make_plot_kfold(
                data_dir=data_dir,
                out_path=kfold_out,
                clf_filter=clf_filter,
                metric=args.metric,
                models_filter=models_filter,
                figsize=tuple(args.figsize) if args.figsize else None,
            )
        return

    # Resolve base output path (auto-generate if not given)
    if args.out is not None:
        base_out = Path(args.out)
    else:
        base_out = _auto_out_path(
            args.plot_type, args.mode,
            args.method if args.mode == "checkpoints" else None,
            args.metric, args.clf, args.probe_source, dataset,
            normalize=normalize,
            cyber_subset=cyber_subset,
        )

    # ── methods mode ─────────────────────────────────────────────────────────
    if args.mode == "methods":
        if args.plot_type == "heatmap":
            make_heatmap_methods(
                checkpoint_dir=checkpoint_dir,
                out_path=base_out,
                clf_filter=clf_filter,
                metric=args.metric,
                probe_source=args.probe_source,
                data_dir=data_dir,
                dataset=dataset,
                cyber_subset=cyber_subset,
            )
        else:
            make_plot(
                checkpoint_dir=checkpoint_dir,
                out_path=base_out,
                clf_filter=clf_filter,
                metric=args.metric,
                probe_source=args.probe_source,
                data_dir=data_dir,
                dataset=dataset,
                cyber_subset=cyber_subset,
            )

    # ── checkpoints mode ──────────────────────────────────────────────────────
    else:
        if args.method is None:
            parser.error("--mode checkpoints requires --method METHOD (or 'all').")
        if args.method not in SWEEP_METHODS and args.method != "all":
            parser.error(
                f"Unknown method '{args.method}'. "
                f"Choose from: {', '.join(SWEEP_METHODS + ['all'])}"
            )

        methods = SWEEP_METHODS if args.method == "all" else [args.method]

        if args.plot_type == "heatmap":
            # For heatmaps: compute a global colour scale across ALL methods so
            # every figure is directly comparable.  Data dicts are small (scalars
            # only — raw hs arrays are deleted inside collect_data_checkpoints).
            clfs            = clf_filter if clf_filter else CLF_NAMES
            metrics_to_plot = [args.metric] if args.metric else METRIC_NAMES
            if dataset == "cyber":
                y_test = load_cyber_y_test_sweep()
            else:
                csv_path = data_dir / "wmdp_tf_pairs.csv"
                y_test   = load_y_test(csv_path)

            # Pre-compute subset mask once (shared across all methods)
            mask = None
            if dataset == "cyber" and cyber_subset is not None:
                pairs_meta = load_cyber_test_pairs_sweep()
                gib_qids   = load_base_cyber_gib_qids(checkpoint_dir)
                mask = compute_cyber_subset_mask(cyber_subset, pairs_meta, gib_qids)
                print(f"  Cyber subset '{cyber_subset}': {mask.sum()}/{len(mask)} pairs kept.")

            print("Computing global colour scale across all methods ...")
            global_vmin, global_vmax = 1.0, 0.0
            all_data = {}
            for method in methods:
                d = collect_data_checkpoints(
                    checkpoint_dir, method, clfs, y_test, metrics_to_plot,
                    args.probe_source, dataset, mask=mask
                )
                all_data[method] = d
                lo, hi = _data_range(d, metrics_to_plot, clfs)
                global_vmin = min(global_vmin, lo)
                global_vmax = max(global_vmax, hi)
            print(f"  Global scale: vmin={global_vmin:.4f}  vmax={global_vmax:.4f}")

            for method in methods:
                if len(methods) > 1:
                    file_out = base_out.parent / f"{base_out.stem}_{_safe_name(method)}{base_out.suffix}"
                else:
                    file_out = base_out

                probe_lbl  = "Method Probes" if args.probe_source == "method" else "Base Probes"
                subset_lbl = f" [{cyber_subset}]" if cyber_subset else ""
                title = (
                    f"{method} — Probe Heatmap Over Training Checkpoints"
                    f"  [{probe_lbl}]  [{dataset.upper()}]{subset_lbl}"
                )
                _render_heatmap(
                    all_data[method], metrics_to_plot, clfs, CK_LABELS,
                    file_out, title,
                    vmin=global_vmin, vmax=global_vmax,
                )

        else:
            for method in methods:
                if len(methods) > 1:
                    file_out = base_out.parent / f"{base_out.stem}_{_safe_name(method)}{base_out.suffix}"
                else:
                    file_out = base_out

                make_plot_checkpoints(
                    checkpoint_dir=checkpoint_dir,
                    out_path=file_out,
                    method_name=method,
                    clf_filter=clf_filter,
                    metric=args.metric,
                    probe_source=args.probe_source,
                    data_dir=data_dir,
                    dataset=dataset,
                    cyber_subset=cyber_subset,
                )


# ---------------------------------------------------------------------------
# K-fold aggregate line-plot (bio only)
# ---------------------------------------------------------------------------

# Map kfold model name → display name used in ALL_MODELS
_KFOLD_MODEL_DISPLAY = {v: k for k, v in ALL_MODELS.items()}
_KFOLD_MODEL_DISPLAY["base"] = "Base (Instruct)"   # kfold uses "base", display uses "Base (Instruct)"


def collect_data_kfold(data_dir: Path, clfs: list, metrics: list) -> dict:
    """
    Read kfold_per_layer_table.csv and return
    data[metric][clf_name][model_name] = (layers, means, ci95s).

    model_name uses the same display keys as ALL_MODELS (e.g. "Base (Instruct)").
    """
    pl_path = data_dir / "kfold_per_layer_table.csv"
    if not pl_path.exists():
        raise FileNotFoundError(
            f"{pl_path} not found — run: python kfold_probe.py --stage kfold_tables"
        )

    # Load: indexed by (model, clf, layer)
    raw: dict = {}   # (model_display, clf, layer) -> {metric: (mean, ci95)}
    with open(pl_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mdl   = _KFOLD_MODEL_DISPLAY.get(row["model"], row["model"])
            clf   = row["clf"]
            layer = int(row["layer"])
            raw.setdefault((mdl, clf, layer), {})
            for m in metrics:
                mean_v = _safe_float(row.get(f"mean_{m}"))
                ci_v   = _safe_float(row.get(f"ci95_{m}"))
                if mean_v is not None:
                    raw[(mdl, clf, layer)][m] = (mean_v, ci_v if ci_v is not None else 0.0)

    # Reshape to data[metric][clf][model] = (layers, means, ci95s)
    data = {m: {clf: {} for clf in clfs} for m in metrics}
    # Gather all layers per (model, clf)
    from collections import defaultdict
    tmp = defaultdict(lambda: defaultdict(dict))  # (model, clf) -> metric -> layer -> (mean, ci)
    for (mdl, clf, layer), m_dict in raw.items():
        if clf not in clfs:
            continue
        for m, (mean_v, ci_v) in m_dict.items():
            if m in metrics:
                tmp[(mdl, clf)][m][layer] = (mean_v, ci_v)

    for (mdl, clf), m_dict in tmp.items():
        for m, layer_dict in m_dict.items():
            if m not in metrics:
                continue
            sorted_layers = sorted(layer_dict.keys())
            means = [layer_dict[l][0] for l in sorted_layers]
            ci95s = [layer_dict[l][1] for l in sorted_layers]
            data[m][clf][mdl] = (sorted_layers, means, ci95s)

    return data


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect_data_kfold_sweep(data_dir: Path, method: str,
                              clfs: list, metrics: list) -> dict:
    """
    Read kfold_sweep_per_layer_aggregated.csv for one method.

    Returns data[metric][clf][ck_label] = (layers, means)
    where ck_label = "ck1" … "ck8".
    """
    pl_path = data_dir / "kfold_sweep_per_layer_aggregated.csv"
    if not pl_path.exists():
        raise FileNotFoundError(
            f"{pl_path} not found — generate it first:\n"
            "  python kfold_probe.py --stage kfold_sweep_summary\n"
            "  python kfold_probe.py --stage kfold_sweep_tables"
        )

    from collections import defaultdict
    # tmp[(clf, ck_label)][metric][layer] = mean_value
    tmp = defaultdict(lambda: defaultdict(dict))
    with open(pl_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] != method:
                continue
            clf   = row["clf"]
            layer = int(row["layer"])
            ck    = int(row["ck"])
            if clf not in clfs:
                continue
            ck_label = f"ck{ck}"
            for m in metrics:
                v = _safe_float(row.get(f"mean_{m}"))
                if v is not None:
                    tmp[(clf, ck_label)][m][layer] = v

    data = {m: {clf: {} for clf in clfs} for m in metrics}
    for (clf, ck_label), m_dict in tmp.items():
        for m, layer_dict in m_dict.items():
            if m not in metrics:
                continue
            sorted_layers = sorted(layer_dict)
            data[m][clf][ck_label] = (sorted_layers,
                                      [layer_dict[l] for l in sorted_layers])
    return data


def make_plot_kfold(data_dir: Path, out_path: Path,
                   clf_filter: list, metric: str | None,
                   models_filter: list | None = None,
                   figsize: tuple | None = None):
    """
    Line plot of per-layer probe accuracy from the 5-fold aggregated data.
    Draws mean line + shaded ±CI95 band per model (bio only).
    """
    clfs            = clf_filter if clf_filter else CLF_NAMES
    metrics_to_plot = [metric] if metric else METRIC_NAMES
    # models_to_show: same order as ALL_MODELS, colors preserved
    models_to_show  = [m for m in ALL_MODELS
                       if models_filter is None or m in models_filter]

    data = collect_data_kfold(data_dir, clfs, metrics_to_plot)

    # Compute per-row y-max from means
    row_ymax = {}
    for m in metrics_to_plot:
        mx = Y_MIN
        for clf_data in data[m].values():
            for _, means, _ in clf_data.values():
                if means:
                    mx = max(mx, max(means))
        row_ymax[m] = min(mx * (1 + Y_PAD), 1.0)

    n_rows = len(metrics_to_plot)
    n_clfs = len(clfs)

    fig, axes = plt.subplots(
        n_rows, n_clfs,
        figsize=figsize if figsize else (7 * n_clfs, 4.5 * n_rows),
        sharey="row",
        squeeze=False,
    )
    fig.suptitle(
        "Per-Layer Probe Accuracy — 5-Fold CV  (mean ± 95 % CI)  [Bio WMDP]",
        fontsize=9, x=0.5, ha="center",
    )

    for row_idx, m in enumerate(metrics_to_plot):
        y_max = row_ymax[m]
        for ax_idx, clf_name in enumerate(clfs):
            ax = axes[row_idx][ax_idx]
            _setup_ax(ax, row_idx, ax_idx, n_rows,
                      clf_name if n_clfs > 1 else None, m)
            ax.set_ylim(Y_MIN, y_max)

            clf_data    = data[m].get(clf_name, {})
            any_plotted = False

            for model_name in models_to_show:
                if model_name not in clf_data:
                    continue
                layers, means, ci95s = clf_data[model_name]
                means  = np.array(means)
                ci95s  = np.array(ci95s)
                color  = MODEL_COLORS[model_name]
                lw     = 2.4 if model_name in THICK_MODELS else 1.2
                ls     = "--" if model_name in DASHED_MODELS else "-"
                zorder = 4 if model_name in THICK_MODELS else 2

                ax.plot(layers, means, color=color, linewidth=lw,
                        linestyle=ls, zorder=zorder)
                # Shade ± CI95 band
                ax.fill_between(layers, means - ci95s, means + ci95s,
                                color=color, alpha=0.15, linewidth=0, zorder=zorder - 1)
                any_plotted = True

            if not any_plotted:
                ax.text(0.5, 0.5, "No data",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)

    # Legend
    legend_handles, legend_labels = [], []
    models_with_data = set()
    for m in metrics_to_plot:
        for clf in clfs:
            models_with_data.update(data[m].get(clf, {}).keys())

    legend_order = [m for m in LEGEND_CURVE_ORDER if models_filter is None or m in models_filter]
    for model_name in legend_order:
        if model_name not in models_with_data:
            continue
        color = MODEL_COLORS[model_name]
        lw    = 2.4 if model_name in THICK_MODELS else 1.2
        ls    = "--" if model_name in DASHED_MODELS else "-"
        legend_handles.append(
            mlines.Line2D([], [], color=color, linewidth=lw, linestyle=ls,
                          label=model_name)
        )
        legend_labels.append(model_name)

    # Add CI band patch to legend
    import matplotlib.patches as mpatches
    legend_handles.append(
        mpatches.Patch(color="grey", alpha=0.3, label="±95 % CI (5-fold)")
    )
    legend_labels.append("±95 % CI (5-fold)")

    _finalize_figure(fig, axes, legend_handles, legend_labels, n_clfs, out_path,
                     legend_outside=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
