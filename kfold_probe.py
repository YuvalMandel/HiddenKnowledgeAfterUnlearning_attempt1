#!/usr/bin/env python3
"""
kfold_probe.py — 5-fold cross-validation probe training for bio WMDP hidden states.

Stages:
  kfold_train   Train & evaluate probes for one (fold, model) pair.
                Reads existing .npy hidden-state files; does NOT load any LLM.
  kfold_summary Aggregate per-fold JSON results → CSV tables.

Usage:
  python kfold_probe.py --stage kfold_train --fold 0 --model base
  python kfold_probe.py --stage kfold_train --fold 2 --model GradDiff
  python kfold_probe.py --stage kfold_summary

Fold assignment for fold k (0-indexed):
  test  = fold[k]
  val   = fold[(k+1) % N_FOLDS]
  train = fold[(k+2)%N] ∪ fold[(k+3)%N] ∪ fold[(k+4)%N]

Each fold contains floor(Q / N_FOLDS) questions; both True and False variants
of each MCQ question always land in the same fold.
"""

import argparse
import csv
import json
import random
import re
import sys
import numpy as np
from pathlib import Path

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "scikit-learn", "-q"], check=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# =============================================================================
# Configuration (must match hidden_knowledge_after_unlearning.py)
# =============================================================================
RANDOM_SEED  = 42
N_FOLDS      = 5
CLF_NAMES    = ["LR", "RF", "AdaBoost"]

CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR       = Path("data")
WMDP_CSV_PATH  = DATA_DIR / "wmdp_tf_pairs.csv"
KFOLD_DIR      = CHECKPOINT_DIR / "kfold"

# Models: base + 8 unlearning methods
ALL_MODELS = ["base", "GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]

# Probe band config (must match main script)
PCA_DIMS_MULTI    = 256
MULTI_LAYER_START = 10
MULTI_LAYER_END   = 22
INIT_BAND_START   = 1
INIT_BAND_EMB_START = 0
INIT_BAND_END     = 9
END_BAND_START    = 23
IB_NO_PCA_START   = 1
IB_NO_PCA_END     = 6


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def get_npy_paths(model: str):
    """Return (train_path, val_path, test_path) for the given model."""
    if model == "base":
        return (
            CHECKPOINT_DIR / "base_hs_train.npy",
            CHECKPOINT_DIR / "base_hs_val.npy",
            CHECKPOINT_DIR / "base_hs_test.npy",
        )
    sn = safe_name(model)
    return (
        CHECKPOINT_DIR / f"{sn}_hs_train.npy",
        CHECKPOINT_DIR / f"{sn}_hs_val.npy",
        CHECKPOINT_DIR / f"{sn}_hs_test.npy",
    )


def result_path(fold: int, model: str) -> Path:
    KFOLD_DIR.mkdir(parents=True, exist_ok=True)
    sn = "base" if model == "base" else safe_name(model)
    return KFOLD_DIR / f"f{fold}_{sn}_results.json"


# =============================================================================
# Fold index building
# =============================================================================

def build_fold_assignment(csv_path=WMDP_CSV_PATH, n_folds=N_FOLDS, seed=RANDOM_SEED):
    """
    Read the bio T/F pairs CSV and group pairs by original_id (MCQ question).

    Each original_id has exactly 2 rows (pos + neg). We assign each question
    to a fold so both variants always land together.

    Returns:
        by_qid  : {original_id: [(combined_npy_idx, label_int), ...]}
        folds   : list of n_folds lists of original_ids (trimmed to n_folds * fold_size)
        n_train : number of train-split rows (for combined index offset)
        n_val   : number of val-split rows
        fold_size : questions per fold
    """
    # Pass 1: count rows per split (needed for offset calculation)
    n_train = n_val = n_test = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                n_train += 1
            elif row["split"] == "val":
                n_val += 1
            else:
                n_test += 1

    # Pass 2: assign combined npy indices to each (original_id, pair) entry
    by_qid = {}
    ctr = {"train": 0, "val": 0, "test": 0}
    offsets = {"train": 0, "val": n_train, "test": n_train + n_val}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split   = row["split"]
            orig_id = int(row["original_id"])
            label   = 1 if row["label"] == "True" else 0
            combined_idx = offsets[split] + ctr[split]
            ctr[split] += 1
            by_qid.setdefault(orig_id, []).append((combined_idx, label))

    # Shuffle question IDs with fixed seed, trim to multiple of n_folds
    qids = sorted(by_qid.keys())
    rng  = random.Random(seed)
    rng.shuffle(qids)
    n_q       = len(qids)
    fold_size = n_q // n_folds
    n_q_used  = fold_size * n_folds
    qids      = qids[:n_q_used]

    folds = [qids[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    print(f"  [kfold] Total questions: {n_q}  using {n_q_used} ({n_folds} folds × {fold_size})",
          flush=True)
    return by_qid, folds, n_train, n_val, fold_size


def get_split_indices(fold_idx: int, by_qid: dict, folds: list, n_folds=N_FOLDS):
    """
    Given fold k:
      test  = folds[k]
      val   = folds[(k+1) % n_folds]
      train = remaining folds

    Returns (train_idx, train_labels, val_idx, val_labels, test_idx, test_labels)
    as numpy arrays. Indices are into the concatenated [train | val | test] npy array.
    """
    test_qids  = folds[fold_idx]
    val_qids   = folds[(fold_idx + 1) % n_folds]
    train_qids = []
    for i in range(n_folds):
        if i != fold_idx and i != (fold_idx + 1) % n_folds:
            train_qids.extend(folds[i])

    def collect(qids):
        idxs, labels = [], []
        for qid in qids:
            for (idx, lbl) in by_qid[qid]:
                idxs.append(idx)
                labels.append(lbl)
        return np.array(idxs, dtype=np.int64), np.array(labels, dtype=np.int32)

    return (
        *collect(train_qids),
        *collect(val_qids),
        *collect(test_qids),
    )


# =============================================================================
# Hidden-state loading
# =============================================================================

def load_hs_for_model(model: str):
    """Load all three npy files and concatenate along axis 0."""
    train_p, val_p, test_p = get_npy_paths(model)
    missing = [p for p in (train_p, val_p, test_p) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing npy files for model '{model}': {missing}")
    print(f"  Loading hidden states for '{model}'...", flush=True)
    hs_tr = np.load(train_p)   # (n_train, n_layers, hidden_dim)
    hs_va = np.load(val_p)
    hs_te = np.load(test_p)
    print(f"    train {hs_tr.shape}  val {hs_va.shape}  test {hs_te.shape}", flush=True)
    return np.concatenate([hs_tr, hs_va, hs_te], axis=0)  # (N, n_layers, hidden_dim)


# =============================================================================
# Probe pipelines (identical to main script)
# =============================================================================

def _make_per_layer_pipeline(clf_name: str) -> Pipeline:
    if clf_name == "LR":
        return Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)),
        ])
    elif clf_name == "RF":
        return Pipeline([
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED)),
        ])
    elif clf_name == "AdaBoost":
        return Pipeline([
            ("sc",  StandardScaler()),
            ("clf", AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED)),
        ])
    raise ValueError(f"Unknown clf_name: {clf_name}")


def _make_multi_layer_pipeline(clf_name: str) -> Pipeline:
    if clf_name == "LR":
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    elif clf_name == "RF":
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED)
    elif clf_name == "AdaBoost":
        clf = AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED)
    else:
        raise ValueError(clf_name)
    return Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=PCA_DIMS_MULTI, random_state=RANDOM_SEED)),
        ("clf", clf),
    ])


def _make_no_pca_multi_layer_pipeline(clf_name: str) -> Pipeline:
    if clf_name == "LR":
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    elif clf_name == "RF":
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED)
    elif clf_name == "AdaBoost":
        clf = AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED)
    else:
        raise ValueError(clf_name)
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", clf),
    ])


# =============================================================================
# Probe training (identical logic to main script's train_probe_set)
# =============================================================================

def train_probe_set(hs_train: np.ndarray, y_train: np.ndarray,
                    hs_val: np.ndarray, y_val: np.ndarray,
                    label: str = "") -> dict:
    """Train all per-layer and multi-layer probes. Returns ProbeSet dict."""
    n_train, n_layers, _ = hs_train.shape
    prefix = f"[{label}] " if label else ""

    # Per-layer probes
    per_layer = {c: {} for c in CLF_NAMES}
    for l in range(n_layers):
        X_tr  = hs_train[:, l, :]
        for clf_name in CLF_NAMES:
            pipe = _make_per_layer_pipeline(clf_name)
            pipe.fit(X_tr, y_train)
            per_layer[clf_name][l] = pipe
        print(f"  {prefix}Per-layer: layer {l+1}/{n_layers}", end="\r", flush=True)
    print(f"  {prefix}Per-layer: {n_layers} layers × {len(CLF_NAMES)} classifiers done.",
          flush=True)

    # Best layer per classifier (on val set)
    best_layers = {}
    for clf_name in CLF_NAMES:
        val_accs = {l: per_layer[clf_name][l].score(hs_val[:, l, :], y_val)
                    for l in range(n_layers)}
        best_l = max(val_accs, key=val_accs.get)
        best_layers[clf_name] = best_l
        print(f"  {prefix}{clf_name} best layer: {best_l}  "
              f"val acc: {val_accs[best_l]:.3f}", flush=True)

    def _band(start, end, pipeline_fn, name):
        s = max(0, start)
        e = min(n_layers - 1, end)
        X_tr  = hs_train[:, s:e + 1, :].reshape(n_train, -1)
        X_val = hs_val[:,   s:e + 1, :].reshape(len(hs_val), -1)
        probes = {}
        for clf_name in CLF_NAMES:
            pipe = pipeline_fn(clf_name)
            pipe.fit(X_tr, y_train)
            va = pipe.score(X_val, y_val)
            probes[clf_name] = pipe
            print(f"  {prefix}{name} {clf_name} val acc: {va:.3f}", flush=True)
        return probes, (s, e)

    mid_band,     mb_range    = _band(MULTI_LAYER_START, MULTI_LAYER_END,
                                       _make_multi_layer_pipeline, "mid_band")
    full_layer,   fl_range    = _band(0, n_layers - 1,
                                       _make_multi_layer_pipeline, "full_layer")
    init_band,    ib_range    = _band(INIT_BAND_START, INIT_BAND_END,
                                       _make_multi_layer_pipeline, "init_band")
    init_band_emb,ibe_range   = _band(INIT_BAND_EMB_START, INIT_BAND_END,
                                       _make_multi_layer_pipeline, "init_band_emb")
    end_band,     eb_range    = _band(END_BAND_START, n_layers - 1,
                                       _make_multi_layer_pipeline, "end_band")
    ib_no_pca,    ibnp_range  = _band(IB_NO_PCA_START, IB_NO_PCA_END,
                                       _make_no_pca_multi_layer_pipeline, "ib_no_pca")

    return {
        "per_layer":           per_layer,
        "mid_band":            mid_band,
        "best_layers":         best_layers,
        "mid_band_range":      mb_range,
        "full_layer":          full_layer,
        "full_layer_range":    fl_range,
        "init_band":           init_band,
        "init_band_range":     ib_range,
        "init_band_emb":       init_band_emb,
        "init_band_emb_range": ibe_range,
        "end_band":            end_band,
        "end_band_range":      eb_range,
        "ib_no_pca":           ib_no_pca,
        "ib_no_pca_range":     ibnp_range,
    }


# =============================================================================
# Probe evaluation (identical to main script)
# =============================================================================

def _clf_metrics(labels, preds, scores=None):
    p  = float(precision_score(labels, preds, zero_division=0))
    r  = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    out = {"precision": p, "recall": r, "f1": f1}
    if scores is not None:
        try:
            out["auc"] = float(roc_auc_score(labels, scores))
        except Exception:
            out["auc"] = 0.0
    return out


def _pipe_stats(pipe, X, labels):
    preds    = pipe.predict(X)
    yes_mask = labels == 1
    no_mask  = labels == 0
    if hasattr(pipe, "predict_proba"):
        scores = pipe.predict_proba(X)[:, 1]
    else:
        scores = pipe.decision_function(X)
    m = _clf_metrics(labels, preds, scores)
    return {
        "accuracy":       float((preds == labels).mean()),
        "true_accuracy":  float((preds[yes_mask] == 1).mean()) if yes_mask.any() else 0.0,
        "false_accuracy": float((preds[no_mask]  == 0).mean()) if no_mask.any()  else 0.0,
        "precision":      m["precision"],
        "recall":         m["recall"],
        "f1":             m["f1"],
        "auc":            m.get("auc", 0.0),
        "tp":             int((preds[yes_mask] == 1).sum()) if yes_mask.any() else 0,
        "fn":             int((preds[yes_mask] == 0).sum()) if yes_mask.any() else 0,
        "tn":             int((preds[no_mask]  == 0).sum()) if no_mask.any()  else 0,
        "fp":             int((preds[no_mask]  == 1).sum()) if no_mask.any()  else 0,
        "n_total":        len(labels),
    }


def _preds_stats(preds, labels, scores=None):
    yes_mask = labels == 1
    no_mask  = labels == 0
    m = _clf_metrics(labels, preds, scores)
    out = {
        "accuracy":       float((preds == labels).mean()),
        "true_accuracy":  float((preds[yes_mask] == 1).mean()) if yes_mask.any() else 0.0,
        "false_accuracy": float((preds[no_mask]  == 0).mean()) if no_mask.any()  else 0.0,
        "precision":      m["precision"],
        "recall":         m["recall"],
        "f1":             m["f1"],
        "tp":             int((preds[yes_mask] == 1).sum()) if yes_mask.any() else 0,
        "fn":             int((preds[yes_mask] == 0).sum()) if yes_mask.any() else 0,
        "tn":             int((preds[no_mask]  == 0).sum()) if no_mask.any()  else 0,
        "fp":             int((preds[no_mask]  == 1).sum()) if no_mask.any()  else 0,
        "n_total":        len(labels),
    }
    if scores is not None:
        out["auc"] = m.get("auc", 0.0)
    return out


def _ensemble_stats(probe_set, hs_test, y_test):
    ml_start, ml_end = probe_set["mid_band_range"]
    empty = {"accuracy": 0.0, "true_accuracy": 0.0, "false_accuracy": 0.0,
             "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}
    result = {"vote": {}, "avg": {}}
    for clf_name in CLF_NAMES:
        layer_probes = probe_set["per_layer"].get(clf_name, {})
        votes_list, proba_list = [], []
        for l in range(ml_start, ml_end + 1):
            pipe = layer_probes.get(l)
            if pipe is None:
                continue
            X = hs_test[:, l, :]
            votes_list.append(pipe.predict(X))
            if hasattr(pipe, "predict_proba"):
                proba_list.append(pipe.predict_proba(X)[:, 1])
            else:
                sc = pipe.decision_function(X)
                proba_list.append(1.0 / (1.0 + np.exp(-sc)))
        if not votes_list:
            result["vote"][clf_name] = dict(empty)
            result["avg"][clf_name]  = dict(empty)
            continue
        votes_arr  = np.stack(votes_list, axis=0)
        vote_preds = (votes_arr.sum(axis=0) > len(votes_list) / 2).astype(int)
        avg_proba  = np.stack(proba_list, axis=0).mean(axis=0)
        avg_preds  = (avg_proba >= 0.5).astype(int)
        result["vote"][clf_name] = _preds_stats(vote_preds, y_test)
        result["avg"][clf_name]  = _preds_stats(avg_preds,  y_test, scores=avg_proba)
    return result


def compute_all_probe_stats(probe_set, hs_test, y_test):
    """Evaluate all probes on test set. Mirrors main script exactly."""
    n_test  = len(hs_test)
    result  = {"per_layer": {}, "mid_band": {}}

    # Per-layer (best layer)
    best_layers = probe_set["best_layers"]
    for clf_name in CLF_NAMES:
        l    = best_layers[clf_name]
        pipe = probe_set["per_layer"][clf_name][l]
        s    = _pipe_stats(pipe, hs_test[:, l, :], y_test)
        result["per_layer"][clf_name] = {**s, "best_layer": l}

    # Mid-band
    mb_start, mb_end = probe_set["mid_band_range"]
    X_mb = hs_test[:, mb_start:mb_end + 1, :].reshape(n_test, -1)
    for clf_name in CLF_NAMES:
        result["mid_band"][clf_name] = _pipe_stats(probe_set["mid_band"][clf_name], X_mb, y_test)

    # Ensemble
    ens = _ensemble_stats(probe_set, hs_test, y_test)
    result["vote_ensemble"] = ens["vote"]
    result["avg_ensemble"]  = ens["avg"]

    # Full-layer
    fl_start, fl_end = probe_set["full_layer_range"]
    X_fl = hs_test[:, fl_start:fl_end + 1, :].reshape(n_test, -1)
    result["full_layer"] = {
        clf_name: _pipe_stats(probe_set["full_layer"][clf_name], X_fl, y_test)
        for clf_name in CLF_NAMES if clf_name in probe_set["full_layer"]
    }

    # Init-band
    ib_start, ib_end = probe_set["init_band_range"]
    X_ib = hs_test[:, ib_start:ib_end + 1, :].reshape(n_test, -1)
    result["init_band"] = {
        clf_name: _pipe_stats(probe_set["init_band"][clf_name], X_ib, y_test)
        for clf_name in CLF_NAMES if clf_name in probe_set["init_band"]
    }

    # Init-band-emb
    ibe_start, ibe_end = probe_set["init_band_emb_range"]
    X_ibe = hs_test[:, ibe_start:ibe_end + 1, :].reshape(n_test, -1)
    result["init_band_emb"] = {
        clf_name: _pipe_stats(probe_set["init_band_emb"][clf_name], X_ibe, y_test)
        for clf_name in CLF_NAMES if clf_name in probe_set["init_band_emb"]
    }

    # End-band
    eb_start, eb_end = probe_set["end_band_range"]
    X_eb = hs_test[:, eb_start:eb_end + 1, :].reshape(n_test, -1)
    result["end_band"] = {
        clf_name: _pipe_stats(probe_set["end_band"][clf_name], X_eb, y_test)
        for clf_name in CLF_NAMES if clf_name in probe_set["end_band"]
    }

    # Init-band-no-PCA
    ibnp_start, ibnp_end = probe_set["ib_no_pca_range"]
    X_ibnp = hs_test[:, ibnp_start:ibnp_end + 1, :].reshape(n_test, -1)
    result["ib_no_pca"] = {
        clf_name: _pipe_stats(probe_set["ib_no_pca"][clf_name], X_ibnp, y_test)
        for clf_name in CLF_NAMES if clf_name in probe_set["ib_no_pca"]
    }

    return result


def compute_all_layers_probe_stats(probe_set, hs_test, y_test):
    """Return per-layer stats for every layer (skipping embedding layer 0)."""
    result = {}
    for clf_name in CLF_NAMES:
        result[clf_name] = {}
        for l in sorted(probe_set["per_layer"].get(clf_name, {}).keys()):
            if l == 0:
                continue
            s = _pipe_stats(probe_set["per_layer"][clf_name][l], hs_test[:, l, :], y_test)
            result[clf_name][l] = {k: s[k] for k in
                                   ("accuracy", "true_accuracy", "false_accuracy",
                                    "precision", "recall", "f1", "auc")}
    return result


# =============================================================================
# Stage: kfold_train
# =============================================================================

def run_kfold_train(fold: int, model: str):
    print(f"\n{'='*60}", flush=True)
    print(f"  kfold_train  fold={fold}  model={model}", flush=True)
    print(f"{'='*60}", flush=True)

    out_path = result_path(fold, model)
    if out_path.exists():
        print(f"  [skip] Result already exists: {out_path}", flush=True)
        return

    # Build fold assignment from CSV
    print("  Building fold assignment from CSV...", flush=True)
    by_qid, folds, n_train, n_val, fold_size = build_fold_assignment()

    # Get split indices for this fold
    (tr_idx, tr_lbl, va_idx, va_lbl, te_idx, te_lbl) = \
        get_split_indices(fold, by_qid, folds)
    print(f"  Split sizes — train: {len(tr_idx)}  val: {len(va_idx)}  test: {len(te_idx)}",
          flush=True)

    # Load hidden states for this model
    hs_all = load_hs_for_model(model)  # (N, n_layers, hidden_dim)

    hs_train = hs_all[tr_idx]
    hs_val   = hs_all[va_idx]
    hs_test  = hs_all[te_idx]
    del hs_all  # free memory

    # Train probes
    print(f"\n  Training probes...", flush=True)
    probe_set = train_probe_set(hs_train, tr_lbl, hs_val, va_lbl,
                                label=f"f{fold}_{model}")

    # Evaluate on test set
    print(f"\n  Evaluating on test set...", flush=True)
    probe_stats    = compute_all_probe_stats(probe_set, hs_test, te_lbl)
    per_layer_all  = compute_all_layers_probe_stats(probe_set, hs_test, te_lbl)

    # Serialise (probe objects are not JSON-serialisable; we only save stats)
    record = {
        "fold":         fold,
        "model":        model,
        "n_train":      int(len(tr_idx)),
        "n_val":        int(len(va_idx)),
        "n_test":       int(len(te_idx)),
        "fold_size_q":  int(fold_size),
        "probe_stats":  probe_stats,
        "per_layer_all": {
            clf_name: {str(l): v for l, v in layers.items()}
            for clf_name, layers in per_layer_all.items()
        },
    }

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)


# =============================================================================
# Stage: kfold_summary
# =============================================================================

# Probe types in the order we'll use for CSVs
PROBE_TYPES = [
    "per_layer",
    "mid_band",
    "full_layer",
    "init_band",
    "init_band_emb",
    "end_band",
    "ib_no_pca",
    "vote_ensemble",
    "avg_ensemble",
]

METRIC_COLS = ["accuracy", "true_accuracy", "false_accuracy",
               "precision", "recall", "f1", "auc",
               "tp", "fn", "tn", "fp", "n_total"]


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run_kfold_summary():
    """Load all per-fold JSON results and write summary CSVs."""
    print("\n[kfold_summary] Loading results...", flush=True)

    # Collect all results: results[model][fold] = record
    results = {m: {} for m in ALL_MODELS}
    missing = []
    for model in ALL_MODELS:
        for fold in range(N_FOLDS):
            p = result_path(fold, model)
            if not p.exists():
                missing.append(str(p))
                continue
            with open(p) as f:
                results[model][fold] = json.load(f)

    if missing:
        print(f"  WARNING: {len(missing)} result file(s) missing:", flush=True)
        for m in missing:
            print(f"    {m}", flush=True)

    # ── CSV 1: per-fold results (one row per fold × model × probe_type × clf) ─
    per_fold_rows = []
    for model in ALL_MODELS:
        for fold in range(N_FOLDS):
            rec = results[model].get(fold)
            if rec is None:
                continue
            ps = rec["probe_stats"]
            for pt in PROBE_TYPES:
                pt_stats = ps.get(pt, {})
                for clf_name in CLF_NAMES:
                    clf_stats = pt_stats.get(clf_name, {})
                    row = {
                        "fold":       fold,
                        "model":      model,
                        "probe_type": pt,
                        "clf":        clf_name,
                        "n_train":    rec["n_train"],
                        "n_val":      rec["n_val"],
                        "n_test":     rec["n_test"],
                    }
                    for m_col in METRIC_COLS:
                        row[m_col] = clf_stats.get(m_col, "")
                    if pt == "per_layer":
                        row["best_layer"] = clf_stats.get("best_layer", "")
                    else:
                        row["best_layer"] = ""
                    per_fold_rows.append(row)

    pf_path = DATA_DIR / "kfold_results_per_fold.csv"
    pf_fields = ["fold", "model", "probe_type", "clf",
                 "n_train", "n_val", "n_test", "best_layer"] + METRIC_COLS
    with open(pf_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pf_fields)
        w.writeheader()
        w.writerows(per_fold_rows)
    print(f"  Wrote {len(per_fold_rows)} rows → {pf_path}", flush=True)

    # ── CSV 2: aggregated (mean ± std across folds) ───────────────────────────
    agg_rows = []
    numeric_metrics = ["accuracy", "true_accuracy", "false_accuracy",
                       "precision", "recall", "f1", "auc"]
    for model in ALL_MODELS:
        for pt in PROBE_TYPES:
            for clf_name in CLF_NAMES:
                # Collect values across folds
                fold_vals = {m: [] for m in numeric_metrics + ["best_layer", "n_test"]}
                for fold in range(N_FOLDS):
                    rec = results[model].get(fold)
                    if rec is None:
                        continue
                    clf_stats = rec["probe_stats"].get(pt, {}).get(clf_name, {})
                    for m_col in numeric_metrics:
                        v = _safe_float(clf_stats.get(m_col))
                        if v is not None:
                            fold_vals[m_col].append(v)
                    if pt == "per_layer":
                        bl = clf_stats.get("best_layer")
                        if bl is not None:
                            fold_vals["best_layer"].append(int(bl))
                    fold_vals["n_test"].append(rec.get("n_test", 0))

                n_complete = len(fold_vals["accuracy"])
                if n_complete == 0:
                    continue

                row = {
                    "model":        model,
                    "probe_type":   pt,
                    "clf":          clf_name,
                    "n_folds_complete": n_complete,
                    "mean_n_test":  float(np.mean(fold_vals["n_test"])) if fold_vals["n_test"] else "",
                }
                for m_col in numeric_metrics:
                    vals = fold_vals[m_col]
                    if vals:
                        row[f"mean_{m_col}"] = float(np.mean(vals))
                        row[f"std_{m_col}"]  = float(np.std(vals))
                    else:
                        row[f"mean_{m_col}"] = ""
                        row[f"std_{m_col}"]  = ""
                if fold_vals["best_layer"]:
                    row["mean_best_layer"] = float(np.mean(fold_vals["best_layer"]))
                    row["std_best_layer"]  = float(np.std(fold_vals["best_layer"]))
                else:
                    row["mean_best_layer"] = ""
                    row["std_best_layer"]  = ""
                agg_rows.append(row)

    agg_fields = (["model", "probe_type", "clf", "n_folds_complete", "mean_n_test",
                   "mean_best_layer", "std_best_layer"] +
                  [f"{s}_{m}" for m in numeric_metrics for s in ("mean", "std")])
    agg_path = DATA_DIR / "kfold_results_aggregated.csv"
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerows(agg_rows)
    print(f"  Wrote {len(agg_rows)} rows → {agg_path}", flush=True)

    # ── CSV 3: per-layer stats for all layers (for per-layer plots) ───────────
    pl_rows = []
    for model in ALL_MODELS:
        for fold in range(N_FOLDS):
            rec = results[model].get(fold)
            if rec is None:
                continue
            pla = rec.get("per_layer_all", {})
            for clf_name in CLF_NAMES:
                for l_str, stats in pla.get(clf_name, {}).items():
                    row = {
                        "fold":  fold,
                        "model": model,
                        "clf":   clf_name,
                        "layer": int(l_str),
                    }
                    row.update(stats)
                    pl_rows.append(row)

    pl_path = DATA_DIR / "kfold_per_layer_all.csv"
    pl_fields = ["fold", "model", "clf", "layer",
                 "accuracy", "true_accuracy", "false_accuracy",
                 "precision", "recall", "f1", "auc"]
    with open(pl_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pl_fields)
        w.writeheader()
        w.writerows(pl_rows)
    print(f"  Wrote {len(pl_rows)} rows → {pl_path}", flush=True)

    # ── CSV 4: per-layer aggregated (mean ± std across folds per model/clf/layer) ─
    from collections import defaultdict
    pl_agg = defaultdict(lambda: defaultdict(list))  # (model,clf,layer,metric) -> [values]
    for row in pl_rows:
        key = (row["model"], row["clf"], row["layer"])
        for m_col in ("accuracy", "true_accuracy", "false_accuracy",
                      "precision", "recall", "f1", "auc"):
            v = _safe_float(row.get(m_col))
            if v is not None:
                pl_agg[key][m_col].append(v)

    pl_agg_rows = []
    for (model, clf_name, layer), m_dict in sorted(pl_agg.items()):
        row = {"model": model, "clf": clf_name, "layer": layer}
        for m_col in ("accuracy", "true_accuracy", "false_accuracy",
                      "precision", "recall", "f1", "auc"):
            vals = m_dict.get(m_col, [])
            row[f"mean_{m_col}"] = float(np.mean(vals)) if vals else ""
            row[f"std_{m_col}"]  = float(np.std(vals))  if vals else ""
        pl_agg_rows.append(row)

    pl_agg_path = DATA_DIR / "kfold_per_layer_aggregated.csv"
    pl_agg_fields = (["model", "clf", "layer"] +
                     [f"{s}_{m}" for m in ("accuracy", "true_accuracy", "false_accuracy",
                                           "precision", "recall", "f1", "auc")
                      for s in ("mean", "std")])
    with open(pl_agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pl_agg_fields)
        w.writeheader()
        w.writerows(pl_agg_rows)
    print(f"  Wrote {len(pl_agg_rows)} rows → {pl_agg_path}", flush=True)

    # Print a quick summary table to stdout
    print("\n[kfold_summary] Mean accuracy (mid_band LR) across folds:", flush=True)
    print(f"  {'Model':<15} {'mean_acc':>9} {'std_acc':>8}", flush=True)
    print("  " + "-" * 35, flush=True)
    for model in ALL_MODELS:
        vals = []
        for fold in range(N_FOLDS):
            rec = results[model].get(fold)
            if rec is None:
                continue
            v = _safe_float(rec["probe_stats"].get("mid_band", {}).get("LR", {}).get("accuracy"))
            if v is not None:
                vals.append(v)
        if vals:
            print(f"  {model:<15} {np.mean(vals):>9.4f} {np.std(vals):>8.4f}", flush=True)
        else:
            print(f"  {model:<15} {'N/A':>9}", flush=True)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="5-fold probe cross-validation")
    parser.add_argument("--stage", required=True,
                        choices=["kfold_train", "kfold_summary"],
                        help="Stage to run")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index 0-4 (required for kfold_train)")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model name (required for kfold_train). "
                             f"One of: {', '.join(ALL_MODELS)}")
    args = parser.parse_args()

    if args.stage == "kfold_train":
        if args.fold is None or args.model is None:
            parser.error("--fold and --model are required for kfold_train")
        if args.fold < 0 or args.fold >= N_FOLDS:
            parser.error(f"--fold must be 0-{N_FOLDS-1}")
        if args.model not in ALL_MODELS:
            parser.error(f"--model must be one of: {ALL_MODELS}")
        run_kfold_train(args.fold, args.model)

    elif args.stage == "kfold_summary":
        DATA_DIR.mkdir(exist_ok=True)
        run_kfold_summary()


if __name__ == "__main__":
    main()
