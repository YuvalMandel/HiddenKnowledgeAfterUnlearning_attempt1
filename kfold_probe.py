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
# 95 % CI helper (uses scipy.stats.t if available; else look-up table)
# =============================================================================
try:
    from scipy.stats import t as _t_dist
    def _ci95_half(sample_std: float, n: int) -> float:
        """Half-width of 95 % CI: t_{n-1, 0.975} × s / √n  (sample std, ddof=1)."""
        if n < 2:
            return float("nan")
        return float(_t_dist.ppf(0.975, df=n - 1) * sample_std / (n ** 0.5))
except ImportError:
    _T95 = {1: float("inf"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
            6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    def _ci95_half(sample_std: float, n: int) -> float:
        if n < 2:
            return float("nan")
        return float(_T95.get(n, 2.0) * sample_std / (n ** 0.5))

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
KFOLD_SWEEP_DIR = CHECKPOINT_DIR / "kfold_sweep"

# Models: base + 8 unlearning methods
ALL_MODELS    = ["base", "GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]
SWEEP_METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]
N_CHECKPOINTS = 8

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


def get_sweep_npy_paths(method: str, ck_num: int):
    """Return (train, val, test) npy paths for a sweep checkpoint."""
    sn = safe_name(method)
    ck_dir = CHECKPOINT_DIR / f"sweep_{sn}" / f"ck{ck_num}"
    return (ck_dir / "hs_train.npy",
            ck_dir / "hs_val.npy",
            ck_dir / "hs_test.npy")


def sweep_result_path(method: str, ck_num: int, fold: int) -> Path:
    sn = safe_name(method)
    d = KFOLD_SWEEP_DIR / f"sweep_{sn}" / f"ck{ck_num}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"f{fold}_results.json"


def load_hs_for_sweep(method: str, ck_num: int) -> np.ndarray:
    train_p, val_p, test_p = get_sweep_npy_paths(method, ck_num)
    missing = [p for p in (train_p, val_p, test_p) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing sweep npy files for {method} ck{ck_num}: {[str(p) for p in missing]}"
        )
    print(f"  Loading sweep hidden states for '{method}' ck{ck_num}...", flush=True)
    hs_tr = np.load(train_p)
    hs_va = np.load(val_p)
    hs_te = np.load(test_p)
    print(f"    train {hs_tr.shape}  val {hs_va.shape}  test {hs_te.shape}", flush=True)
    return np.concatenate([hs_tr, hs_va, hs_te], axis=0)


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

def _run_kfold_core(fold: int, hs_all: np.ndarray, out_path: Path,
                    label: str, by_qid: dict, folds: list, fold_size: int):
    """Shared probe-train + evaluate logic. Called by both kfold_train and kfold_sweep_train."""
    if out_path.exists():
        print(f"  [skip] Result already exists: {out_path}", flush=True)
        return

    (tr_idx, tr_lbl, va_idx, va_lbl, te_idx, te_lbl) = \
        get_split_indices(fold, by_qid, folds)
    print(f"  Split sizes — train: {len(tr_idx)}  val: {len(va_idx)}  test: {len(te_idx)}",
          flush=True)

    hs_train = hs_all[tr_idx]
    hs_val   = hs_all[va_idx]
    hs_test  = hs_all[te_idx]

    print(f"\n  Training probes...", flush=True)
    probe_set = train_probe_set(hs_train, tr_lbl, hs_val, va_lbl, label=label)

    print(f"\n  Evaluating on test set...", flush=True)
    probe_stats   = compute_all_probe_stats(probe_set, hs_test, te_lbl)
    per_layer_all = compute_all_layers_probe_stats(probe_set, hs_test, te_lbl)

    record = {
        "fold":         fold,
        "label":        label,
        "n_train":      int(len(tr_idx)),
        "n_val":        int(len(va_idx)),
        "n_test":       int(len(te_idx)),
        "fold_size_q":  int(fold_size),
        "probe_stats":  probe_stats,
        "per_layer_all": {
            clf: {str(l): v for l, v in layers.items()}
            for clf, layers in per_layer_all.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)


def run_kfold_train(fold: int, model: str):
    print(f"\n{'='*60}", flush=True)
    print(f"  kfold_train  fold={fold}  model={model}", flush=True)
    print(f"{'='*60}", flush=True)

    out_path = result_path(fold, model)
    if out_path.exists():
        print(f"  [skip] Result already exists: {out_path}", flush=True)
        return

    print("  Building fold assignment from CSV...", flush=True)
    by_qid, folds, _, _, fold_size = build_fold_assignment()

    hs_all = load_hs_for_model(model)
    _run_kfold_core(fold, hs_all, out_path,
                    label=f"f{fold}_{model}",
                    by_qid=by_qid, folds=folds, fold_size=fold_size)


# =============================================================================
# Stage: kfold_sweep_train
# =============================================================================

def run_kfold_sweep_train(method: str, ck_num: int, fold: int):
    print(f"\n{'='*60}", flush=True)
    print(f"  kfold_sweep_train  method={method}  ck={ck_num}  fold={fold}", flush=True)
    print(f"{'='*60}", flush=True)

    out_path = sweep_result_path(method, ck_num, fold)
    if out_path.exists():
        print(f"  [skip] Result already exists: {out_path}", flush=True)
        return

    print("  Building fold assignment from CSV...", flush=True)
    by_qid, folds, _, _, fold_size = build_fold_assignment()

    hs_all = load_hs_for_sweep(method, ck_num)
    _run_kfold_core(fold, hs_all, out_path,
                    label=f"f{fold}_{method}_ck{ck_num}",
                    by_qid=by_qid, folds=folds, fold_size=fold_size)


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
                        row[f"mean_{m_col}"]      = float(np.mean(vals))
                        row[f"std_{m_col}"]       = float(np.std(vals))
                        ssd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
                        row[f"ci95_half_{m_col}"] = _ci95_half(ssd, n_complete)
                    else:
                        row[f"mean_{m_col}"]      = ""
                        row[f"std_{m_col}"]       = ""
                        row[f"ci95_half_{m_col}"] = ""
                if fold_vals["best_layer"]:
                    bls = fold_vals["best_layer"]
                    row["mean_best_layer"]      = float(np.mean(bls))
                    row["std_best_layer"]       = float(np.std(bls))
                    ssd = float(np.std(bls, ddof=1)) if len(bls) >= 2 else 0.0
                    row["ci95_half_best_layer"] = _ci95_half(ssd, len(bls))
                else:
                    row["mean_best_layer"]      = ""
                    row["std_best_layer"]       = ""
                    row["ci95_half_best_layer"] = ""
                agg_rows.append(row)

    agg_fields = (["model", "probe_type", "clf", "n_folds_complete", "mean_n_test",
                   "mean_best_layer", "std_best_layer", "ci95_half_best_layer"] +
                  [f"{s}_{m}" for m in numeric_metrics
                   for s in ("mean", "std", "ci95_half")])
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
            if vals:
                row[f"mean_{m_col}"]      = float(np.mean(vals))
                row[f"std_{m_col}"]       = float(np.std(vals))
                ssd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
                row[f"ci95_half_{m_col}"] = _ci95_half(ssd, len(vals))
            else:
                row[f"mean_{m_col}"]      = ""
                row[f"std_{m_col}"]       = ""
                row[f"ci95_half_{m_col}"] = ""
        pl_agg_rows.append(row)

    pl_agg_path = DATA_DIR / "kfold_per_layer_aggregated.csv"
    pl_agg_fields = (["model", "clf", "layer"] +
                     [f"{s}_{m}" for m in ("accuracy", "true_accuracy", "false_accuracy",
                                           "precision", "recall", "f1", "auc")
                      for s in ("mean", "std", "ci95_half")])
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
# K-fold wide-format table (mirrors single-fold summary_table3_method_probes.csv)
# =============================================================================

# Abbreviation maps — must match hidden_knowledge_after_unlearning.py
_PT_ABBR = {
    "per_layer":     "pl",
    "mid_band":      "mb",
    "vote_ensemble": "vote",
    "avg_ensemble":  "avg",
    "full_layer":    "fl",
    "init_band":     "ib",
    "init_band_emb": "ibe",
    "end_band":      "eb",
    "ib_no_pca":     "ibnp",
}
_METRIC_ABBR = {
    "accuracy":       "acc",
    "true_accuracy":  "true",
    "false_accuracy": "fals",
    "precision":      "prec",
    "recall":         "rec",
    "f1":             "f1",
    "auc":            "auc",
}
_CLF_ABBR = {"LR": "lr", "RF": "rf", "AdaBoost": "ada"}

_NUMERIC_METRICS = ["accuracy", "true_accuracy", "false_accuracy",
                    "precision", "recall", "f1", "auc"]


def run_kfold_tables():
    """
    Read kfold_results_aggregated.csv and kfold_per_layer_aggregated.csv and
    produce a wide-format CSV that mirrors the structure of
    summary_table3_method_probes.csv (each row = one model; each probe×clf×metric
    triple becomes three columns: _mean, _std, _ci95).

    Output: data/kfold_table3_probes.csv
    Also writes: data/kfold_per_layer_table.csv (per-layer means ± CI, one row
    per model×clf×layer for easy plotting).
    """
    agg_path = DATA_DIR / "kfold_results_aggregated.csv"
    pl_path  = DATA_DIR / "kfold_per_layer_aggregated.csv"

    if not agg_path.exists():
        print(f"[kfold_tables] {agg_path} not found — run --stage kfold_summary first.",
              flush=True)
        return

    # ── Load aggregated CSV into nested dict ─────────────────────────────────
    # agg[model][probe_type][clf] = {mean_accuracy: float, std_accuracy: float, ...}
    agg = {}
    with open(agg_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m   = row["model"]
            pt  = row["probe_type"]
            clf = row["clf"]
            agg.setdefault(m, {}).setdefault(pt, {})[clf] = row

    # ── Build column list ─────────────────────────────────────────────────────
    # Order: same probe-type order as PROBE_TYPES, same metric/clf order as main script.
    # Each column cell expands to three: _mean, _std, _ci95.
    cols = ["model"]
    for pt in PROBE_TYPES:
        pa = _PT_ABBR[pt]
        for clf in CLF_NAMES:
            ca = _CLF_ABBR[clf]
            if pt == "per_layer":
                for stat in ("mean", "std", "ci95"):
                    cols.append(f"{pa}_{ca}_lyr_{stat}")
            for ma_key, ma in _METRIC_ABBR.items():
                for stat in ("mean", "std", "ci95"):
                    cols.append(f"{pa}_{ca}_{ma}_{stat}")

    # ── Build rows ────────────────────────────────────────────────────────────
    rows = []
    for model in ALL_MODELS:
        row = {"model": model}
        for pt in PROBE_TYPES:
            pa = _PT_ABBR[pt]
            for clf in CLF_NAMES:
                ca  = _CLF_ABBR[clf]
                src = agg.get(model, {}).get(pt, {}).get(clf, {})
                if pt == "per_layer":
                    for stat, src_key in (("mean", "mean_best_layer"),
                                          ("std",  "std_best_layer"),
                                          ("ci95", "ci95_half_best_layer")):
                        v = src.get(src_key, "")
                        row[f"{pa}_{ca}_lyr_{stat}"] = v
                for ma_key, ma in _METRIC_ABBR.items():
                    for stat, src_key in (("mean", f"mean_{ma_key}"),
                                          ("std",  f"std_{ma_key}"),
                                          ("ci95", f"ci95_half_{ma_key}")):
                        v = src.get(src_key, "")
                        row[f"{pa}_{ca}_{ma}_{stat}"] = v
        rows.append(row)

    out_path = DATA_DIR / "kfold_table3_probes.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[kfold_tables] Wrote {len(rows)} rows × {len(cols)} cols → {out_path}",
          flush=True)

    # ── Per-layer table (for kfold-aware line plots) ──────────────────────────
    if not pl_path.exists():
        print(f"[kfold_tables] {pl_path} not found — skipping per-layer table.",
              flush=True)
        return

    # Pass-through: pl_path already has model/clf/layer + mean_*/std_*/ci95_half_* columns.
    # Rename ci95_half_* → ci95_* for plot-script consistency, write kfold_per_layer_table.csv.
    pl_rows = []
    with open(pl_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        orig_fields = reader.fieldnames or []
        for row in reader:
            new_row = {}
            for k, v in row.items():
                new_key = k.replace("ci95_half_", "ci95_")
                new_row[new_key] = v
            pl_rows.append(new_row)

    pl_out_fields = [k.replace("ci95_half_", "ci95_") for k in orig_fields]
    pl_out_path = DATA_DIR / "kfold_per_layer_table.csv"
    with open(pl_out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pl_out_fields)
        w.writeheader()
        w.writerows(pl_rows)
    print(f"[kfold_tables] Wrote {len(pl_rows)} rows → {pl_out_path}", flush=True)

    # ── Print quick summary table ─────────────────────────────────────────────
    print("\n[kfold_tables] mid_band LR accuracy  (mean ± CI95):", flush=True)
    print(f"  {'Model':<15} {'mean':>8} {'± CI95':>8}", flush=True)
    print("  " + "─" * 35, flush=True)
    for model in ALL_MODELS:
        src = agg.get(model, {}).get("mid_band", {}).get("LR", {})
        mean_v = src.get("mean_accuracy", "")
        ci_v   = src.get("ci95_half_accuracy", "")
        try:
            print(f"  {model:<15} {float(mean_v):>8.4f} {float(ci_v):>8.4f}", flush=True)
        except (ValueError, TypeError):
            print(f"  {model:<15} {'N/A':>8}", flush=True)


# =============================================================================
# Stage: kfold_sweep_summary
# =============================================================================

def run_kfold_sweep_summary():
    """
    Aggregate kfold_sweep per-fold JSONs across all (method, ck) pairs.

    Reads:  checkpoints/kfold_sweep/sweep_{method}/ck{N}/f{fold}_results.json
    Writes:
      data/kfold_sweep_results_per_fold.csv
      data/kfold_sweep_results_aggregated.csv
      data/kfold_sweep_per_layer_all.csv
      data/kfold_sweep_per_layer_aggregated.csv
    """
    print("\n[kfold_sweep_summary] Loading results...", flush=True)

    # results[method][ck_num][fold] = record
    results = {m: {ck: {} for ck in range(1, N_CHECKPOINTS + 1)}
               for m in SWEEP_METHODS}
    missing = []
    for method in SWEEP_METHODS:
        for ck_num in range(1, N_CHECKPOINTS + 1):
            for fold in range(N_FOLDS):
                p = sweep_result_path(method, ck_num, fold)
                if not p.exists():
                    missing.append(str(p))
                    continue
                with open(p) as f:
                    results[method][ck_num][fold] = json.load(f)

    if missing:
        print(f"  WARNING: {len(missing)} result file(s) missing:", flush=True)
        for m in missing[:10]:
            print(f"    {m}", flush=True)
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more", flush=True)

    numeric_metrics = ["accuracy", "true_accuracy", "false_accuracy",
                       "precision", "recall", "f1", "auc"]

    # ── CSV 1: per-fold rows ──────────────────────────────────────────────────
    pf_rows = []
    for method in SWEEP_METHODS:
        for ck_num in range(1, N_CHECKPOINTS + 1):
            for fold in range(N_FOLDS):
                rec = results[method][ck_num].get(fold)
                if rec is None:
                    continue
                ps = rec["probe_stats"]
                for pt in PROBE_TYPES:
                    pt_stats = ps.get(pt, {})
                    for clf_name in CLF_NAMES:
                        s = pt_stats.get(clf_name, {})
                        row = {
                            "method": method, "ck": ck_num,
                            "fold": fold, "probe_type": pt, "clf": clf_name,
                            "n_train": rec.get("n_train", ""),
                            "n_val":   rec.get("n_val",   ""),
                            "n_test":  rec.get("n_test",  ""),
                            "best_layer": s.get("best_layer", ""),
                        }
                        for mc in METRIC_COLS:
                            row[mc] = s.get(mc, "")
                        pf_rows.append(row)

    pf_fields = ["method", "ck", "fold", "probe_type", "clf",
                 "n_train", "n_val", "n_test", "best_layer"] + METRIC_COLS
    pf_path = DATA_DIR / "kfold_sweep_results_per_fold.csv"
    with open(pf_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pf_fields)
        w.writeheader(); w.writerows(pf_rows)
    print(f"  Wrote {len(pf_rows)} rows → {pf_path}", flush=True)

    # ── CSV 2: aggregated (mean ± std ± CI95 across folds) ───────────────────
    agg_rows = []
    for method in SWEEP_METHODS:
        for ck_num in range(1, N_CHECKPOINTS + 1):
            for pt in PROBE_TYPES:
                for clf_name in CLF_NAMES:
                    fold_vals = {m: [] for m in numeric_metrics + ["best_layer", "n_test"]}
                    for fold in range(N_FOLDS):
                        rec = results[method][ck_num].get(fold)
                        if rec is None:
                            continue
                        s = rec["probe_stats"].get(pt, {}).get(clf_name, {})
                        for mc in numeric_metrics:
                            v = _safe_float(s.get(mc))
                            if v is not None:
                                fold_vals[mc].append(v)
                        if pt == "per_layer":
                            bl = s.get("best_layer")
                            if bl is not None:
                                fold_vals["best_layer"].append(int(bl))
                        fold_vals["n_test"].append(rec.get("n_test", 0))

                    n_complete = len(fold_vals["accuracy"])
                    if n_complete == 0:
                        continue

                    row = {
                        "method": method, "ck": ck_num,
                        "probe_type": pt, "clf": clf_name,
                        "n_folds_complete": n_complete,
                        "mean_n_test": float(np.mean(fold_vals["n_test"])) if fold_vals["n_test"] else "",
                    }
                    for mc in numeric_metrics:
                        vals = fold_vals[mc]
                        if vals:
                            ssd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
                            row[f"mean_{mc}"]      = float(np.mean(vals))
                            row[f"std_{mc}"]       = float(np.std(vals))
                            row[f"ci95_half_{mc}"] = _ci95_half(ssd, n_complete)
                        else:
                            row[f"mean_{mc}"] = row[f"std_{mc}"] = row[f"ci95_half_{mc}"] = ""
                    if fold_vals["best_layer"]:
                        bls = fold_vals["best_layer"]
                        ssd = float(np.std(bls, ddof=1)) if len(bls) >= 2 else 0.0
                        row["mean_best_layer"]      = float(np.mean(bls))
                        row["std_best_layer"]       = float(np.std(bls))
                        row["ci95_half_best_layer"] = _ci95_half(ssd, len(bls))
                    else:
                        row["mean_best_layer"] = row["std_best_layer"] = row["ci95_half_best_layer"] = ""
                    agg_rows.append(row)

    agg_fields = (["method", "ck", "probe_type", "clf", "n_folds_complete", "mean_n_test",
                   "mean_best_layer", "std_best_layer", "ci95_half_best_layer"] +
                  [f"{s}_{m}" for m in numeric_metrics for s in ("mean", "std", "ci95_half")])
    agg_path = DATA_DIR / "kfold_sweep_results_aggregated.csv"
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader(); w.writerows(agg_rows)
    print(f"  Wrote {len(agg_rows)} rows → {agg_path}", flush=True)

    # ── CSV 3: per-layer all folds ─────────────────────────────────────────────
    pl_rows = []
    for method in SWEEP_METHODS:
        for ck_num in range(1, N_CHECKPOINTS + 1):
            for fold in range(N_FOLDS):
                rec = results[method][ck_num].get(fold)
                if rec is None:
                    continue
                for clf_name in CLF_NAMES:
                    for l_str, stats in rec.get("per_layer_all", {}).get(clf_name, {}).items():
                        row = {"method": method, "ck": ck_num,
                               "fold": fold, "clf": clf_name, "layer": int(l_str)}
                        row.update(stats)
                        pl_rows.append(row)

    pl_path = DATA_DIR / "kfold_sweep_per_layer_all.csv"
    pl_fields = ["method", "ck", "fold", "clf", "layer",
                 "accuracy", "true_accuracy", "false_accuracy",
                 "precision", "recall", "f1", "auc"]
    with open(pl_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pl_fields)
        w.writeheader(); w.writerows(pl_rows)
    print(f"  Wrote {len(pl_rows)} rows → {pl_path}", flush=True)

    # ── CSV 4: per-layer aggregated ───────────────────────────────────────────
    from collections import defaultdict
    pl_agg: dict = defaultdict(lambda: defaultdict(list))
    for row in pl_rows:
        key = (row["method"], row["ck"], row["clf"], row["layer"])
        for mc in ("accuracy", "true_accuracy", "false_accuracy",
                   "precision", "recall", "f1", "auc"):
            v = _safe_float(row.get(mc))
            if v is not None:
                pl_agg[key][mc].append(v)

    pl_agg_rows = []
    for (method, ck_num, clf_name, layer), m_dict in sorted(pl_agg.items()):
        row = {"method": method, "ck": ck_num, "clf": clf_name, "layer": layer}
        for mc in ("accuracy", "true_accuracy", "false_accuracy",
                   "precision", "recall", "f1", "auc"):
            vals = m_dict.get(mc, [])
            if vals:
                ssd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
                row[f"mean_{mc}"]      = float(np.mean(vals))
                row[f"std_{mc}"]       = float(np.std(vals))
                row[f"ci95_{mc}"]      = _ci95_half(ssd, len(vals))
            else:
                row[f"mean_{mc}"] = row[f"std_{mc}"] = row[f"ci95_{mc}"] = ""
        pl_agg_rows.append(row)

    pl_agg_path = DATA_DIR / "kfold_sweep_per_layer_aggregated.csv"
    pl_agg_fields = (["method", "ck", "clf", "layer"] +
                     [f"{s}_{m}" for m in ("accuracy", "true_accuracy", "false_accuracy",
                                           "precision", "recall", "f1", "auc")
                      for s in ("mean", "std", "ci95")])
    with open(pl_agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pl_agg_fields)
        w.writeheader(); w.writerows(pl_agg_rows)
    print(f"  Wrote {len(pl_agg_rows)} rows → {pl_agg_path}", flush=True)

    # Quick summary
    print("\n[kfold_sweep_summary] mid_band LR accuracy (mean ± CI95) at ck8:", flush=True)
    print(f"  {'Method':<15} {'mean':>8} {'± CI95':>8}", flush=True)
    print("  " + "─" * 35, flush=True)
    for method in SWEEP_METHODS:
        for row in agg_rows:
            if (row["method"] == method and row["ck"] == 8
                    and row["probe_type"] == "mid_band" and row["clf"] == "LR"):
                try:
                    print(f"  {method:<15} {float(row['mean_accuracy']):>8.4f}"
                          f" {float(row['ci95_half_accuracy']):>8.4f}", flush=True)
                except (ValueError, TypeError):
                    print(f"  {method:<15} {'N/A':>8}", flush=True)
                break


# =============================================================================
# Stage: kfold_sweep_tables
# =============================================================================

def run_kfold_sweep_tables():
    """
    Pivot kfold_sweep_results_aggregated.csv into wide-format tables for plot scripts.

    Writes:
      data/kfold_sweep_table_probes.csv   — rows: (method, ck); cols: pt_clf_metric_{mean/std/ci95}
      data/kfold_sweep_per_layer_table.csv — rename ci95_half → ci95 in per_layer_aggregated
    """
    agg_path = DATA_DIR / "kfold_sweep_results_aggregated.csv"
    pl_path  = DATA_DIR / "kfold_sweep_per_layer_aggregated.csv"

    if not agg_path.exists():
        print(f"[kfold_sweep_tables] {agg_path} not found — run kfold_sweep_summary first.",
              flush=True)
        return

    # Load aggregated → nested dict: agg[method][ck][pt][clf] = row
    agg: dict = {}
    with open(agg_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m  = row["method"]
            ck = int(row["ck"])
            pt = row["probe_type"]
            cl = row["clf"]
            agg.setdefault(m, {}).setdefault(ck, {}).setdefault(pt, {})[cl] = row

    # Build column list (same structure as kfold_table3_probes.csv)
    cols = ["method", "ck"]
    for pt in PROBE_TYPES:
        pa = _PT_ABBR[pt]
        for clf in CLF_NAMES:
            ca = _CLF_ABBR[clf]
            if pt == "per_layer":
                for stat in ("mean", "std", "ci95"):
                    cols.append(f"{pa}_{ca}_lyr_{stat}")
            for ma_key, ma in _METRIC_ABBR.items():
                for stat in ("mean", "std", "ci95"):
                    cols.append(f"{pa}_{ca}_{ma}_{stat}")

    rows = []
    for method in SWEEP_METHODS:
        for ck_num in range(1, N_CHECKPOINTS + 1):
            row = {"method": method, "ck": ck_num}
            for pt in PROBE_TYPES:
                pa = _PT_ABBR[pt]
                for clf in CLF_NAMES:
                    ca  = _CLF_ABBR[clf]
                    src = agg.get(method, {}).get(ck_num, {}).get(pt, {}).get(clf, {})
                    if pt == "per_layer":
                        for stat, src_key in (("mean", "mean_best_layer"),
                                              ("std",  "std_best_layer"),
                                              ("ci95", "ci95_half_best_layer")):
                            row[f"{pa}_{ca}_lyr_{stat}"] = src.get(src_key, "")
                    for ma_key, ma in _METRIC_ABBR.items():
                        for stat, src_key in (("mean", f"mean_{ma_key}"),
                                              ("std",  f"std_{ma_key}"),
                                              ("ci95", f"ci95_half_{ma_key}")):
                            row[f"{pa}_{ca}_{ma}_{stat}"] = src.get(src_key, "")
            rows.append(row)

    out_path = DATA_DIR / "kfold_sweep_table_probes.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"[kfold_sweep_tables] Wrote {len(rows)} rows × {len(cols)} cols → {out_path}",
          flush=True)

    # Per-layer table (ci95_half_ → ci95_ rename for plot-script consistency)
    if not pl_path.exists():
        print(f"[kfold_sweep_tables] {pl_path} not found — skipping per-layer table.",
              flush=True)
        return

    pl_rows, orig_fields = [], None
    with open(pl_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        orig_fields = reader.fieldnames or []
        for row in reader:
            pl_rows.append({k.replace("ci95_half_", "ci95_"): v for k, v in row.items()})

    pl_out_fields = [k.replace("ci95_half_", "ci95_") for k in orig_fields]
    pl_out_path = DATA_DIR / "kfold_sweep_per_layer_table.csv"
    with open(pl_out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pl_out_fields)
        w.writeheader(); w.writerows(pl_rows)
    print(f"[kfold_sweep_tables] Wrote {len(pl_rows)} rows → {pl_out_path}", flush=True)

    # Quick summary
    print("\n[kfold_sweep_tables] mid_band LR accuracy  (mean ± CI95) at ck8:", flush=True)
    print(f"  {'Method':<15} {'mean':>8} {'± CI95':>8}", flush=True)
    print("  " + "─" * 35, flush=True)
    for method in SWEEP_METHODS:
        src = agg.get(method, {}).get(8, {}).get("mid_band", {}).get("LR", {})
        try:
            print(f"  {method:<15} {float(src['mean_accuracy']):>8.4f}"
                  f" {float(src['ci95_half_accuracy']):>8.4f}", flush=True)
        except (ValueError, TypeError, KeyError):
            print(f"  {method:<15} {'N/A':>8}", flush=True)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="5-fold probe cross-validation")
    parser.add_argument("--stage", required=True,
                        choices=["kfold_train", "kfold_summary", "kfold_tables",
                                 "kfold_sweep_train", "kfold_sweep_summary",
                                 "kfold_sweep_tables"],
                        help="Stage to run")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index 0-4 (required for kfold_train / kfold_sweep_train)")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model name (required for kfold_train). "
                             f"One of: {', '.join(ALL_MODELS)}")
    parser.add_argument("--method", type=str, default=None,
                        help=f"Unlearning method (required for kfold_sweep_train). "
                             f"One of: {', '.join(SWEEP_METHODS)}")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Checkpoint number 1-8 (required for kfold_sweep_train)")
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

    elif args.stage == "kfold_tables":
        DATA_DIR.mkdir(exist_ok=True)
        run_kfold_tables()

    elif args.stage == "kfold_sweep_train":
        if args.fold is None or args.method is None or args.checkpoint is None:
            parser.error("--fold, --method, and --checkpoint are required for kfold_sweep_train")
        if args.fold < 0 or args.fold >= N_FOLDS:
            parser.error(f"--fold must be 0-{N_FOLDS-1}")
        if args.method not in SWEEP_METHODS:
            parser.error(f"--method must be one of: {SWEEP_METHODS}")
        if args.checkpoint < 1 or args.checkpoint > N_CHECKPOINTS:
            parser.error(f"--checkpoint must be 1-{N_CHECKPOINTS}")
        run_kfold_sweep_train(args.method, args.checkpoint, args.fold)

    elif args.stage == "kfold_sweep_summary":
        DATA_DIR.mkdir(exist_ok=True)
        run_kfold_sweep_summary()

    elif args.stage == "kfold_sweep_tables":
        DATA_DIR.mkdir(exist_ok=True)
        run_kfold_sweep_tables()


if __name__ == "__main__":
    main()
