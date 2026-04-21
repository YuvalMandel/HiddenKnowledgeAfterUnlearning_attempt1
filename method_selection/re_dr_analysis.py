#!/usr/bin/env python3
"""
re_dr_analysis.py  —  RE / DR dual-path band analysis for one unlearning method.

Adapted from re_dr_dual_path_robust_cv.py (Handover: Start Here) to use the
file naming conventions of this project's pipeline:

  checkpoints/base_hs_{train,val,test}.npy        — base hidden states
  checkpoints/{METHOD}_hs_{train,val,test}.npy    — method hidden states
  checkpoints/bio_labels.csv                      — gold labels (run export_bio_labels_csv.py first)

For each (method, band) it computes:
  RE  — Normalised Erasure AUC: how much did probe AUC drop after unlearning?
  DR  — Deletion Readout: cosine distance between base and method probe weights.
  frozen recovery rate   — base probe (frozen) applied to method states: still classifies?
  retrained recovery rate — method probe retrained on method states: still classifies?

Saves per-example band prediction CSVs needed by build_authoritative_measurements.py
and scatter plots showing the RE/DR trade-off across all 8 methods.

Usage (one method per SLURM task):
  python method_selection/re_dr_analysis.py --method GradDiff
  python method_selection/re_dr_analysis.py --method PB_J

Aggregation step (after all method tasks finish):
  python method_selection/build_authoritative_measurements.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# ── Project-level constants ───────────────────────────────────────────────────

ALL_METHODS = [
    "GradDiff", "RMU", "RMU-LAT", "RepNoise",
    "ELM", "RR", "TAR", "PB_J",
]

DEFAULT_DATA_DIR    = "checkpoints"
DEFAULT_LABELS_CSV  = "checkpoints/bio_labels.csv"
DEFAULT_OUT_ROOT    = "method_selection_out/re_dr"
DEFAULT_BASE_PREFIX = "base_hs"
DEFAULT_METHOD_SUFFIX = "_hs"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SplitData:
    train: np.ndarray
    val:   np.ndarray
    test:  np.ndarray


@dataclass
class ProbeResult:
    weights:   np.ndarray
    intercept: float
    y_true:    np.ndarray
    score:     np.ndarray
    pred:      np.ndarray
    acc:       float
    auc:       float


@dataclass
class CachedProbe:
    weights:   np.ndarray
    intercept: float
    metadata:  dict


@dataclass
class BandMetrics:
    band_name:   str
    layers:      List[int]
    n_eval:      int
    n_pos:       int
    n_neg:       int
    re_auc_valid:      bool
    re_auc_nan_reason: str
    base_probe_on_base_eval:         ProbeResult
    method_probe_on_method_eval:     ProbeResult
    frozen_base_probe_on_method_eval: ProbeResult
    dr_weight_cos:          float
    dr_score_cos:           float
    re_acc:                 float
    re_auc:                 float
    retrained_recovered_count: int
    retrained_recovered_rate:  float
    frozen_recovered_count:    int
    frozen_recovered_rate:     float
    recovery_gain_count:       int
    recovery_gain_rate:        float


# ── Utilities ─────────────────────────────────────────────────────────────────

def safe_auc(y_true: np.ndarray, score: np.ndarray) -> Tuple[float, bool, str]:
    unique = np.unique(y_true)
    if unique.size < 2:
        return float("nan"), False, "single_class_y_true"
    if np.std(score) < 1e-12:
        return float("nan"), False, "degenerate_scores"
    return float(roc_auc_score(y_true, score)), True, "ok"


def cosine_distance(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < eps or n2 < eps:
        return float("nan")
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return 1.0 - max(-1.0, min(1.0, cos))


def normalized_erasure(base_metric: float, post_metric: float, chance: float = 0.5) -> float:
    if np.isnan(base_metric) or np.isnan(post_metric):
        return float("nan")
    denom = base_metric - chance
    if abs(denom) < 1e-12:
        return float("nan")
    return (base_metric - post_metric) / denom


def load_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Could not load ndarray from: {path}")
    return arr


def load_split_data(prefix_dir: str, prefix: str) -> SplitData:
    return SplitData(
        train=load_npy(os.path.join(prefix_dir, f"{prefix}_train.npy")),
        val=load_npy(os.path.join(prefix_dir, f"{prefix}_val.npy")),
        test=load_npy(os.path.join(prefix_dir, f"{prefix}_test.npy")),
    )


# ── Labels ────────────────────────────────────────────────────────────────────

def load_labels_df(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if "split" not in df.columns:
        raise ValueError("labels CSV must contain a 'split' column")
    return df


def to_binary(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        vals = series.astype(int).to_numpy()
        if set(np.unique(vals)).issubset({0, 1}):
            return vals
        raise ValueError(f"Numeric label must be binary 0/1, got: {sorted(set(vals))}")
    s = series.astype(str).str.strip().str.lower()
    mapping = {"true": 1, "t": 1, "1": 1, "yes": 1,
               "false": 0, "f": 0, "0": 0, "no": 0}
    unknown = sorted(set(s.unique()) - set(mapping.keys()))
    if unknown:
        raise ValueError(f"Unsupported labels: {unknown}")
    return s.map(mapping).astype(int).to_numpy()


def load_labels_from_csv(labels_csv: str) -> SplitData:
    df = load_labels_df(labels_csv)
    label_col = next(
        (c for c in ["gold_label", "label", "answer", "is_true", "target"] if c in df.columns),
        None
    )
    if label_col is None:
        raise ValueError(f"No label column found. Available: {list(df.columns)}")
    split_str = df["split"].astype(str).str.lower()
    return SplitData(
        train=to_binary(df.loc[split_str == "train", label_col]),
        val=to_binary(df.loc[split_str == "val", label_col]),
        test=to_binary(df.loc[split_str.eq("test") | split_str.str.match(r"^test\d+$", na=False), label_col]),
    )


# ── Band / split helpers ──────────────────────────────────────────────────────

def default_band_indices(num_layers: int, band_name: str) -> List[int]:
    one_third  = num_layers // 3
    two_thirds = 2 * one_third
    if band_name == "early":  return list(range(0, one_third))
    if band_name == "mid":    return list(range(one_third, two_thirds))
    if band_name == "late":   return list(range(two_thirds, num_layers))
    raise ValueError(f"Unknown band: {band_name}")


def parse_layer_spec(spec: Optional[str], num_layers: int, band_name: str) -> List[int]:
    if not spec or str(spec).strip() == "":
        return default_band_indices(num_layers, band_name)
    layers = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    uniq = sorted(set(layers))
    bad = [x for x in uniq if x < 0 or x >= num_layers]
    if bad:
        raise ValueError(f"Layer indices out of bounds (num_layers={num_layers}): {bad}")
    return uniq


def extract_band_features(hs: np.ndarray, band_layers: List[int], mode: str = "concat") -> np.ndarray:
    if hs.ndim == 2:
        return hs
    X = hs[:, band_layers, :]
    if mode == "concat":  return X.reshape(X.shape[0], -1)
    if mode == "mean":    return X.mean(axis=1)
    if mode == "max":     return X.max(axis=1)
    raise ValueError(f"Unsupported band mode: {mode}")


def resolve_eval_source(eval_split: str) -> str:
    s = eval_split.lower()
    if s == "val":
        return "val"
    if s == "test" or re.match(r"^test\d+$", s) or re.match(r"^testw\d+$", s):
        return "test"
    raise ValueError(f"Unsupported eval_split: {eval_split}")


def get_split_array(split_data: SplitData, eval_split: str) -> np.ndarray:
    return split_data.val if resolve_eval_source(eval_split) == "val" else split_data.test


def get_train_window_indices(train_scheme: str, train_window_id: str, n_train: int) -> Optional[np.ndarray]:
    if train_scheme == "full":
        return None
    if train_scheme != "traincv":
        raise ValueError(f"Unsupported train_scheme: {train_scheme}")
    m = re.match(r"^traincv(\d+)$", train_window_id.lower())
    if not m:
        raise ValueError("For traincv, train_window_id must be traincv1..traincv6")
    k = int(m.group(1))
    if not (1 <= k <= 6):
        raise ValueError("train_window_id must be traincv1..traincv6")
    start = (k - 1) * 100
    end   = start + 500
    if end > n_train:
        raise ValueError(f"Train CV window {train_window_id} exceeds n_train={n_train}")
    return np.arange(start, end, dtype=int)


def load_subset_indices(subset_csv_path: str, expected_n: int) -> np.ndarray:
    df = pd.read_csv(subset_csv_path)
    if len(df) == 0:
        return np.array([], dtype=int)
    idx_col = next((c for c in ["base_row_id", "row_id"] if c in df.columns), None)
    if idx_col is None:
        raise ValueError(f"Subset CSV must contain base_row_id or row_id. Columns: {list(df.columns)}")
    idx = df[idx_col].astype(int).to_numpy()
    bad = idx[(idx < 0) | (idx >= expected_n)]
    if len(bad) > 0:
        raise ValueError(f"Subset indices out of bounds: {sorted(set(bad.tolist()))[:10]}")
    return idx


def load_eval_indices_from_labels(labels_csv: str, eval_split: str, expected_n: int) -> Optional[np.ndarray]:
    s = eval_split.lower()
    if s in {"val", "test"}:
        return None
    if re.match(r"^test\d+$", s):
        df = load_labels_df(labels_csv)
        split_str = df["split"].astype(str).str.lower()
        test_df = df.loc[split_str.eq("test") | split_str.str.match(r"^test\d+$", na=False)].reset_index(drop=True)
        idx = np.where(test_df["split"].astype(str).str.lower().to_numpy() == s)[0]
        if len(idx) == 0:
            raise ValueError(f"No rows found for eval_split={eval_split}")
        return idx.astype(int)
    m = re.match(r"^testw(\d+)$", s)
    if m:
        k = int(m.group(1))
        if not (1 <= k <= 7):
            raise ValueError("test window must be testw1..testw7")
        start = (k - 1) * 100
        end   = start + 520
        if end > expected_n:
            raise ValueError(f"Window {eval_split} exceeds expected_n={expected_n}")
        return np.arange(start, end, dtype=int)
    raise ValueError(f"Unsupported eval_split: {eval_split}")


# ── Probe caching ──────────────────────────────────────────────────────────────

def _sha1_array(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()


def _probe_cache_key(probe_owner, train_scheme, train_window_id,
                     band_name, band_layers, band_mode,
                     seed, c_value, max_iter, X_train, y_train) -> str:
    payload = {
        "probe_owner": probe_owner, "train_scheme": train_scheme,
        "train_window_id": train_window_id, "band_name": band_name,
        "band_layers": list(band_layers), "band_mode": band_mode,
        "seed": seed, "C": c_value, "max_iter": max_iter,
        "x_shape": tuple(X_train.shape), "y_shape": tuple(y_train.shape),
        "x_sha1": _sha1_array(X_train), "y_sha1": _sha1_array(y_train),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.npz")


def _save_probe(path: str, probe: CachedProbe) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, weights=probe.weights,
                        intercept=np.array([probe.intercept], dtype=float),
                        metadata_json=json.dumps(probe.metadata))


def _load_probe(path: str) -> CachedProbe:
    obj = np.load(path, allow_pickle=False)
    return CachedProbe(weights=obj["weights"], intercept=float(obj["intercept"][0]),
                       metadata=json.loads(str(obj["metadata_json"])))


# ── Probe fit / eval ──────────────────────────────────────────────────────────

def fit_linear_probe(X_train, y_train, X_eval, y_eval,
                     C: float = 1.0, max_iter: int = 5000, seed: int = 0) -> ProbeResult:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xev_s = scaler.transform(X_eval)
    clf = LogisticRegression(penalty="l2", C=C, solver="liblinear",
                             max_iter=max_iter, random_state=seed)
    clf.fit(Xtr_s, y_train)
    score = clf.decision_function(Xev_s)
    pred  = (score >= 0.0).astype(int)
    acc   = float(accuracy_score(y_eval, pred))
    auc, _, _ = safe_auc(y_eval, score)
    coef  = clf.coef_.reshape(-1)
    w_orig = coef / scaler.scale_
    b_orig = float(clf.intercept_[0] - np.sum((coef * scaler.mean_) / scaler.scale_))
    return ProbeResult(weights=w_orig, intercept=b_orig,
                       y_true=y_eval, score=score, pred=pred, acc=acc, auc=auc)


def eval_from_weights(X_eval, y_eval, weights, intercept) -> ProbeResult:
    score = X_eval @ weights + intercept
    pred  = (score >= 0.0).astype(int)
    acc   = float(accuracy_score(y_eval, pred))
    auc, _, _ = safe_auc(y_eval, score)
    return ProbeResult(weights=weights, intercept=intercept,
                       y_true=y_eval, score=score, pred=pred, acc=acc, auc=auc)


def get_or_fit_probe(
    probe_owner, train_scheme, train_window_id,
    band_name, band_layers, band_mode,
    X_train, y_train, X_eval, y_eval,
    seed, c_value, max_iter,
    use_probe_cache, probe_cache_dir,
) -> Tuple[ProbeResult, str]:
    key   = _probe_cache_key(probe_owner, train_scheme, train_window_id,
                              band_name, band_layers, band_mode,
                              seed, c_value, max_iter, X_train, y_train)
    path  = _cache_path(probe_cache_dir, key)

    if use_probe_cache and os.path.exists(path):
        cached = _load_probe(path)
        return eval_from_weights(X_eval, y_eval, cached.weights, cached.intercept), "hit"

    result = fit_linear_probe(X_train, y_train, X_eval, y_eval,
                               C=c_value, max_iter=max_iter, seed=seed)
    if use_probe_cache:
        _save_probe(path, CachedProbe(weights=result.weights, intercept=result.intercept,
                                      metadata={"probe_owner": probe_owner, "band": band_name,
                                                "C": c_value}))
        return result, "miss_trained_saved"
    return result, "miss_trained_nocache"


# ── Band metrics ──────────────────────────────────────────────────────────────

def compute_band_metrics(
    base_hs, method_hs, labels,
    method_name, train_scheme, train_window_id,
    band_name, band_mode, band_layers,
    eval_split, eval_indices,
    seed, c_value, max_iter,
    use_probe_cache, probe_cache_dir,
) -> Tuple[BandMetrics, str, str]:
    # --- train features
    Xb_tr_full = extract_band_features(base_hs.train, band_layers, band_mode)
    Xm_tr_full = extract_band_features(method_hs.train, band_layers, band_mode)
    y_tr_full  = labels.train

    train_idx  = get_train_window_indices(train_scheme, train_window_id, len(y_tr_full))
    if train_idx is None:
        Xb_tr, Xm_tr, y_tr = Xb_tr_full, Xm_tr_full, y_tr_full
    else:
        Xb_tr, Xm_tr, y_tr = Xb_tr_full[train_idx], Xm_tr_full[train_idx], y_tr_full[train_idx]

    # --- eval features
    Xb_ev_full = extract_band_features(get_split_array(base_hs, eval_split), band_layers, band_mode)
    Xm_ev_full = extract_band_features(get_split_array(method_hs, eval_split), band_layers, band_mode)
    y_ev_full  = get_split_array(labels, eval_split)

    if eval_indices is not None:
        Xb_ev, Xm_ev, y_ev = Xb_ev_full[eval_indices], Xm_ev_full[eval_indices], y_ev_full[eval_indices]
    else:
        Xb_ev, Xm_ev, y_ev = Xb_ev_full, Xm_ev_full, y_ev_full

    base_probe, base_cache = get_or_fit_probe(
        "base", train_scheme, train_window_id, band_name, band_layers, band_mode,
        Xb_tr, y_tr, Xb_ev, y_ev, seed, c_value, max_iter, use_probe_cache, probe_cache_dir)

    method_probe, method_cache = get_or_fit_probe(
        method_name, train_scheme, train_window_id, band_name, band_layers, band_mode,
        Xm_tr, y_tr, Xm_ev, y_ev, seed, c_value, max_iter, use_probe_cache, probe_cache_dir)

    frozen_probe = eval_from_weights(Xm_ev, y_ev, base_probe.weights, base_probe.intercept)

    dr_weight = cosine_distance(base_probe.weights, method_probe.weights)
    dr_score  = cosine_distance(base_probe.score,   method_probe.score)
    re_acc    = normalized_erasure(base_probe.acc, method_probe.acc)
    re_auc    = normalized_erasure(base_probe.auc, method_probe.auc)
    _, re_valid, re_nan_reason = safe_auc(y_ev, method_probe.score)

    def _recovery(pred):
        count = int((pred == y_ev).sum())
        rate  = float(count / len(y_ev)) if len(y_ev) > 0 else float("nan")
        return count, rate

    ret_count, ret_rate  = _recovery(method_probe.pred)
    frz_count, frz_rate  = _recovery(frozen_probe.pred)
    gain_count = ret_count - frz_count
    gain_rate  = ret_rate  - frz_rate

    return BandMetrics(
        band_name=band_name, layers=band_layers,
        n_eval=len(y_ev), n_pos=int((y_ev == 1).sum()), n_neg=int((y_ev == 0).sum()),
        re_auc_valid=bool(re_valid), re_auc_nan_reason=re_nan_reason,
        base_probe_on_base_eval=base_probe,
        method_probe_on_method_eval=method_probe,
        frozen_base_probe_on_method_eval=frozen_probe,
        dr_weight_cos=dr_weight, dr_score_cos=dr_score,
        re_acc=re_acc, re_auc=re_auc,
        retrained_recovered_count=ret_count, retrained_recovered_rate=ret_rate,
        frozen_recovered_count=frz_count,   frozen_recovered_rate=frz_rate,
        recovery_gain_count=gain_count,      recovery_gain_rate=gain_rate,
    ), base_cache, method_cache


# ── Band prediction CSV ───────────────────────────────────────────────────────

def save_band_predictions_csv(
    out_dir, method, band, eval_split, subset_mode, train_scheme, train_window_id,
    y_true, base_probe, method_probe, frozen_probe,
):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir,
        f"{method}_{band}_{subset_mode}_{eval_split}_{train_scheme}_{train_window_id}_predictions.csv"
    )
    pd.DataFrame({
        "eval_row_order":                  np.arange(len(y_true)),
        "gold_label":                      y_true,
        "base_probe_on_base_score":        base_probe.score,
        "base_probe_on_base_pred":         base_probe.pred,
        "method_probe_on_method_score":    method_probe.score,
        "method_probe_on_method_pred":     method_probe.pred,
        "frozen_base_probe_on_method_score": frozen_probe.score,
        "frozen_base_probe_on_method_pred":  frozen_probe.pred,
    }).to_csv(path, index=False)


# ── Scatter plot ──────────────────────────────────────────────────────────────

def _build_axis(metrics_by_band: Dict[str, BandMetrics], expr: str, re_metric: str, dr_src: str) -> float:
    def RE(b): return metrics_by_band[b].re_auc if re_metric == "auc" else metrics_by_band[b].re_acc
    def DR(b): return metrics_by_band[b].dr_weight_cos if dr_src == "weight" else metrics_by_band[b].dr_score_cos
    atoms = {
        "RE_early": RE("early"), "RE_mid": RE("mid"), "RE_late": RE("late"),
        "DR_early": DR("early"), "DR_mid": DR("mid"), "DR_late": DR("late"),
    }
    if expr in atoms:
        return atoms[expr]
    eps = 1e-8
    composed = {
        "DR_late_minus_mid":   DR("late") - DR("mid"),
        "DR_late_minus_early": DR("late") - DR("early"),
        "RE_mid_minus_early":  RE("mid")  - RE("early"),
        "RE_late_minus_mid":   RE("late") - RE("mid"),
        "DR_late_over_mid":    DR("late") / max(DR("mid"),   eps),
        "DR_late_over_early":  DR("late") / max(DR("early"), eps),
        "RE_mid_over_early":   RE("mid")  / max(RE("early"), eps),
    }
    if expr in composed:
        return composed[expr]
    raise ValueError(f"Unsupported axis expression: {expr}")


def make_scatter(df: pd.DataFrame, x_col: str, y_col: str, out_path: str, title: str):
    plt.figure(figsize=(8, 6))
    for _, row in df.iterrows():
        plt.scatter(row[x_col], row[y_col], s=110)
        plt.text(row[x_col], row[y_col], f"  {row['method']}", fontsize=9, va="center")
    plt.axhline(0, linewidth=1, linestyle="--", color="gray")
    plt.axvline(0, linewidth=1, linestyle="--", color="gray")
    plt.xlabel(x_col, fontsize=11)
    plt.ylabel(y_col, fontsize=11)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RE/DR dual-path band analysis for one unlearning method."
    )
    # single-method mode (used by SLURM array)
    p.add_argument("--method",  type=str, default=None,
                   help="One method name (e.g. GradDiff, PB_J).  If omitted, pass --methods.")
    # multi-method mode (for manual runs)
    p.add_argument("--methods", type=str, default=None,
                   help="Comma-separated method list.  Overrides --method.")

    p.add_argument("--data_dir",      type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--labels_csv",    type=str, default=DEFAULT_LABELS_CSV)
    p.add_argument("--base_prefix",   type=str, default=DEFAULT_BASE_PREFIX,
                   help="Prefix for base hidden-state files (default: base_hs)")
    p.add_argument("--method_suffix", type=str, default=DEFAULT_METHOD_SUFFIX,
                   help="Suffix for method hidden-state files (default: _hs)")
    p.add_argument("--out_root",      type=str, default=DEFAULT_OUT_ROOT,
                   help="Root output dir; per-method sub-dirs are created automatically")
    p.add_argument("--band_mode",     type=str, default="concat",
                   choices=["concat", "mean", "max"])
    p.add_argument("--re_metric",     type=str, default="auc", choices=["auc", "acc"])
    p.add_argument("--x_expr",        type=str, default="DR_late_minus_mid")
    p.add_argument("--y_expr",        type=str, default="RE_mid_minus_early")
    p.add_argument("--eval_split",    type=str, default="test")
    p.add_argument("--train_scheme",  type=str, default="full",
                   choices=["full", "traincv"])
    p.add_argument("--train_window_id", type=str, default="full")
    p.add_argument("--early_layers",  type=str, default="")
    p.add_argument("--mid_layers",    type=str, default="")
    p.add_argument("--late_layers",   type=str, default="")
    p.add_argument("--subset_mode", type=str, default="none",
                   choices=["none", "suppressed", "emerged"],
                   help="Restrict eval to a subset of questions (none = all)")
    p.add_argument("--subset_dir", type=str, default="",
                   help="Dir containing <method>_<subset_mode>_<split>.csv files (defaults to data_dir)")
    p.add_argument("--save_band_prediction_csvs", action="store_true",
                   help="Write per-band prediction CSVs (required by build_authoritative_measurements.py)")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--probe_C",       type=float, default=1.0)
    p.add_argument("--probe_max_iter",type=int, default=5000)
    p.add_argument("--use_probe_cache", action="store_true")
    p.add_argument("--probe_cache_dir", type=str, default="")
    p.add_argument("--save_json",     action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve method list
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    elif args.method:
        methods = [args.method.strip()]
    else:
        raise ValueError("Provide --method METHOD or --methods M1,M2,...")

    # Output dir: one sub-dir per method under out_root
    tag = f"{args.train_scheme}_{args.train_window_id}"
    method_out_dir = os.path.join(args.out_root, methods[0] if len(methods) == 1 else "multi", tag)
    os.makedirs(method_out_dir, exist_ok=True)
    prediction_dir = os.path.join(method_out_dir, "band_predictions")
    probe_cache_dir = (args.probe_cache_dir if args.probe_cache_dir
                       else os.path.join(method_out_dir, "probe_cache"))
    if args.use_probe_cache:
        os.makedirs(probe_cache_dir, exist_ok=True)
    subset_dir = args.subset_dir if args.subset_dir else args.data_dir

    # Load base hidden states and labels
    print(f"Loading base hidden states from {args.data_dir}/{args.base_prefix}_{{train,val,test}}.npy …")
    base_hs = load_split_data(args.data_dir, args.base_prefix)
    labels  = load_labels_from_csv(args.labels_csv)

    num_layers = base_hs.train.shape[1] if base_hs.train.ndim == 3 else 0
    band_specs = {
        "early": parse_layer_spec(args.early_layers, num_layers, "early"),
        "mid":   parse_layer_spec(args.mid_layers,   num_layers, "mid"),
        "late":  parse_layer_spec(args.late_layers,  num_layers, "late"),
    }
    print(f"Layers — early: {band_specs['early'][:3]}..  mid: {band_specs['mid'][:3]}..  "
          f"late: {band_specs['late'][:3]}..")

    split_n   = len(get_split_array(labels, args.eval_split))
    eval_idx  = load_eval_indices_from_labels(args.labels_csv, args.eval_split, split_n)

    rows, recovery_rows, full_metrics = [], [], {}

    for method in methods:
        print(f"\n── Method: {method} ──")
        method_hs = load_split_data(args.data_dir, f"{method}{args.method_suffix}")

        # Resolve per-method eval subset (intersection with any testw window indices)
        final_eval_indices = eval_idx.copy() if eval_idx is not None else None
        if args.subset_mode != "none":
            candidates = []
            if args.eval_split.startswith("testw"):
                candidates.append(os.path.join(subset_dir, f"{method}_{args.subset_mode}_test.csv"))
            else:
                candidates.append(os.path.join(subset_dir, f"{method}_{args.subset_mode}_{args.eval_split}.csv"))
                if args.eval_split.startswith("test") and args.eval_split != "test":
                    candidates.append(os.path.join(subset_dir, f"{method}_{args.subset_mode}_test.csv"))
            subset_path = next((p for p in candidates if os.path.exists(p)), None)
            if subset_path is None:
                raise FileNotFoundError(
                    f"No subset CSV for method={method}, subset_mode={args.subset_mode}, "
                    f"eval_split={args.eval_split}. Tried: {candidates}"
                )
            subset_idx = load_subset_indices(subset_path, expected_n=split_n)
            final_eval_indices = (subset_idx if final_eval_indices is None
                                  else np.intersect1d(final_eval_indices, subset_idx))

        metrics_by_band = {}
        for band in ["early", "mid", "late"]:
            bm, _, _ = compute_band_metrics(
                base_hs=base_hs, method_hs=method_hs, labels=labels,
                method_name=method,
                train_scheme=args.train_scheme, train_window_id=args.train_window_id,
                band_name=band, band_mode=args.band_mode, band_layers=band_specs[band],
                eval_split=args.eval_split, eval_indices=final_eval_indices,
                seed=args.seed, c_value=args.probe_C, max_iter=args.probe_max_iter,
                use_probe_cache=args.use_probe_cache, probe_cache_dir=probe_cache_dir,
            )
            metrics_by_band[band] = bm
            print(f"  {band:5s}  RE_auc={bm.re_auc:.3f}  DR_w={bm.dr_weight_cos:.3f}  "
                  f"frozen_rec={bm.frozen_recovered_rate:.3f}  "
                  f"retrain_rec={bm.retrained_recovered_rate:.3f}")

            recovery_rows.append({
                "method": method, "band": band,
                "subset_mode": args.subset_mode,
                "train_scheme": args.train_scheme, "train_window_id": args.train_window_id,
                "eval_split": args.eval_split,
                "n_eval": bm.n_eval,
                "n_pos": bm.n_pos, "n_neg": bm.n_neg,
                "re_auc_valid": int(bm.re_auc_valid),
                "re_auc_nan_reason": bm.re_auc_nan_reason,
                "re_acc": bm.re_acc, "re_auc": bm.re_auc,
                "dr_weight_cos": bm.dr_weight_cos, "dr_score_cos": bm.dr_score_cos,
                "base_eval_auc": bm.base_probe_on_base_eval.auc,
                "method_eval_auc": bm.method_probe_on_method_eval.auc,
                "frozen_on_method_auc": bm.frozen_base_probe_on_method_eval.auc,
                "base_eval_acc": bm.base_probe_on_base_eval.acc,
                "method_eval_acc": bm.method_probe_on_method_eval.acc,
                "frozen_on_method_acc": bm.frozen_base_probe_on_method_eval.acc,
                "retrained_recovered_count": bm.retrained_recovered_count,
                "retrained_recovered_rate":  bm.retrained_recovered_rate,
                "frozen_recovered_count":    bm.frozen_recovered_count,
                "frozen_recovered_rate":     bm.frozen_recovered_rate,
                "recovery_gain_count":       bm.recovery_gain_count,
                "recovery_gain_rate":        bm.recovery_gain_rate,
            })

            if args.save_band_prediction_csvs:
                save_band_predictions_csv(
                    prediction_dir, method, band,
                    args.eval_split, args.subset_mode,
                    args.train_scheme, args.train_window_id,
                    bm.base_probe_on_base_eval.y_true,
                    bm.base_probe_on_base_eval,
                    bm.method_probe_on_method_eval,
                    bm.frozen_base_probe_on_method_eval,
                )

        x_w = _build_axis(metrics_by_band, args.x_expr, args.re_metric, "weight")
        x_s = _build_axis(metrics_by_band, args.x_expr, args.re_metric, "score")
        y   = _build_axis(metrics_by_band, args.y_expr, args.re_metric, "weight")

        rows.append({
            "method": method,
            "train_scheme": args.train_scheme, "train_window_id": args.train_window_id,
            "x_weight": x_w, "x_score": x_s, "y_re": y,
            **{f"RE_{args.re_metric}_{b}": getattr(metrics_by_band[b], f"re_{args.re_metric}")
               for b in ["early", "mid", "late"]},
            **{f"DR_weight_{b}": metrics_by_band[b].dr_weight_cos for b in ["early", "mid", "late"]},
            **{f"frozen_rate_{b}": metrics_by_band[b].frozen_recovered_rate for b in ["early", "mid", "late"]},
            **{f"retrain_rate_{b}": metrics_by_band[b].retrained_recovered_rate for b in ["early", "mid", "late"]},
        })

        if args.save_json:
            full_metrics[method] = {
                b: {
                    "re_auc": metrics_by_band[b].re_auc,
                    "re_acc": metrics_by_band[b].re_acc,
                    "dr_weight_cos": metrics_by_band[b].dr_weight_cos,
                    "frozen_recovered_rate": metrics_by_band[b].frozen_recovered_rate,
                    "retrained_recovered_rate": metrics_by_band[b].retrained_recovered_rate,
                    "recovery_gain_rate": metrics_by_band[b].recovery_gain_rate,
                    "base_eval_auc": metrics_by_band[b].base_probe_on_base_eval.auc,
                    "frozen_on_method_auc": metrics_by_band[b].frozen_base_probe_on_method_eval.auc,
                    "layers": metrics_by_band[b].layers,
                }
                for b in ["early", "mid", "late"]
            }

    # ── Save outputs ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(method_out_dir, "re_dr_summary.csv"), index=False)

    rec_df = pd.DataFrame(recovery_rows).sort_values(["method", "band"]).reset_index(drop=True)
    rec_df.to_csv(os.path.join(method_out_dir, "recovery_by_band.csv"), index=False)

    if args.save_json and full_metrics:
        with open(os.path.join(method_out_dir, "full_metrics.json"), "w") as f:
            json.dump(full_metrics, f, indent=2)

    # ── Scatter plots (only meaningful when >1 method was run) ──────────────
    if len(rows) > 1:
        plot_tag = f"{args.train_scheme}_{args.train_window_id}_{args.subset_mode}_{args.eval_split}"
        make_scatter(
            df, x_col="x_weight", y_col="y_re",
            out_path=os.path.join(method_out_dir, f"scatter_DR_vs_RE_{plot_tag}.png"),
            title=f"DR ({args.x_expr}) vs RE ({args.y_expr}) — weight space",
        )
        make_scatter(
            df, x_col="x_score", y_col="y_re",
            out_path=os.path.join(method_out_dir, f"scatter_DR_score_vs_RE_{plot_tag}.png"),
            title=f"DR score ({args.x_expr}) vs RE ({args.y_expr})",
        )
        print(f"\nScatter plots saved to {method_out_dir}")

    print(f"\nDone.  Outputs in {method_out_dir}")
    if args.save_band_prediction_csvs:
        print(f"  Band predictions: {prediction_dir}/")
    print(f"  Recovery CSV:     {method_out_dir}/recovery_by_band.csv")
    print(f"  Summary CSV:      {method_out_dir}/re_dr_summary.csv")
    if args.subset_mode != "none":
        print(f"  Subset mode:      {args.subset_mode}")


if __name__ == "__main__":
    main()
