#!/usr/bin/env python3
"""
quadrant_counts.py

For each model × probe_type × classifier, counts how many bio (WMDP) QA
pairs fall into each of the 4 knowledge quadrants:

  Remains:     base probe correct  AND  method probe correct
  Unlearned:   base probe correct  AND  method probe incorrect
  Emerged:     base probe incorrect AND  method probe correct
  Never Known: base probe incorrect AND  method probe incorrect

Two "method knows" variants:
  frozen_base  — base probe weights frozen, applied to method hidden states
  method_probe — method's own probe applied to method hidden states

Modes
-----
Default (no --kfold):
  Uses pre-trained probes from checkpoints/{model}_probes.pkl applied to
  the test-split hidden states only.  Fast (seconds).

--kfold:
  Replicates the kfold_probe.py fold scheme: concatenates train+val+test
  hidden states, splits questions into 5 folds, trains probes from scratch
  on each fold's train slice, and evaluates on the fold's test slice.
  Outputs one row per fold (fold=0..4) plus aggregate rows (mean, std, ci95).
  No LLM needed — pure sklearn on pre-saved .npy files (~10-20 min on CPU).

Probe types (all × LR, RF, AdaBoost):
  per_layer, mid_band, full_layer, init_band, init_band_emb, end_band, ib_no_pca

Usage:
  python quadrant_counts.py
  python quadrant_counts.py --kfold
  python quadrant_counts.py --kfold --out quadrant_counts_kfold.csv
  python quadrant_counts.py --no_sweep --checkpoint_dir /path/to/checkpoints
"""

import argparse
import csv
import json
import pickle
import random
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Constants (must match hidden_knowledge_after_unlearning.py / kfold_probe.py)

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]
CLF_NAMES = ["LR", "RF", "AdaBoost"]
N_SWEEP_CHECKPOINTS = 8
N_FOLDS = 5
RANDOM_SEED = 42
PCA_DIMS = 256

MULTI_LAYER_START = 10
MULTI_LAYER_END   = 22
INIT_BAND_START   = 1
INIT_BAND_EMB_START = 0
INIT_BAND_END     = 9
END_BAND_START    = 23
IB_NO_PCA_START   = 1
IB_NO_PCA_END     = 6

BAND_KEYS = [
    "per_layer",
    "mid_band",
    "full_layer",
    "init_band",
    "init_band_emb",
    "end_band",
    "ib_no_pca",
]

DEFAULT_BAND_RANGES = {
    "mid_band":       (MULTI_LAYER_START, MULTI_LAYER_END),
    "full_layer":     (0, 32),
    "init_band":      (INIT_BAND_START, INIT_BAND_END),
    "init_band_emb":  (INIT_BAND_EMB_START, INIT_BAND_END),
    "end_band":       (END_BAND_START, 32),
    "ib_no_pca":      (IB_NO_PCA_START, IB_NO_PCA_END),
}

# ── Name helpers ──────────────────────────────────────────────────────────────

def safe_name(name: str) -> str:
    """Filesystem-safe name — matches hidden_knowledge_after_unlearning.py exactly."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_npy(path: Path):
    return np.load(str(path)) if path.exists() else None


def _load_pkl(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ── Probe pipelines (identical to kfold_probe.py) ─────────────────────────────

def _per_layer_pipeline(clf_name: str) -> Pipeline:
    if clf_name == "LR":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                                    random_state=RANDOM_SEED))])
    if clf_name == "RF":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                        random_state=RANDOM_SEED))])
    return Pipeline([("sc", StandardScaler()),
                     ("clf", AdaBoostClassifier(n_estimators=100,
                                                random_state=RANDOM_SEED))])


def _multi_layer_pipeline(clf_name: str) -> Pipeline:
    clf = (LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
           if clf_name == "LR" else
           RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED)
           if clf_name == "RF" else
           AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED))
    return Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=PCA_DIMS, random_state=RANDOM_SEED)),
                     ("clf", clf)])


def _no_pca_pipeline(clf_name: str) -> Pipeline:
    clf = (LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
           if clf_name == "LR" else
           RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED)
           if clf_name == "RF" else
           AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED))
    return Pipeline([("sc", StandardScaler()), ("clf", clf)])


# ── Probe training (used only in kfold mode) ──────────────────────────────────

def train_probe_set(hs_train: np.ndarray, y_train: np.ndarray,
                    hs_val: np.ndarray, y_val: np.ndarray,
                    label: str = "") -> dict:
    """Train all bands × classifiers. Returns a probe_set dict."""
    n_train, n_layers, _ = hs_train.shape
    prefix = f"[{label}] " if label else ""

    per_layer = {c: {} for c in CLF_NAMES}
    for l in range(n_layers):
        X_tr = hs_train[:, l, :]
        for clf_name in CLF_NAMES:
            pipe = _per_layer_pipeline(clf_name)
            pipe.fit(X_tr, y_train)
            per_layer[clf_name][l] = pipe
        print(f"  {prefix}per-layer {l+1}/{n_layers}", end="\r", flush=True)
    print(f"  {prefix}per-layer done ({n_layers} layers)", flush=True)

    best_layers = {}
    for clf_name in CLF_NAMES:
        val_accs = {l: per_layer[clf_name][l].score(hs_val[:, l, :], y_val)
                    for l in range(n_layers)}
        best_layers[clf_name] = max(val_accs, key=val_accs.get)

    def _band(s, e, pipeline_fn, name):
        s, e = max(0, s), min(n_layers - 1, e)
        X_tr  = hs_train[:, s:e + 1, :].reshape(n_train, -1)
        X_val = hs_val[:,   s:e + 1, :].reshape(len(hs_val), -1)
        probes = {}
        for clf_name in CLF_NAMES:
            pipe = pipeline_fn(clf_name)
            pipe.fit(X_tr, y_train)
            probes[clf_name] = pipe
        print(f"  {prefix}{name} done", flush=True)
        return probes, (s, e)

    mid_band,      mb_r  = _band(MULTI_LAYER_START, MULTI_LAYER_END,  _multi_layer_pipeline, "mid_band")
    full_layer,    fl_r  = _band(0, n_layers - 1,                     _multi_layer_pipeline, "full_layer")
    init_band,     ib_r  = _band(INIT_BAND_START,   INIT_BAND_END,    _multi_layer_pipeline, "init_band")
    init_band_emb, ibe_r = _band(INIT_BAND_EMB_START, INIT_BAND_END,  _multi_layer_pipeline, "init_band_emb")
    end_band,      eb_r  = _band(END_BAND_START,    n_layers - 1,     _multi_layer_pipeline, "end_band")
    ib_no_pca,     ibnp_r= _band(IB_NO_PCA_START,  IB_NO_PCA_END,    _no_pca_pipeline,       "ib_no_pca")

    return {
        "per_layer": per_layer, "best_layers": best_layers,
        "mid_band": mid_band, "mid_band_range": mb_r,
        "full_layer": full_layer, "full_layer_range": fl_r,
        "init_band": init_band, "init_band_range": ib_r,
        "init_band_emb": init_band_emb, "init_band_emb_range": ibe_r,
        "end_band": end_band, "end_band_range": eb_r,
        "ib_no_pca": ib_no_pca, "ib_no_pca_range": ibnp_r,
    }


# ── Prediction helpers ────────────────────────────────────────────────────────

def _band_range(probe_set: dict, band: str, n_layers: int):
    key = f"{band}_range"
    if key in probe_set:
        return tuple(probe_set[key])
    if band == "mid_band" and "multi_layer_range" in probe_set:
        return tuple(probe_set["multi_layer_range"])
    lo, hi = DEFAULT_BAND_RANGES.get(band, (0, n_layers - 1))
    return lo, min(hi, n_layers - 1)


def get_predictions(probe_set: dict, hs: np.ndarray,
                    band: str, clf_name: str):
    """Per-sample integer predictions (0/1). Returns None if unavailable."""
    n, n_layers, _ = hs.shape

    if band == "per_layer":
        l = probe_set.get("best_layers", {}).get(clf_name)
        if l is None:
            return None
        pipe = probe_set.get("per_layer", {}).get(clf_name, {}).get(l)
        if pipe is None:
            return None
        return pipe.predict(hs[:, l, :]).astype(int)

    band_dict = probe_set.get(band)
    if not band_dict or clf_name not in band_dict:
        return None

    lo, hi = _band_range(probe_set, band, n_layers)
    if lo > hi:
        return None
    X = hs[:, lo:hi + 1, :].reshape(n, -1)
    try:
        return band_dict[clf_name].predict(X).astype(int)
    except Exception as e:
        print(f"    [warn] predict failed ({band}/{clf_name}): {e}", flush=True)
        return None


# ── Quadrant counting ─────────────────────────────────────────────────────────

def count_quadrants(base_correct: np.ndarray, method_correct: np.ndarray) -> dict:
    bc = base_correct.astype(bool)
    mc = method_correct.astype(bool)
    return {
        "remains":     int(( bc &  mc).sum()),
        "unlearned":   int(( bc & ~mc).sum()),
        "emerged":     int((~bc &  mc).sum()),
        "never_known": int((~bc & ~mc).sum()),
        "total":       len(bc),
    }


def process_pair(model_name, checkpoint, fold,
                 base_probe_set, base_hs,
                 method_probe_set, method_hs,
                 y_test) -> list[dict]:
    """One CSV row per (band × clf × variant) for this model/checkpoint/fold."""
    rows = []
    if len(method_hs) != len(y_test):
        print(f"  [warn] {model_name}/{checkpoint}: hs len {len(method_hs)} "
              f"!= y_test len {len(y_test)} — skipping.", flush=True)
        return rows

    for band in BAND_KEYS:
        for clf_name in CLF_NAMES:
            base_preds = get_predictions(base_probe_set, base_hs, band, clf_name)
            if base_preds is None:
                continue
            base_correct = base_preds == y_test

            # frozen_base: base probe weights on method HS
            fp = get_predictions(base_probe_set, method_hs, band, clf_name)
            if fp is not None:
                rows.append({"model": model_name, "checkpoint": checkpoint,
                             "fold": fold, "probe_type": band,
                             "classifier": clf_name, "probe_variant": "frozen_base",
                             **count_quadrants(base_correct, fp == y_test)})

            # method_probe: method's own probe on method HS
            if method_probe_set is not None:
                mp = get_predictions(method_probe_set, method_hs, band, clf_name)
                if mp is not None:
                    rows.append({"model": model_name, "checkpoint": checkpoint,
                                 "fold": fold, "probe_type": band,
                                 "classifier": clf_name, "probe_variant": "method_probe",
                                 **count_quadrants(base_correct, mp == y_test)})
    return rows


# ── Kfold: fold assignment (mirrors kfold_probe.py exactly) ──────────────────

def build_fold_assignment(csv_path: Path, n_folds=N_FOLDS, seed=RANDOM_SEED):
    n_train = n_val = n_test = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":   n_train += 1
            elif row["split"] == "val":   n_val   += 1
            else:                         n_test  += 1

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

    qids = sorted(by_qid.keys())
    rng  = random.Random(seed)
    rng.shuffle(qids)
    fold_size = len(qids) // n_folds
    qids = qids[:fold_size * n_folds]
    folds = [qids[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    print(f"  [kfold] {len(qids)} questions, {n_folds} folds × {fold_size}",
          flush=True)
    return by_qid, folds


def get_fold_indices(fold_idx, by_qid, folds):
    """Returns (train_idx, train_y, val_idx, val_y, test_idx, test_y)."""
    test_qids  = folds[fold_idx]
    val_qids   = folds[(fold_idx + 1) % N_FOLDS]
    train_qids = [qid for i, f in enumerate(folds)
                  for qid in f
                  if i != fold_idx and i != (fold_idx + 1) % N_FOLDS]

    def collect(qids):
        idxs, labels = [], []
        for qid in qids:
            for (idx, lbl) in by_qid[qid]:
                idxs.append(idx)
                labels.append(lbl)
        return np.array(idxs, np.int64), np.array(labels, np.int32)

    return (*collect(train_qids), *collect(val_qids), *collect(test_qids))


def _ci95(arr):
    n = len(arr)
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(0.975, df=n - 1) * np.std(arr, ddof=1) / n ** 0.5)
    except ImportError:
        return float(np.std(arr, ddof=1) / n ** 0.5 * 2.0)


def aggregate_fold_rows(fold_rows: list[dict]) -> list[dict]:
    """Given rows with fold=0..4, return mean + std + ci95 aggregate rows."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in fold_rows:
        key = (r["model"], r["checkpoint"], r["probe_type"],
               r["classifier"], r["probe_variant"])
        groups[key].append(r)

    agg_rows = []
    for key, rows in groups.items():
        model, checkpoint, probe_type, clf, variant = key
        base = {"model": model, "checkpoint": checkpoint,
                "probe_type": probe_type, "classifier": clf,
                "probe_variant": variant}
        for stat, fn in [("mean", np.mean), ("std", np.std),
                         ("ci95", _ci95)]:
            row = {**base, "fold": stat}
            for q in ("remains", "unlearned", "emerged", "never_known", "total"):
                vals = [r[q] for r in rows]
                row[q] = round(float(fn(vals)), 3) if stat != "ci95" else round(_ci95(vals), 3)
            agg_rows.append(row)
    return agg_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--out", default=None,
                    help="Output CSV (default: quadrant_counts.csv or "
                         "quadrant_counts_kfold.csv)")
    ap.add_argument("--kfold", action="store_true",
                    help="5-fold CV mode: re-trains probes per fold, "
                         "averages quadrant counts across folds")
    ap.add_argument("--no_sweep", action="store_true",
                    help="Skip per-training-step sweep checkpoints")
    args = ap.parse_args()

    ck_dir   = Path(args.checkpoint_dir)
    csv_path = Path("data") / "wmdp_tf_pairs.csv"
    out_path = Path(args.out) if args.out else Path(
        "quadrant_counts_kfold.csv" if args.kfold else "quadrant_counts.csv"
    )

    # ── Validate inputs ───────────────────────────────────────────────────────
    for req in ("base_probes.pkl", "base_hs_test.npy", "base_results.json"):
        if not (ck_dir / req).exists():
            raise FileNotFoundError(
                f"{ck_dir / req} not found. "
                "Run --stage base of hidden_knowledge_after_unlearning.py first."
            )

    all_rows: list[dict] = []

    # =========================================================================
    # MODE A: default — use pre-trained probes + test-split HS only
    # =========================================================================
    if not args.kfold:
        print("Mode: single-split (pre-trained probes, test set only)", flush=True)

        base_probe_set = _load_pkl(ck_dir / "base_probes.pkl")
        base_hs_test   = _load_npy(ck_dir / "base_hs_test.npy")

        # y_test from CSV
        y_test_list = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == "test":
                    y_test_list.append(1 if row["label"] == "True" else 0)
        y_test = np.array(y_test_list, np.int32)
        print(f"  base_hs_test: {base_hs_test.shape}   y_test: {len(y_test)}", flush=True)

        # Final checkpoints
        print("\n── Final checkpoints ─────────────────────────────────────────", flush=True)
        for method_name in METHODS:
            sn        = safe_name(method_name)
            hs_path   = ck_dir / f"{sn}_hs_test.npy"
            pkl_path  = ck_dir / f"{sn}_probes.pkl"

            if not hs_path.exists():
                print(f"  {method_name}: {hs_path.name} not found — skipping.", flush=True)
                continue

            print(f"  {method_name} (ck8) ...", end="  ", flush=True)
            method_hs    = _load_npy(hs_path)
            method_probe = _load_pkl(pkl_path)
            if method_probe is not None and "per_layer" not in method_probe:
                method_probe = None

            rows = process_pair(method_name, "ck8", "single",
                                base_probe_set, base_hs_test,
                                method_probe, method_hs, y_test)
            all_rows.extend(rows)
            print(f"{len(rows)} rows", flush=True)

        # Sweep checkpoints
        if not args.no_sweep:
            print("\n── Sweep checkpoints ─────────────────────────────────────────", flush=True)
            for method_name in METHODS:
                sn = safe_name(method_name)
                for ck_num in range(1, N_SWEEP_CHECKPOINTS + 1):
                    ck_sub = ck_dir / f"{sn}_ck{ck_num}"
                    if not ck_sub.exists():
                        continue
                    hs_path  = ck_sub / "hs_test.npy"
                    pkl_path = ck_sub / "probes.pkl"
                    if not hs_path.exists():
                        continue
                    print(f"  {method_name} ck{ck_num} ...", end="  ", flush=True)
                    method_hs    = _load_npy(hs_path)
                    method_probe = _load_pkl(pkl_path)
                    if method_probe is not None and "per_layer" not in method_probe:
                        method_probe = None
                    rows = process_pair(method_name, f"ck{ck_num}", "single",
                                        base_probe_set, base_hs_test,
                                        method_probe, method_hs, y_test)
                    all_rows.extend(rows)
                    print(f"{len(rows)} rows", flush=True)

    # =========================================================================
    # MODE B: kfold — retrain probes per fold, average quadrant counts
    # =========================================================================
    else:
        print("Mode: 5-fold CV (retraining probes from scratch per fold)", flush=True)
        print("Note: ~10-20 min on CPU for all methods", flush=True)

        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found.")

        by_qid, folds = build_fold_assignment(csv_path)

        # Load full concatenated hidden states for all models up front
        def load_all_hs(model_name: str):
            if model_name == "base":
                paths = [ck_dir / f"base_hs_{s}.npy" for s in ("train", "val", "test")]
            else:
                sn    = safe_name(model_name)
                paths = [ck_dir / f"{sn}_hs_{s}.npy" for s in ("train", "val", "test")]
            missing = [p for p in paths if not p.exists()]
            if missing:
                print(f"  [warn] Missing for {model_name}: {[p.name for p in missing]}",
                      flush=True)
                return None
            parts = [np.load(str(p)) for p in paths]
            hs = np.concatenate(parts, axis=0)
            print(f"  {model_name}: loaded {hs.shape}", flush=True)
            return hs

        print("\nLoading all hidden states ...", flush=True)
        base_hs_all = load_all_hs("base")
        if base_hs_all is None:
            raise FileNotFoundError("Base hidden states (train/val/test) required for kfold.")

        method_hs_all = {}
        for method_name in METHODS:
            hs = load_all_hs(method_name)
            if hs is not None:
                method_hs_all[method_name] = hs

        fold_rows: list[dict] = []

        for fold_idx in range(N_FOLDS):
            print(f"\n── Fold {fold_idx} ───────────────────────────────────────────────",
                  flush=True)
            tr_idx, tr_y, val_idx, val_y, te_idx, te_y = get_fold_indices(
                fold_idx, by_qid, folds
            )
            print(f"  train={len(tr_idx)}  val={len(val_idx)}  test={len(te_idx)}", flush=True)

            # Train base probe for this fold
            print(f"  Training base probe (fold {fold_idx}) ...", flush=True)
            base_probe = train_probe_set(
                base_hs_all[tr_idx], tr_y,
                base_hs_all[val_idx], val_y,
                label=f"base/f{fold_idx}",
            )
            base_hs_test = base_hs_all[te_idx]

            for method_name in METHODS:
                if method_name not in method_hs_all:
                    print(f"  {method_name}: no HS — skipping.", flush=True)
                    continue

                mhs = method_hs_all[method_name]
                print(f"  Training {method_name} probe (fold {fold_idx}) ...", flush=True)
                method_probe = train_probe_set(
                    mhs[tr_idx], tr_y,
                    mhs[val_idx], val_y,
                    label=f"{method_name}/f{fold_idx}",
                )

                rows = process_pair(
                    method_name, "ck8", fold_idx,
                    base_probe, base_hs_test,
                    method_probe, mhs[te_idx], te_y,
                )
                fold_rows.extend(rows)
                print(f"    → {len(rows)} rows", flush=True)

        # Per-fold rows + aggregate
        all_rows.extend(fold_rows)
        all_rows.extend(aggregate_fold_rows(fold_rows))

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if not all_rows:
        print("\nNo data — check checkpoint files.", flush=True)
        return

    fieldnames = ["model", "checkpoint", "fold", "probe_type", "classifier",
                  "probe_variant", "remains", "unlearned", "emerged",
                  "never_known", "total"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows → {out_path}", flush=True)

    # Quick sanity table
    print("\nSample (frozen_base, full_layer, LR"
          + (", fold=mean" if args.kfold else "") + "):", flush=True)
    hdr = f"  {'model':<12} {'ck':<5} {'fold':<6} {'remains':>8} "
    hdr += f"{'unlearned':>9} {'emerged':>8} {'never_known':>12} {'total':>7}"
    print(hdr, flush=True)
    for row in all_rows:
        if (row["probe_type"] == "full_layer"
                and row["classifier"] == "LR"
                and row["probe_variant"] == "frozen_base"
                and (not args.kfold or row["fold"] == "mean")):
            print(f"  {row['model']:<12} {str(row['checkpoint']):<5} "
                  f"{str(row['fold']):<6} {str(row['remains']):>8} "
                  f"{str(row['unlearned']):>9} {str(row['emerged']):>8} "
                  f"{str(row['never_known']):>12} {str(row['total']):>7}",
                  flush=True)


if __name__ == "__main__":
    main()
