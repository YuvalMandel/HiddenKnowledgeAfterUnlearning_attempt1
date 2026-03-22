#!/usr/bin/env python3
"""
generate_dual_confidence_inputs.py
===================================
Generates a per-method prediction CSV for the dual-confidence figure package.

Probes are fitted fresh from scratch on BASE model training hidden states
(matching the original two_channel_calibrated_late_fusion.py approach):
  - early_rf   : RandomForestClassifier on layers 1–6, mean across layers
  - mid_linear : StandardScaler → LogisticRegression on layers 12–22, concat, no PCA

Then applied to each method's test hidden states.

Reads:
  checkpoints/base_hs_train.npy   → fit probes (always base model train states)
  checkpoints/base_hs_test.npy    (or {sn}_hs_test.npy)  → apply probes
  checkpoints/base_partial.json   (or {sn}_partial.json)  → logit_scores
  data/wmdp_tf_pairs.csv                                   → gold labels & split

Output columns:
  method_name, gold_label, raw_margin, abs_raw_margin,
  early_rf_label_prob, mid_linear_label_score

Usage (one method per job, see slurm_dual_conf_gen.sh):
  python generate_dual_confidence_inputs.py --method ELM --out_dir dual_confidence_inputs
  python generate_dual_confidence_inputs.py --method Base --out_dir dual_confidence_inputs
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR       = Path("data")
WMDP_CSV_PATH  = DATA_DIR / "wmdp_tf_pairs.csv"

# Probe configuration — matches two_channel_calibrated_late_fusion.py defaults
EARLY_LAYERS  = list(range(1, 7))    # layers 1–6, mean
MID_LAYERS    = list(range(12, 23))  # layers 12–22, concat, no PCA

ALL_METHODS = ["Base", "GradDiff", "RMU", "RMU-LAT", "RepNoise",
               "ELM", "RR", "TAR", "PB&J"]


def safe_name(method_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", method_name)


def load_split_labels(csv_path: Path, split: str) -> list[int]:
    """Return gold labels (1=true, 0=false) for the given split, in CSV order."""
    labels = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            labels.append(1 if row["label"].strip().lower() == "true" else 0)
    return labels


def load_logit_scores(sn: str) -> list:
    """Returns list of [true_logit, false_logit] per test example from partial.json."""
    partial_path = CHECKPOINT_DIR / f"{sn}_partial.json"
    if partial_path.exists():
        with open(partial_path) as f:
            partial = json.load(f)
        if "logit_scores" in partial:
            return partial["logit_scores"]
    print(f"  [warn] logit_scores not found in {partial_path}", file=sys.stderr)
    return []


def load_hs(path: Path) -> np.ndarray | None:
    if not path.exists():
        print(f"  [warn] Hidden states not found: {path}", file=sys.stderr)
        return None
    hs = np.load(path)
    print(f"  Loaded {path.name}  shape={hs.shape}")
    return hs


def extract_early(hs: np.ndarray) -> np.ndarray:
    """Mean of layers 1–6 → shape (N, 4096)."""
    return hs[:, EARLY_LAYERS, :].mean(axis=1)


def extract_mid(hs: np.ndarray) -> np.ndarray:
    """Concat of layers 12–22 → shape (N, 11*4096)."""
    return hs[:, MID_LAYERS, :].reshape(len(hs), -1)


def fit_probes(hs_train: np.ndarray, y_train: np.ndarray):
    """
    Fit early RF and mid linear probes on base model training hidden states.
    Returns (early_rf, mid_linear_pipe).
    """
    X_early = extract_early(hs_train)
    X_mid   = extract_mid(hs_train)

    print(f"  Fitting early RF  (layers 1–6 mean,   shape={X_early.shape})...")
    early_rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=1,
    )
    early_rf.fit(X_early, y_train)

    print(f"  Fitting mid LR    (layers 12–22 concat, no PCA, shape={X_mid.shape})...")
    mid_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   LogisticRegression(max_iter=4000, C=1.0, random_state=42)),
    ])
    mid_pipe.fit(X_mid, y_train)

    return early_rf, mid_pipe


def compute_probe_scores(hs_test: np.ndarray, early_rf, mid_pipe):
    """Apply fitted probes to test hidden states."""
    X_early = extract_early(hs_test)
    X_mid   = extract_mid(hs_test)

    classes = list(early_rf.classes_)
    proba   = early_rf.predict_proba(X_early)
    early_rf_proba = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(hs_test))

    mid_linear_score = mid_pipe.decision_function(X_mid)

    return early_rf_proba, mid_linear_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",  required=True,
                    help=f"Method name. One of: {ALL_METHODS}")
    ap.add_argument("--out_dir", default="dual_confidence_inputs")
    args = ap.parse_args()

    method = args.method
    if method not in ALL_METHODS:
        ap.error(f"Unknown method '{method}'. Choices: {ALL_METHODS}")

    sn  = safe_name(method)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / f"{sn}_test_predictions.csv"

    if out_csv.exists():
        print(f"  [skip] {out_csv} already exists.")
        return

    print(f"\n=== Generating prediction CSV for: {method} ===")

    # ── Load test labels ───────────────────────────────────────────────────────
    test_labels = load_split_labels(WMDP_CSV_PATH, "test")
    print(f"  Test pairs from CSV: {len(test_labels)}")

    # ── Load logit scores ──────────────────────────────────────────────────────
    logit_sn    = "base" if method == "Base" else sn
    logit_scores = load_logit_scores(logit_sn)
    if not logit_scores:
        sys.exit(f"[error] No logit scores for {method} — aborting.")
    print(f"  Logit scores: {len(logit_scores)}")

    # ── Load base training hidden states + labels (to fit probes) ─────────────
    hs_train = load_hs(CHECKPOINT_DIR / "base_hs_train.npy")
    if hs_train is None:
        sys.exit("[error] base_hs_train.npy missing — aborting.")
    y_train = np.array(load_split_labels(WMDP_CSV_PATH, "train"))
    N_train = min(len(y_train), len(hs_train))
    hs_train = hs_train[:N_train]
    y_train  = y_train[:N_train]
    print(f"  Train set: {N_train} samples")

    # ── Load method test hidden states ─────────────────────────────────────────
    hs_test_path = CHECKPOINT_DIR / ("base_hs_test.npy" if method == "Base"
                                     else f"{sn}_hs_test.npy")
    hs_test = load_hs(hs_test_path)
    if hs_test is None:
        sys.exit(f"[error] No test hidden states for {method} — aborting.")

    # ── Fit probes on base training states ─────────────────────────────────────
    early_rf, mid_pipe = fit_probes(hs_train, y_train)

    # ── Align test lengths ─────────────────────────────────────────────────────
    N = min(len(test_labels), len(logit_scores), len(hs_test))
    if N < len(test_labels):
        print(f"  [warn] Truncating to N={N} (labels={len(test_labels)}, "
              f"logit={len(logit_scores)}, hs={len(hs_test)})")
    test_labels  = test_labels[:N]
    logit_scores = logit_scores[:N]
    hs_test      = hs_test[:N]

    # ── Compute probe scores ───────────────────────────────────────────────────
    print(f"  Applying probes to {method} test hidden states...")
    early_rf_proba, mid_linear_score = compute_probe_scores(hs_test, early_rf, mid_pipe)

    # ── Build and save CSV ─────────────────────────────────────────────────────
    rows = []
    for i in range(N):
        t, f   = logit_scores[i][0], logit_scores[i][1]
        margin = t - f
        rows.append({
            "method_name":            method,
            "gold_label":             test_labels[i],
            "raw_margin":             round(margin, 6),
            "abs_raw_margin":         round(abs(margin), 6),
            "early_rf_label_prob":    round(float(early_rf_proba[i]), 8),
            "mid_linear_label_score": round(float(mid_linear_score[i]), 8),
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} rows → {out_csv}")
    print(f"  gold=1: {sum(r['gold_label'] for r in rows)}  "
          f"gold=0: {sum(1 - r['gold_label'] for r in rows)}")


if __name__ == "__main__":
    main()
