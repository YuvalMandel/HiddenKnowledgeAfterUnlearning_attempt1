#!/usr/bin/env python3
"""
generate_dual_confidence_inputs.py
===================================
Generates a per-method prediction CSV for the dual-confidence figure package.

Reads from the main pipeline checkpoints (no GPU needed):
  checkpoints/base_partial.json          (or {sn}_partial.json)  → logit_scores
  checkpoints/base_hs_test.npy           (or {sn}_hs_test.npy)   → hidden states
  checkpoints/base_probes.pkl                                      → probe pipelines
  data/wmdp_tf_pairs.csv                                           → gold labels & test order

Always uses BASE probes applied to the method's hidden states (measuring
whether base-trained probe directions survive unlearning).

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
import pickle
import re
import sys
from pathlib import Path

import numpy as np

CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR       = Path("data")
WMDP_CSV_PATH  = DATA_DIR / "wmdp_tf_pairs.csv"

ALL_METHODS = ["Base", "GradDiff", "RMU", "RMU-LAT", "RepNoise",
               "ELM", "RR", "TAR", "PB&J"]


def safe_name(method_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", method_name)


def load_test_labels(csv_path: Path) -> list[dict]:
    """Return list of {gold_label, pair_type} for the test split, in CSV order."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            gold = 1 if row["label"].strip().lower() == "true" else 0
            rows.append({"gold_label": gold, "pair_type": row["pair_type"]})
    return rows


def load_logit_scores(sn: str) -> list:
    """
    Returns list of [true_logit, false_logit] per test example.
    Checks partial.json first, then results.json (method format).
    """
    # Try partial.json (base and method both save here)
    partial_path = CHECKPOINT_DIR / f"{sn}_partial.json"
    if partial_path.exists():
        with open(partial_path) as f:
            partial = json.load(f)
        if "logit_scores" in partial:
            return partial["logit_scores"]

    # Try method results.json — for methods the key is "logit" (stats only)
    # but logit_scores are only in partial.json; warn if missing
    print(f"  [warn] logit_scores not found in {partial_path} — "
          f"make sure the pipeline ran to completion for {sn}",
          file=sys.stderr)
    return []


def load_hs_test(sn: str) -> np.ndarray | None:
    """Load hs_test.npy for Base or a method."""
    # Base is stored as base_hs_test.npy
    if sn.lower() == "base":
        path = CHECKPOINT_DIR / "base_hs_test.npy"
    else:
        path = CHECKPOINT_DIR / f"{sn}_hs_test.npy"
    if not path.exists():
        print(f"  [warn] Hidden states not found: {path}", file=sys.stderr)
        return None
    hs = np.load(path)
    print(f"  Loaded {path.name}  shape={hs.shape}")
    return hs


def load_base_probes() -> dict | None:
    path = CHECKPOINT_DIR / "base_probes.pkl"
    if not path.exists():
        print(f"  [error] base_probes.pkl not found at {path}", file=sys.stderr)
        return None
    with open(path, "rb") as f:
        ps = pickle.load(f)
    print(f"  Loaded base_probes.pkl")
    return ps


def compute_probe_scores(hs: np.ndarray, probe_set: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      early_rf_proba    (N,) — init_band RF predict_proba[:, 1]
      mid_linear_score  (N,) — mid_band  LR decision_function
    """
    # ── Early RF: init_band (layers 1–9) ────────────────────────────────────
    ib_start, ib_end = probe_set["init_band_range"]
    X_ib = hs[:, ib_start: ib_end + 1, :].reshape(len(hs), -1)
    rf_pipe = probe_set["init_band"]["RF"]
    # Force single-threaded RF to avoid /dev/shm exhaustion on shared nodes
    rf_pipe[-1].set_params(n_jobs=1)
    early_rf_proba = rf_pipe.predict_proba(X_ib)[:, 1]

    # ── Mid linear: mid_band (layers 10–22) ─────────────────────────────────
    mb_start, mb_end = probe_set["mid_band_range"]
    X_mb = hs[:, mb_start: mb_end + 1, :].reshape(len(hs), -1)
    lr_pipe = probe_set["mid_band"]["LR"]
    mid_linear_score = lr_pipe.decision_function(X_mb)

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

    sn = safe_name(method)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / f"{sn}_test_predictions.csv"

    if out_csv.exists():
        print(f"  [skip] {out_csv} already exists.")
        return

    print(f"\n=== Generating prediction CSV for: {method} ===")

    # ── Load test labels ──────────────────────────────────────────────────────
    test_labels = load_test_labels(WMDP_CSV_PATH)
    N_labels = len(test_labels)
    print(f"  Test pairs from CSV: {N_labels}")

    # ── Load logit scores ─────────────────────────────────────────────────────
    logit_sn = "base" if method == "Base" else sn
    logit_scores = load_logit_scores(logit_sn)
    if not logit_scores:
        sys.exit(f"[error] No logit scores for {method} — aborting.")
    print(f"  Logit scores: {len(logit_scores)}")

    # ── Load hidden states ────────────────────────────────────────────────────
    hs = load_hs_test("base" if method == "Base" else sn)
    if hs is None:
        sys.exit(f"[error] No hidden states for {method} — aborting.")

    # ── Load base probes ──────────────────────────────────────────────────────
    probe_set = load_base_probes()
    if probe_set is None:
        sys.exit("[error] base_probes.pkl missing — aborting.")

    # ── Align lengths ─────────────────────────────────────────────────────────
    N = min(len(test_labels), len(logit_scores), len(hs))
    if N < len(test_labels):
        print(f"  [warn] Truncating to N={N} (labels={len(test_labels)}, "
              f"logit={len(logit_scores)}, hs={len(hs)})")
    test_labels  = test_labels[:N]
    logit_scores = logit_scores[:N]
    hs           = hs[:N]

    # ── Compute probe scores ──────────────────────────────────────────────────
    print(f"  Computing probe scores (base probes on {method} hidden states)...")
    early_rf_proba, mid_linear_score = compute_probe_scores(hs, probe_set)

    # ── Build and save CSV ────────────────────────────────────────────────────
    rows = []
    for i in range(N):
        t, f   = logit_scores[i][0], logit_scores[i][1]
        margin = t - f
        rows.append({
            "method_name":          method,
            "gold_label":           test_labels[i]["gold_label"],
            "raw_margin":           round(margin, 6),
            "abs_raw_margin":       round(abs(margin), 6),
            "early_rf_label_prob":  round(float(early_rf_proba[i]), 8),
            "mid_linear_label_score": round(float(mid_linear_score[i]), 8),
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} rows → {out_csv}")
    print(f"  gold=1: {sum(r['gold_label'] for r in rows)}  "
          f"gold=0: {sum(1-r['gold_label'] for r in rows)}")


if __name__ == "__main__":
    main()
