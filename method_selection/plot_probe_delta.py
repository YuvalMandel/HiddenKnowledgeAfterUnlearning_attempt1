#!/usr/bin/env python3
"""
plot_probe_delta.py — Per-layer AUC delta: method probe minus frozen base probe.

For every method and every transformer layer, computes:
    delta[layer] = method_probe_AUC[layer] - base_probe_AUC_on_method[layer]

where:
  - method_probe_AUC[layer]: AUC of a probe trained on the *method's* train hidden
    states and evaluated on the method's test hidden states at that layer
    (read from data/kfold_per_layer_all.csv, mean over 5 folds).
  - base_probe_AUC_on_method[layer]: AUC of the *base model's* probe (trained on
    base train hidden states) evaluated on the method's test hidden states at
    that layer (computed here by loading checkpoints/base_probes.pkl).

A positive delta means the method-specific probe recovers information that the
frozen base probe misses — i.e. unlearning shifted the representational geometry
to a new subspace that still encodes the forget-domain knowledge.

Usage:
    python method_selection/plot_probe_delta.py
    python method_selection/plot_probe_delta.py --out_dir plots --clf LR
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CHECKPOINTS_DIR = Path("checkpoints")
DATA_DIR = Path("data")

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]

METHOD_DISPLAY = {
    "GradDiff": "GradDiff", "RMU": "RMU", "RMU-LAT": "RMU-LAT",
    "RepNoise": "RepNoise", "ELM": "ELM", "RR": "RR",
    "TAR": "TAR", "PB_J": "PB&J",
}
METHOD_COLORS = {
    "GradDiff": "#e41a1c", "RMU": "#377eb8", "RMU-LAT": "#4daf4a",
    "RepNoise": "#984ea3", "ELM": "#ff7f00", "RR": "#a65628",
    "TAR": "#f781bf", "PB_J": "#999999",
}

# CSV uses PB&J; we translate
CSV_NAME_MAP = {m: m for m in METHODS}
CSV_NAME_MAP["PB_J"] = "PB&J"


def load_y_test() -> np.ndarray:
    df = pd.read_csv(CHECKPOINTS_DIR / "bio_labels.csv")
    return df[df["split"] == "test"]["gold_label"].values


def load_base_per_layer_probes(clf_name: str) -> dict:
    """Returns {layer_idx: sklearn Pipeline} for base probes."""
    with open(CHECKPOINTS_DIR / "base_probes.pkl", "rb") as f:
        bp = pickle.load(f)
    return bp["per_layer"][clf_name]


def apply_probes_per_layer(probes: dict, hs: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate each layer probe on hidden states hs, return {layer: AUC}."""
    aucs = {}
    for layer_idx, clf in probes.items():
        X = hs[:, layer_idx, :]
        try:
            scores = clf.predict_proba(X)[:, 1]
            aucs[layer_idx] = roc_auc_score(y, scores)
        except Exception:
            aucs[layer_idx] = float("nan")
    return aucs


def load_kfold_method_auc(clf_name: str) -> dict[str, dict[int, float]]:
    """Load mean per-layer method probe AUC from kfold_per_layer_all.csv."""
    df = pd.read_csv(DATA_DIR / "kfold_per_layer_all.csv")
    df = df[df["clf"] == clf_name]
    result: dict[str, dict[int, float]] = {}
    for method in METHODS:
        csv_name = CSV_NAME_MAP[method]
        sub = df[df["model"] == csv_name]
        if sub.empty:
            print(f"  WARNING: no kfold data for {method} (csv_name={csv_name})")
            result[method] = {}
            continue
        result[method] = dict(sub.groupby("layer")["auc"].mean())
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="plots")
    ap.add_argument("--clf", default="LR", choices=["LR", "RF", "AdaBoost"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading y_test...")
    y_test = load_y_test()

    print("Loading base per-layer probes...")
    base_probes = load_base_per_layer_probes(args.clf)
    layers = sorted(base_probes.keys())

    print("Loading kfold per-layer method probe AUC...")
    kfold_aucs = load_kfold_method_auc(args.clf)

    # Base probe on base states (reference line)
    print("Computing base probe AUC on base states...")
    base_hs = np.load(CHECKPOINTS_DIR / "base_hs_test.npy")
    base_on_base = apply_probes_per_layer(base_probes, base_hs, y_test)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Reference: base method probe AUC from kfold vs base probe on base states
    kfold_base = kfold_aucs.get("base", {})  # might not exist
    # Just plot the delta=0 reference
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, label="No recovery (delta=0)")

    for method in METHODS:
        print(f"  Processing {method}...")
        hs_path = CHECKPOINTS_DIR / f"{method}_hs_test.npy"
        if not hs_path.exists():
            print(f"    WARNING: {hs_path} not found, skipping")
            continue
        method_hs = np.load(hs_path)

        # Base probe applied to this method's hidden states
        base_on_method = apply_probes_per_layer(base_probes, method_hs, y_test)

        # Method probe AUC from kfold (mean over folds)
        method_auc = kfold_aucs.get(method, {})

        delta_x, delta_y = [], []
        for l in layers:
            base_auc = base_on_method.get(l, float("nan"))
            meth_auc = method_auc.get(l, float("nan"))
            if not (np.isnan(base_auc) or np.isnan(meth_auc)):
                delta_x.append(l)
                delta_y.append(meth_auc - base_auc)

        color = METHOD_COLORS.get(method, "#333333")
        ax.plot(delta_x, delta_y,
                label=METHOD_DISPLAY.get(method, method),
                color=color, linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel(r"$\Delta$ AUC (Method Probe $-$ Base Probe on Method States)", fontsize=11)
    ax.set_title(
        f"Per-Layer Probe Recovery: How Much Does Re-Training the Probe Help?\n"
        f"({args.clf}, evaluated on each method's own test hidden states)",
        fontsize=12,
    )
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        out = os.path.join(args.out_dir, f"probe_delta_per_layer.{ext}")
        plt.savefig(out, dpi=220)
        print(f"Saved: {out}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
