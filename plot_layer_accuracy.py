#!/usr/bin/env python3
"""
plot_layer_accuracy.py

Per-layer probe accuracy plot for all 10 models:
  - Base (Instruct)        — meta-llama/Meta-Llama-3-8B-Instruct
  - Llama3-8B              — meta-llama/Meta-Llama-3-8B (non-instruct)
  - 8 unlearning methods   — GradDiff, RMU, RMU-LAT, RepNoise, ELM, RR, TAR, PB&J

X axis: transformer layer index (0 = embedding layer, 1–32 = transformer layers)
Y axis: probe accuracy on the forget-set test split
One subplot per classifier (LR / RF / AdaBoost), 10 lines each.

Usage:
    python plot_layer_accuracy.py [--checkpoint_dir checkpoints]
                                  [--out layer_accuracy.png]
                                  [--clf LR]        # only one classifier
                                  [--skip_embed]    # skip layer 0 (embedding)
"""

import argparse
import csv
import pickle
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless; switch to "TkAgg" / "Qt5Agg" for interactive
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — mirrors hidden_knowledge_after_unlearning.py
# ---------------------------------------------------------------------------

# Display name  →  safe-name prefix used in checkpoint filenames.
# The base model uses the prefix "base"; methods use safe_name(method_name).
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
    "Llama3-8B":       "Llama3-8B",
}

CLF_NAMES = ["LR", "RF", "AdaBoost"]

# Visual style: models with a special style
THICK_MODELS  = {"Base (Instruct)", "Llama3-8B"}
DASHED_MODELS = {"Llama3-8B"}

# Colour palette: 10 distinct colours
_PALETTE = [
    "#1f77b4",   # blue        — Base (Instruct)
    "#ff7f0e",   # orange      — GradDiff
    "#2ca02c",   # green       — RMU
    "#d62728",   # red         — RMU-LAT
    "#9467bd",   # purple      — RepNoise
    "#8c564b",   # brown       — ELM
    "#e377c2",   # pink        — RR
    "#7f7f7f",   # grey        — TAR
    "#bcbd22",   # yellow-green— PB&J
    "#17becf",   # cyan        — Llama3-8B
]

MODEL_COLORS = {name: _PALETTE[i] for i, name in enumerate(ALL_MODELS)}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_y_test(csv_path: Path) -> np.ndarray:
    """Read test-split labels from the WMDP True/False pairs CSV."""
    labels = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                labels.append(1 if row["label"] == "True" else 0)
    if not labels:
        raise RuntimeError(f"No test-split rows found in {csv_path}")
    return np.array(labels, dtype=np.int32)


def load_checkpoint(sn: str, checkpoint_dir: Path):
    """
    Load (hs_test, probe_set) for a model identified by its safe-name prefix.
    Returns (None, None) if any file is missing.
    """
    hs_path    = checkpoint_dir / f"{sn}_hs_test.npy"
    probe_path = checkpoint_dir / f"{sn}_probes.pkl"

    if not hs_path.exists():
        print(f"  [skip] {hs_path.name} not found")
        return None, None
    if not probe_path.exists():
        print(f"  [skip] {probe_path.name} not found")
        return None, None

    hs = np.load(hs_path)                       # (n_test, n_layers, hidden_dim)
    with open(probe_path, "rb") as f:
        probe_set = pickle.load(f)

    if not isinstance(probe_set, dict) or "per_layer" not in probe_set:
        print(f"  [skip] {probe_path.name}: old probe format")
        return None, None

    return hs, probe_set


def per_layer_accuracy(probe_set: dict, hs_test: np.ndarray,
                       y_test: np.ndarray, clf_name: str) -> list:
    """
    Return a list of (layer_idx, accuracy) for every layer in hs_test.
    Layers for which no probe was trained are silently skipped.
    """
    n_layers = hs_test.shape[1]
    layer_probes = probe_set["per_layer"].get(clf_name, {})
    pairs = []
    for l in range(n_layers):
        pipe = layer_probes.get(l)
        if pipe is None:
            continue
        acc = pipe.score(hs_test[:, l, :], y_test)
        pairs.append((l, float(acc)))
    return pairs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(checkpoint_dir: Path, out_path: Path,
              clf_filter: list, skip_embed: bool):

    csv_path = checkpoint_dir / "wmdp_tf_pairs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"WMDP CSV not found at {csv_path}. "
            "Run --stage base first to generate it."
        )

    print("Loading y_test from CSV ...")
    y_test = load_y_test(csv_path)
    print(f"  Test set size: {len(y_test)}  "
          f"(pos={y_test.sum()}  neg={(y_test==0).sum()})")

    clfs = clf_filter if clf_filter else CLF_NAMES
    n_clfs = len(clfs)

    fig, axes = plt.subplots(1, n_clfs,
                             figsize=(7 * n_clfs, 5),
                             sharey=True, squeeze=False)
    fig.suptitle(
        "Per-Layer Probe Accuracy on Forget Set (Test Split)",
        fontsize=13, y=1.01
    )

    legend_handles = []
    legend_labels  = []

    for ax_idx, clf_name in enumerate(clfs):
        ax = axes[0][ax_idx]
        ax.set_title(clf_name, fontsize=12)
        ax.set_xlabel("Layer index", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Accuracy", fontsize=10)
        ax.axhline(0.5, color="gray", linestyle="--",
                   linewidth=0.9, label="Chance (0.5)")
        ax.set_ylim(0.35, 1.02)
        ax.grid(True, alpha=0.25, linewidth=0.5)

        any_plotted = False
        for model_name, sn in ALL_MODELS.items():
            color = MODEL_COLORS[model_name]
            lw    = 2.4 if model_name in THICK_MODELS else 1.2
            ls    = "--" if model_name in DASHED_MODELS else "-"
            zorder = 4 if model_name in THICK_MODELS else 2

            print(f"  Loading {model_name} ({sn}) ...")
            hs, probe_set = load_checkpoint(sn, checkpoint_dir)
            if hs is None:
                continue

            pairs = per_layer_accuracy(probe_set, hs, y_test, clf_name)
            if not pairs:
                print(f"    No per-layer probes found for clf={clf_name}")
                continue

            layers, accs = zip(*pairs)
            if skip_embed:
                # Drop layer 0 (embedding layer output)
                pairs = [(l, a) for l, a in zip(layers, accs) if l > 0]
                if pairs:
                    layers, accs = zip(*pairs)
                else:
                    continue

            ax.plot(layers, accs,
                    color=color, linewidth=lw, linestyle=ls, zorder=zorder)
            any_plotted = True

            # Build legend entry on first subplot only (shared)
            if ax_idx == 0:
                handle = mlines.Line2D(
                    [], [], color=color, linewidth=lw, linestyle=ls,
                    label=model_name
                )
                legend_handles.append(handle)
                legend_labels.append(model_name)

        if not any_plotted:
            ax.text(0.5, 0.5, "No data",
                    ha="center", va="center", transform=ax.transAxes)

        # Mark best-layer dots per method (only on last subplot to avoid clutter)
        # (omitted for clarity)

    # Add chance line to legend
    chance_handle = mlines.Line2D(
        [], [], color="gray", linewidth=0.9, linestyle="--", label="Chance (0.5)"
    )
    legend_handles.append(chance_handle)
    legend_labels.append("Chance (0.5)")

    # Place a single shared legend to the right of all subplots
    fig.legend(
        legend_handles, legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        framealpha=0.9,
        title="Model",
        title_fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot per-layer probe accuracy for all 10 models."
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints",
        help="Directory containing .npy and .pkl checkpoint files (default: checkpoints)"
    )
    parser.add_argument(
        "--out", default="layer_accuracy.png",
        help="Output PNG path (default: layer_accuracy.png)"
    )
    parser.add_argument(
        "--clf", choices=CLF_NAMES, default=None,
        help="Only plot one classifier (default: all three)"
    )
    parser.add_argument(
        "--skip_embed", action="store_true",
        help="Skip layer 0 (embedding layer output) on the x-axis"
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_path       = Path(args.out)
    clf_filter     = [args.clf] if args.clf else []

    make_plot(checkpoint_dir, out_path, clf_filter, args.skip_embed)


if __name__ == "__main__":
    main()
