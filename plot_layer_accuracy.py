#!/usr/bin/env python3
"""
plot_layer_accuracy.py

Per-layer probe accuracy plot for all 10 models:
  - Base (Instruct)        — meta-llama/Meta-Llama-3-8B-Instruct
  - Llama3-8B              — meta-llama/Meta-Llama-3-8B (non-instruct)
  - 8 unlearning methods   — GradDiff, RMU, RMU-LAT, RepNoise, ELM, RR, TAR, PB&J

X axis: transformer layers 1–32  (embedding layer 0 is skipped)
Y axis: probe accuracy on the forget-set test split
         lower = 0.475,  upper = max observed accuracy + 5 % padding
One subplot per classifier (LR / RF / AdaBoost), 10 lines each.

Usage:
    python plot_layer_accuracy.py [--checkpoint_dir checkpoints]
                                  [--out layer_accuracy.png]
                                  [--clf LR]   # only one classifier
"""

import argparse
import csv
import pickle
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

Y_MIN = 0.475          # fixed lower bound
Y_PAD = 0.05           # 5 % padding above the highest point
N_LAYERS = 32          # transformer layers (indices 1–32; layer 0 = embedding, skipped)

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


def load_checkpoint(sn: str, checkpoint_dir: Path):
    hs_path    = checkpoint_dir / f"{sn}_hs_test.npy"
    probe_path = checkpoint_dir / f"{sn}_probes.pkl"

    if not hs_path.exists():
        print(f"  [skip] {hs_path.name} not found")
        return None, None
    if not probe_path.exists():
        print(f"  [skip] {probe_path.name} not found")
        return None, None

    hs = np.load(hs_path)
    with open(probe_path, "rb") as f:
        probe_set = pickle.load(f)

    if not isinstance(probe_set, dict) or "per_layer" not in probe_set:
        print(f"  [skip] {probe_path.name}: old probe format")
        return None, None

    return hs, probe_set


def per_layer_accuracy(probe_set: dict, hs_test: np.ndarray,
                       y_test: np.ndarray, clf_name: str,
                       layer_start: int = 1) -> list:
    """
    Return [(layer_idx, accuracy), ...] for layers layer_start … n_layers-1.
    Layer 0 (embedding) is skipped by default.
    """
    n_layers     = hs_test.shape[1]
    layer_probes = probe_set["per_layer"].get(clf_name, {})
    result = []
    for l in range(layer_start, n_layers):
        pipe = layer_probes.get(l)
        if pipe is None:
            continue
        acc = pipe.score(hs_test[:, l, :], y_test)
        result.append((l, float(acc)))
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(checkpoint_dir: Path, out_path: Path, clf_filter: list):

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

    # ── First pass: collect all curve data & find global max accuracy ─────────
    # data[clf_name][model_name] = (layers_list, accs_list)
    data = {clf: {} for clf in clfs}
    global_max = Y_MIN   # will be updated

    for model_name, sn in ALL_MODELS.items():
        print(f"  Loading {model_name} ({sn}) ...")
        hs, probe_set = load_checkpoint(sn, checkpoint_dir)
        if hs is None:
            continue
        for clf_name in clfs:
            pairs = per_layer_accuracy(probe_set, hs, y_test, clf_name, layer_start=1)
            if not pairs:
                continue
            layers, accs = zip(*pairs)
            data[clf_name][model_name] = (list(layers), list(accs))
            global_max = max(global_max, max(accs))

    y_max = global_max * (1 + Y_PAD)

    # ── Second pass: draw ─────────────────────────────────────────────────────
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
        ax.set_xlabel("Layer", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Accuracy", fontsize=10)

        # Axes limits and ticks
        ax.set_xlim(0.5, N_LAYERS + 0.5)
        ax.set_ylim(Y_MIN, y_max)
        ax.set_xticks(range(1, N_LAYERS + 1))
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=8)

        # Grid
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)

        # Chance line
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, zorder=1)

        clf_data = data.get(clf_name, {})
        any_plotted = False

        for model_name in ALL_MODELS:        # keep consistent ordering
            if model_name not in clf_data:
                continue
            layers, accs = clf_data[model_name]
            color  = MODEL_COLORS[model_name]
            lw     = 2.4 if model_name in THICK_MODELS else 1.2
            ls     = "--" if model_name in DASHED_MODELS else "-"
            zorder = 4 if model_name in THICK_MODELS else 2

            ax.plot(layers, accs,
                    color=color, linewidth=lw, linestyle=ls, zorder=zorder)
            any_plotted = True

            if ax_idx == 0:
                legend_handles.append(
                    mlines.Line2D([], [], color=color, linewidth=lw,
                                  linestyle=ls, label=model_name)
                )
                legend_labels.append(model_name)

        if not any_plotted:
            ax.text(0.5, 0.5, "No data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)

    # Chance line legend entry
    legend_handles.append(
        mlines.Line2D([], [], color="gray", linewidth=1.2,
                      linestyle=":", label="Chance (0.5)")
    )
    legend_labels.append("Chance (0.5)")

    # Reserve space on the right for the legend and at the bottom for x-axis labels.
    # tight_layout() alone clips the bottom when a figure-level legend is placed
    # outside the axes, so we use explicit margins instead.
    right_margin = 0.78 if n_clfs == 3 else (0.72 if n_clfs == 2 else 0.65)
    fig.subplots_adjust(left=0.06, right=right_margin, bottom=0.13, top=0.93)

    fig.legend(
        legend_handles, legend_labels,
        loc="center left",
        bbox_to_anchor=(right_margin + 0.01, 0.5),
        fontsize=9,
        framealpha=0.9,
        title="Model",
        title_fontsize=9,
    )

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
        help="Checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--out", default="layer_accuracy.png",
        help="Output PNG path (default: layer_accuracy.png)"
    )
    parser.add_argument(
        "--clf", choices=CLF_NAMES, default=None,
        help="Only plot one classifier (default: all three)"
    )
    args = parser.parse_args()

    make_plot(
        checkpoint_dir=Path(args.checkpoint_dir),
        out_path=Path(args.out),
        clf_filter=[args.clf] if args.clf else [],
    )


if __name__ == "__main__":
    main()
