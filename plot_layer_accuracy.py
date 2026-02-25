#!/usr/bin/env python3
"""
plot_layer_accuracy.py

Per-layer probe accuracy plot for all 10 models:
  - Base (Instruct)        — meta-llama/Meta-Llama-3-8B-Instruct
  - Llama3-8B              — meta-llama/Meta-Llama-3-8B (non-instruct)
  - 8 unlearning methods   — GradDiff, RMU, RMU-LAT, RepNoise, ELM, RR, TAR, PB&J

Two probe-source modes (--probe_source):
  method  (default / Table 3)
          Each model is evaluated with its own per-layer probes trained on its
          own hidden states.  This is the standard unlearning-probe comparison.

  base    (Table 2)
          The base model's probes (trained on base hidden states) are applied to
          every model's hidden states.  This tests whether the base model's linear
          classifiers still transfer to unlearned representations.

X axis: transformer layers 1–32  (embedding layer 0 is skipped)
Y axis: probe metric on the forget-set test split
         lower = Y_MIN,  upper = max observed value + 5 % padding
One subplot per classifier (LR / RF / AdaBoost), 10 lines each.

Usage:
    # method probes (default, Table 3)
    python plot_layer_accuracy.py

    # base probes applied to all models
    python plot_layer_accuracy.py --probe_source base --out layer_base_probes.png

    # single classifier, true-accuracy metric
    python plot_layer_accuracy.py --clf LR --metric true_accuracy
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

CLF_NAMES         = ["LR", "RF", "AdaBoost"]
METRIC_NAMES      = ["accuracy", "true_accuracy", "false_accuracy"]
PROBE_SOURCE_NAMES = ["method", "base"]

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

Y_MIN    = 0.4    # fixed lower bound of y-axis
Y_PAD    = 0.05   # padding fraction above the highest point
N_LAYERS = 32     # transformer layers (indices 1–32; layer 0 = embedding, skipped)

METRIC_LABELS = {
    "accuracy":       "Accuracy (overall)",
    "true_accuracy":  "Accuracy on True-label examples",
    "false_accuracy": "Accuracy on False-label examples",
}

PROBE_SOURCE_TITLES = {
    "method": "Method Probes (Table 3) — each model's own probes",
    "base":   "Base Probes (Table 2) applied to all models",
}

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


def load_hs(sn: str, checkpoint_dir: Path) -> np.ndarray | None:
    """Load test hidden states for a model; returns None if file missing."""
    path = checkpoint_dir / f"{sn}_hs_test.npy"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    return np.load(path)


def load_probe_set(sn: str, checkpoint_dir: Path) -> dict | None:
    """Load the probe set for a model; returns None if file missing or wrong format."""
    path = checkpoint_dir / f"{sn}_probes.pkl"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    with open(path, "rb") as f:
        ps = pickle.load(f)
    if not isinstance(ps, dict) or "per_layer" not in ps:
        print(f"  [skip] {path.name}: old probe format")
        return None
    return ps


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def per_layer_metric(probe_set: dict, hs_test: np.ndarray,
                     y_test: np.ndarray, clf_name: str,
                     metric: str = "accuracy",
                     layer_start: int = 1) -> list:
    """
    Return [(layer_idx, metric_value), ...] for layers layer_start … n_layers-1.
    Uses probes from probe_set but hidden states from hs_test — these can come
    from different models (e.g. base probes on an unlearned model's hs).
    metric: "accuracy" | "true_accuracy" | "false_accuracy"
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
        preds = pipe.predict(hs_test[:, l, :])
        if metric == "accuracy":
            val = float((preds == y_test).mean())
        elif metric == "true_accuracy":
            val = float((preds[true_mask] == 1).mean()) if true_mask.any() else 0.0
        else:  # false_accuracy
            val = float((preds[false_mask] == 0).mean()) if false_mask.any() else 0.0
        result.append((l, val))
    return result


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data(checkpoint_dir: Path, clfs: list,
                 y_test: np.ndarray, metric: str,
                 probe_source: str) -> dict:
    """
    Returns data[clf_name][model_name] = (layers_list, values_list).

    probe_source == "method":
        Each model's probes are evaluated on that model's own hidden states.
    probe_source == "base":
        The base model's probes are evaluated on every model's hidden states.
    """
    data = {clf: {} for clf in clfs}

    if probe_source == "base":
        # Load the base probe set once.
        base_probe_set = load_probe_set("base", checkpoint_dir)
        if base_probe_set is None:
            raise RuntimeError("base_probes.pkl not found — run --stage base first.")
        print(f"  Base probe set loaded.")

        for model_name, sn in ALL_MODELS.items():
            print(f"  Loading hs for {model_name} ({sn}) ...")
            hs = load_hs(sn, checkpoint_dir)
            if hs is None:
                continue
            for clf_name in clfs:
                pairs = per_layer_metric(base_probe_set, hs, y_test,
                                         clf_name, metric=metric, layer_start=1)
                if pairs:
                    layers, vals = zip(*pairs)
                    data[clf_name][model_name] = (list(layers), list(vals))

    else:  # probe_source == "method"
        for model_name, sn in ALL_MODELS.items():
            print(f"  Loading {model_name} ({sn}) ...")
            hs        = load_hs(sn, checkpoint_dir)
            probe_set = load_probe_set(sn, checkpoint_dir)
            if hs is None or probe_set is None:
                continue
            for clf_name in clfs:
                pairs = per_layer_metric(probe_set, hs, y_test,
                                         clf_name, metric=metric, layer_start=1)
                if pairs:
                    layers, vals = zip(*pairs)
                    data[clf_name][model_name] = (list(layers), list(vals))

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(checkpoint_dir: Path, out_path: Path,
              clf_filter: list, metric: str, probe_source: str):

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

    # ── First pass: collect all curve data & find global max ──────────────────
    data       = collect_data(checkpoint_dir, clfs, y_test, metric, probe_source)
    global_max = Y_MIN
    for clf_data in data.values():
        for layers, vals in clf_data.values():
            global_max = max(global_max, max(vals))

    y_max = min(global_max * (1 + Y_PAD), 1.0)

    # ── Second pass: draw ─────────────────────────────────────────────────────
    n_clfs = len(clfs)
    fig, axes = plt.subplots(1, n_clfs,
                             figsize=(7 * n_clfs, 5),
                             sharey=True, squeeze=False)
    fig.suptitle(
        f"{PROBE_SOURCE_TITLES[probe_source]}\n"
        f"{METRIC_LABELS[metric]} — Forget Set (Test Split)",
        fontsize=11,
    )

    legend_handles = []
    legend_labels  = []

    for ax_idx, clf_name in enumerate(clfs):
        ax = axes[0][ax_idx]
        ax.set_title(clf_name, fontsize=12)
        ax.set_xlabel("Layer", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)

        ax.set_xlim(0.5, N_LAYERS + 0.5)
        ax.set_ylim(Y_MIN, y_max)
        ax.set_xticks(range(1, N_LAYERS + 1))
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, zorder=1)

        clf_data    = data.get(clf_name, {})
        any_plotted = False

        for model_name in ALL_MODELS:      # consistent colour ordering
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

            if ax_idx == 0:
                legend_handles.append(
                    mlines.Line2D([], [], color=color, linewidth=lw,
                                  linestyle=ls, label=model_name)
                )
                legend_labels.append(model_name)

        if not any_plotted:
            ax.text(0.5, 0.5, "No data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)

    legend_handles.append(
        mlines.Line2D([], [], color="gray", linewidth=1.2,
                      linestyle=":", label="Chance (0.5)")
    )
    legend_labels.append("Chance (0.5)")

    right_margin = 0.78 if n_clfs == 3 else (0.72 if n_clfs == 2 else 0.65)
    fig.subplots_adjust(left=0.07, right=right_margin, bottom=0.13, top=0.89)

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
        description="Plot per-layer probe accuracy for all 10 models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
    parser.add_argument(
        "--metric", choices=METRIC_NAMES, default="accuracy",
        help="Metric to plot: accuracy (default) | true_accuracy | false_accuracy"
    )
    parser.add_argument(
        "--probe_source", choices=PROBE_SOURCE_NAMES, default="method",
        help=(
            "Which probes to use (default: method).\n"
            "  method — each model evaluated with its own probes (Table 3)\n"
            "  base   — base model's probes applied to every model's hidden states"
        ),
    )
    args = parser.parse_args()

    make_plot(
        checkpoint_dir=Path(args.checkpoint_dir),
        out_path=Path(args.out),
        clf_filter=[args.clf] if args.clf else [],
        metric=args.metric,
        probe_source=args.probe_source,
    )


if __name__ == "__main__":
    main()
