#!/usr/bin/env python3
"""
plot_layer_accuracy.py

Two plot modes (--mode):

  methods    (default)
             Per-layer probe accuracy for all 10 models at their final-checkpoint
             state.  X axis = layer, one curve per model.

  checkpoints
             Per-layer probe accuracy for ONE unlearning method across all 8
             training checkpoints, with the Base (Instruct) model as reference.
             X axis = layer, one curve per checkpoint + one for base.

             --method METHOD   which method to show (required, or "all")
             --method all      produce one graph per method (8 files)

Two probe-source modes (--probe_source):
  method  (default / Table 3)
          Each model/checkpoint is evaluated with its own per-layer probes.

  base    (Table 2)
          The base model's probes are applied to every model's hidden states.
          Exception: Llama3-8B (methods mode) and Base (Instruct) (both modes)
          always use their own probes.

X axis: transformer layers 1-32  (embedding layer 0 is skipped)
Y axis: probe metric on the forget-set test split
         lower = Y_MIN,  upper = max observed value + 5 % padding

--metric: if omitted, all three metrics are shown as separate rows in one figure.
          If specified, only that metric is shown.

Usage:
    # --- methods mode (original behaviour) ---
    python plot_layer_accuracy.py
    python plot_layer_accuracy.py --probe_source base --out layer_base_probes.png
    python plot_layer_accuracy.py --clf LR --metric true_accuracy

    # --- checkpoints mode ---
    python plot_layer_accuracy.py --mode checkpoints --method GradDiff
    python plot_layer_accuracy.py --mode checkpoints --method RMU --probe_source base
    python plot_layer_accuracy.py --mode checkpoints --method all   # 8 files
    python plot_layer_accuracy.py --mode checkpoints --method all --out ck_plot.png
    #  ^ produces ck_plot_GradDiff.png, ck_plot_RMU.png, ...
"""

import argparse
import csv
import gc
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless; switch to "TkAgg" / "Qt5Agg" for interactive
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
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

# Unlearning methods that have checkpoint sweeps (in the same order as SLURM layout)
SWEEP_METHODS   = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]
N_CHECKPOINTS   = 8

CLF_NAMES          = ["LR", "RF", "AdaBoost"]
METRIC_NAMES       = ["accuracy", "true_accuracy", "false_accuracy"]
PROBE_SOURCE_NAMES = ["method", "base"]

# In base mode (methods), these models always use their own probes.
OWN_PROBE_MODELS = {"Llama3-8B"}

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

# Checkpoint-mode colours: base = same blue as "Base (Instruct)"; ck1-ck8 = plasma gradient
_CK_PLASMA   = cm.plasma(np.linspace(0.15, 0.85, N_CHECKPOINTS))
CK_COLORS    = {"Base (Instruct)": MODEL_COLORS["Base (Instruct)"]}
CK_COLORS.update({f"ck{n}": tuple(_CK_PLASMA[n - 1]) for n in range(1, N_CHECKPOINTS + 1)})
# Ordered label list for checkpoint plots (base first, then ck1..ck8)
CK_LABELS    = ["Base (Instruct)"] + [f"ck{n}" for n in range(1, N_CHECKPOINTS + 1)]

Y_MIN    = 0.4    # fixed lower bound of y-axis
Y_PAD    = 0.05   # padding fraction above the highest point
N_LAYERS = 32     # transformer layers (indices 1-32; layer 0 = embedding, skipped)

METRIC_LABELS = {
    "accuracy":       "Accuracy (overall)",
    "true_accuracy":  "True-label accuracy",
    "false_accuracy": "False-label accuracy",
}

PROBE_SOURCE_TITLES = {
    "method": "Method Probes (Table 3) — each model's own probes",
    "base":   "Base Probes (Table 2) — base probes on all models\n"
              "(Llama3-8B uses its own probes)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Mirror of hidden_knowledge_after_unlearning.safe_name()."""
    return name.replace("&", "_").replace("/", "_").replace(" ", "_")


def _ck_dir(method_name: str, ck_num: int, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"sweep_{_safe_name(method_name)}" / f"ck{ck_num}"


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


def load_hs(sn: str, checkpoint_dir: Path):
    path = checkpoint_dir / f"{sn}_hs_test.npy"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    return np.load(path)


def load_probe_set(sn: str, checkpoint_dir: Path):
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


def load_hs_ck(method_name: str, ck_num: int, checkpoint_dir: Path):
    """Load hidden states for a sweep checkpoint (no method-name prefix)."""
    path = _ck_dir(method_name, ck_num, checkpoint_dir) / "hs_test.npy"
    if not path.exists():
        print(f"  [skip] sweep/{_safe_name(method_name)}/ck{ck_num}/hs_test.npy not found")
        return None
    return np.load(path)


def load_probe_set_ck(method_name: str, ck_num: int, checkpoint_dir: Path):
    """Load probe set for a sweep checkpoint (no method-name prefix)."""
    path = _ck_dir(method_name, ck_num, checkpoint_dir) / "probes.pkl"
    if not path.exists():
        print(f"  [skip] sweep/{_safe_name(method_name)}/ck{ck_num}/probes.pkl not found")
        return None
    with open(path, "rb") as f:
        ps = pickle.load(f)
    if not isinstance(ps, dict) or "per_layer" not in ps:
        print(f"  [skip] sweep/ck{ck_num}/probes.pkl: old probe format")
        return None
    return ps


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def per_layer_metric(probe_set: dict, hs_test: np.ndarray,
                     y_test: np.ndarray, clf_name: str,
                     metric: str, layer_start: int = 1) -> list:
    """
    Return [(layer_idx, value), ...] for layers layer_start .. n_layers-1.
    probe_set and hs_test may come from different models.
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
# Data collection — methods mode (loads each model's hs once for all metrics)
# ---------------------------------------------------------------------------

def collect_data(checkpoint_dir: Path, clfs: list,
                 y_test: np.ndarray, metrics: list,
                 probe_source: str) -> dict:
    """
    Returns data[metric][clf_name][model_name] = (layers_list, values_list).

    probe_source == "method":
        Each model's probes on its own hidden states.
    probe_source == "base":
        Base model's probes on every model's hidden states, EXCEPT models in
        OWN_PROBE_MODELS (currently Llama3-8B) which use their own probes.
    """
    data = {m: {clf: {} for clf in clfs} for m in metrics}

    base_probe_set = None
    if probe_source == "base":
        base_probe_set = load_probe_set("base", checkpoint_dir)
        if base_probe_set is None:
            raise RuntimeError("base_probes.pkl not found — run --stage base first.")
        print("  Base probe set loaded.")

    for model_name, sn in ALL_MODELS.items():
        print(f"  Loading {model_name} ({sn}) ...")

        hs = load_hs(sn, checkpoint_dir)
        if hs is None:
            continue

        # Determine which probe set to use for this model.
        if probe_source == "method" or model_name in OWN_PROBE_MODELS:
            probe_set = load_probe_set(sn, checkpoint_dir)
            if probe_set is None:
                continue
        else:
            probe_set = base_probe_set   # already loaded above

        for metric in metrics:
            for clf_name in clfs:
                pairs = per_layer_metric(probe_set, hs, y_test,
                                         clf_name, metric=metric, layer_start=1)
                if pairs:
                    layers, vals = zip(*pairs)
                    data[metric][clf_name][model_name] = (list(layers), list(vals))

    return data


# ---------------------------------------------------------------------------
# Data collection — checkpoints mode
# ---------------------------------------------------------------------------

def collect_data_checkpoints(checkpoint_dir: Path, method_name: str, clfs: list,
                              y_test: np.ndarray, metrics: list,
                              probe_source: str) -> dict:
    """
    Returns data[metric][clf_name][label] = (layers_list, values_list)
    where label is "Base (Instruct)", "ck1", ..., "ck8".

    probe_source == "method":
        Base model uses base_probes; each checkpoint uses its own probes.pkl.
    probe_source == "base":
        All checkpoints are evaluated with the base model's probes.
        Base (Instruct) still uses its own probes.
    """
    data = {m: {clf: {} for clf in clfs} for m in metrics}

    # Always load base probe set (needed for base (Instruct) curve and optionally probe_source=base)
    base_probe_set = load_probe_set("base", checkpoint_dir)
    if base_probe_set is None:
        raise RuntimeError("base_probes.pkl not found — run --stage base first.")

    # --- Base (Instruct) — always own probes ---
    print("  Loading Base (Instruct) ...")
    base_hs = load_hs("base", checkpoint_dir)
    if base_hs is not None:
        for metric in metrics:
            for clf_name in clfs:
                pairs = per_layer_metric(base_probe_set, base_hs, y_test,
                                         clf_name, metric=metric, layer_start=1)
                if pairs:
                    layers, vals = zip(*pairs)
                    data[metric][clf_name]["Base (Instruct)"] = (list(layers), list(vals))
        del base_hs
        gc.collect()

    # --- Sweep checkpoints — one at a time to keep peak memory low ---
    ck_probe_set = base_probe_set if probe_source == "base" else None

    for ck_num in range(1, N_CHECKPOINTS + 1):
        label = f"ck{ck_num}"
        print(f"  Loading {method_name} {label} ...")

        hs = load_hs_ck(method_name, ck_num, checkpoint_dir)
        if hs is None:
            continue

        if probe_source == "method":
            ck_probe_set = load_probe_set_ck(method_name, ck_num, checkpoint_dir)
            if ck_probe_set is None:
                del hs
                gc.collect()
                continue

        for metric in metrics:
            for clf_name in clfs:
                pairs = per_layer_metric(ck_probe_set, hs, y_test,
                                         clf_name, metric=metric, layer_start=1)
                if pairs:
                    layers, vals = zip(*pairs)
                    data[metric][clf_name][label] = (list(layers), list(vals))

        # Free the large hidden-state array immediately — only scalars are kept.
        del hs
        gc.collect()

    return data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_ax(ax, row_idx, ax_idx, n_rows, clf_name, metric):
    if row_idx == 0:
        ax.set_title(clf_name, fontsize=12)
    if ax_idx == 0:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
    if row_idx == n_rows - 1:
        ax.set_xlabel("Layer", fontsize=10)
    ax.set_xlim(0.5, N_LAYERS + 0.5)
    ax.set_xticks(range(1, N_LAYERS + 1))
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, zorder=1)


def _finalize_figure(fig, axes, legend_handles, legend_labels, n_clfs, out_path):
    legend_handles.append(
        mlines.Line2D([], [], color="gray", linewidth=1.2,
                      linestyle=":", label="Chance (0.5)")
    )
    legend_labels.append("Chance (0.5)")

    right_margin = 0.78 if n_clfs == 3 else (0.72 if n_clfs == 2 else 0.65)
    fig.subplots_adjust(
        left=0.07, right=right_margin,
        bottom=0.07, top=0.94,
        hspace=0.35,
    )
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
    print(f"\nSaved -> {out_path}")


# ---------------------------------------------------------------------------
# Plot — methods mode (original)
# ---------------------------------------------------------------------------

def make_plot(checkpoint_dir: Path, out_path: Path,
              clf_filter: list, metric: str | None, probe_source: str):

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

    clfs            = clf_filter if clf_filter else CLF_NAMES
    metrics_to_plot = [metric] if metric else METRIC_NAMES

    data = collect_data(checkpoint_dir, clfs, y_test, metrics_to_plot, probe_source)

    row_ymax = {}
    for m in metrics_to_plot:
        mx = Y_MIN
        for clf_data in data[m].values():
            for _, vals in clf_data.values():
                mx = max(mx, max(vals))
        row_ymax[m] = min(mx * (1 + Y_PAD), 1.0)

    n_rows = len(metrics_to_plot)
    n_clfs = len(clfs)

    fig, axes = plt.subplots(
        n_rows, n_clfs,
        figsize=(7 * n_clfs, 4.5 * n_rows),
        sharey="row",
        squeeze=False,
    )
    fig.suptitle(PROBE_SOURCE_TITLES[probe_source], fontsize=11, y=1.01)

    legend_handles = []
    legend_labels  = []
    legend_built   = False

    for row_idx, m in enumerate(metrics_to_plot):
        y_max = row_ymax[m]

        for ax_idx, clf_name in enumerate(clfs):
            ax = axes[row_idx][ax_idx]
            _setup_ax(ax, row_idx, ax_idx, n_rows, clf_name, m)
            ax.set_ylim(Y_MIN, y_max)

            clf_data    = data[m].get(clf_name, {})
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

                if not legend_built and ax_idx == 0 and row_idx == 0:
                    legend_handles.append(
                        mlines.Line2D([], [], color=color, linewidth=lw,
                                      linestyle=ls, label=model_name)
                    )
                    legend_labels.append(model_name)

            if not any_plotted:
                ax.text(0.5, 0.5, "No data",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)

        legend_built = True

    _finalize_figure(fig, axes, legend_handles, legend_labels, n_clfs, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot — checkpoints mode (one method across training checkpoints)
# ---------------------------------------------------------------------------

def make_plot_checkpoints(checkpoint_dir: Path, out_path: Path,
                           method_name: str,
                           clf_filter: list, metric: str | None,
                           probe_source: str):

    csv_path = checkpoint_dir / "wmdp_tf_pairs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"WMDP CSV not found at {csv_path}. "
            "Run --stage base first to generate it."
        )

    print(f"\n=== Checkpoints mode: {method_name} ===")
    print("Loading y_test from CSV ...")
    y_test = load_y_test(csv_path)
    print(f"  Test set size: {len(y_test)}  "
          f"(pos={y_test.sum()}  neg={(y_test==0).sum()})")

    clfs            = clf_filter if clf_filter else CLF_NAMES
    metrics_to_plot = [metric] if metric else METRIC_NAMES

    data = collect_data_checkpoints(
        checkpoint_dir, method_name, clfs, y_test, metrics_to_plot, probe_source
    )

    row_ymax = {}
    for m in metrics_to_plot:
        mx = Y_MIN
        for clf_data in data[m].values():
            for _, vals in clf_data.values():
                mx = max(mx, max(vals))
        row_ymax[m] = min(mx * (1 + Y_PAD), 1.0)

    n_rows = len(metrics_to_plot)
    n_clfs = len(clfs)

    probe_lbl = "Method Probes" if probe_source == "method" else "Base Probes"
    fig, axes = plt.subplots(
        n_rows, n_clfs,
        figsize=(7 * n_clfs, 4.5 * n_rows),
        sharey="row",
        squeeze=False,
    )
    fig.suptitle(
        f"{method_name} — Probe Accuracy Over Training Checkpoints  [{probe_lbl}]",
        fontsize=11, y=1.01,
    )

    legend_handles = []
    legend_labels  = []
    legend_built   = False

    for row_idx, m in enumerate(metrics_to_plot):
        y_max = row_ymax[m]

        for ax_idx, clf_name in enumerate(clfs):
            ax = axes[row_idx][ax_idx]
            _setup_ax(ax, row_idx, ax_idx, n_rows, clf_name, m)
            ax.set_ylim(Y_MIN, y_max)

            clf_data    = data[m].get(clf_name, {})
            any_plotted = False

            for label in CK_LABELS:     # Base first, then ck1..ck8
                if label not in clf_data:
                    continue
                layers, vals = clf_data[label]
                color  = CK_COLORS[label]
                is_base = (label == "Base (Instruct)")
                lw     = 2.4 if is_base else 1.4
                ls     = "-"
                zorder = 4 if is_base else 2

                ax.plot(layers, vals,
                        color=color, linewidth=lw, linestyle=ls, zorder=zorder)
                any_plotted = True

                if not legend_built and ax_idx == 0 and row_idx == 0:
                    legend_handles.append(
                        mlines.Line2D([], [], color=color, linewidth=lw,
                                      linestyle=ls, label=label)
                    )
                    legend_labels.append(label)

            if not any_plotted:
                ax.text(0.5, 0.5, "No data",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)

        legend_built = True

    _finalize_figure(fig, axes, legend_handles, legend_labels, n_clfs, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-layer probe accuracy.\n"
            "  methods mode     — all models at final checkpoint (original)\n"
            "  checkpoints mode — one method across training checkpoints"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["methods", "checkpoints"], default="methods",
        help=(
            "Plot mode (default: methods).\n"
            "  methods      — all 10 models at final checkpoint state\n"
            "  checkpoints  — one unlearning method across 8 training checkpoints"
        ),
    )
    parser.add_argument(
        "--method", default=None,
        help=(
            "For --mode checkpoints: method name or 'all' (to produce one file per method).\n"
            f"Available: {', '.join(SWEEP_METHODS + ['all'])}"
        ),
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints",
        help="Checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--out", default="layer_accuracy.png",
        help=(
            "Output PNG path (default: layer_accuracy.png).\n"
            "For --mode checkpoints --method all, used as a filename template:\n"
            "  e.g. layer_accuracy.png → layer_accuracy_GradDiff.png, ..."
        )
    )
    parser.add_argument(
        "--clf", choices=CLF_NAMES, default=None,
        help="Only plot one classifier (default: all three)"
    )
    parser.add_argument(
        "--metric", choices=METRIC_NAMES, default=None,
        help=(
            "Metric to plot (default: all three as separate rows).\n"
            "Options: accuracy | true_accuracy | false_accuracy"
        ),
    )
    parser.add_argument(
        "--probe_source", choices=PROBE_SOURCE_NAMES, default="method",
        help=(
            "Which probes to use (default: method).\n"
            "  method — each model/checkpoint's own probes (Table 3)\n"
            "  base   — base probes on all models; base (Instruct) always uses own probes"
        ),
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_path       = Path(args.out)
    clf_filter     = [args.clf] if args.clf else []

    if args.mode == "methods":
        make_plot(
            checkpoint_dir=checkpoint_dir,
            out_path=out_path,
            clf_filter=clf_filter,
            metric=args.metric,
            probe_source=args.probe_source,
        )

    else:  # checkpoints
        if args.method is None:
            parser.error("--mode checkpoints requires --method METHOD (or 'all').")

        methods = SWEEP_METHODS if args.method == "all" else [args.method]

        if args.method not in SWEEP_METHODS and args.method != "all":
            parser.error(
                f"Unknown method '{args.method}'. "
                f"Choose from: {', '.join(SWEEP_METHODS + ['all'])}"
            )

        for method in methods:
            if len(methods) > 1:
                # Auto-name: stem_Method.suffix
                file_out = out_path.parent / f"{out_path.stem}_{_safe_name(method)}{out_path.suffix}"
            else:
                file_out = out_path

            make_plot_checkpoints(
                checkpoint_dir=checkpoint_dir,
                out_path=file_out,
                method_name=method,
                clf_filter=clf_filter,
                metric=args.metric,
                probe_source=args.probe_source,
            )


if __name__ == "__main__":
    main()
