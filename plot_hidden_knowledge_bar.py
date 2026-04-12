#!/usr/bin/env python3
"""
plot_hidden_knowledge_bar.py

Two bar plots:

  Plot 1 — Accuracy (surface vs internal):
    • Generation accuracy   (surface)
    • Logit accuracy        (surface)
    • Full-layer probe acc  (internal — LR on all-layers concat, PCA-256)

  Plot 2 — AUC (surface vs internal):
    • Logit AUC             (surface — no gen AUC exists)
    • Full-layer probe AUC  (internal — LR on all-layers concat, PCA-256)

Data sources:
  data/summary_table1_gen_logit.csv     -> gen_acc, logit_acc, logit_auc
  data/summary_table2_base_probes.csv   -> fl_lr_acc, fl_lr_auc  (Base row)
  data/summary_table3_method_probes.csv -> fl_lr_acc, fl_lr_auc  (method rows)

K-fold mode (--kfold):
  Reads data/kfold_table3_probes.csv for probe values (mean ± CI95).
  Gen/logit values still come from the single-fold summary tables (no kfold gen/logit).
  Probe bars show mean height; ±CI95 is drawn as an error cap on each bar.
"""

import argparse
import csv
import matplotlib
matplotlib.use("Agg")   # no popup window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

# ── Model order & display names ───────────────────────────────────────────────

MODELS = [
    ("Base",      "Base\n(Llama-3-8B-I)"),
    ("GradDiff",  "GradDiff"),
    ("RMU",       "RMU"),
    ("RMU-LAT",   "RMU-LAT"),
    ("RepNoise",  "RepNoise"),
    ("ELM",       "ELM"),
    ("RR",        "RR"),
    ("TAR",       "TAR"),
    ("PB&J",      "PB&J"),
]

# kfold CSV uses "base" for the base model; others match
_KFOLD_KEY = {
    "Base": "base",
    **{k: k for k, _ in MODELS[1:]},
}

# ── Shared helpers ────────────────────────────────────────────────────────────

COLORS = {
    "gen":   "#E07B54",
    "logit": "#F2C14E",
    "probe": "#4C72B0",
}


def read_csv_as_dict(path, key_col="method"):
    with open(path, newline="") as f:
        return {row[key_col]: row for row in csv.DictReader(f)}


def fget(row, col):
    v = row.get(col, "")
    return float(v) if v not in ("", None) else float("nan")


def add_value_labels(ax, bar_groups):
    for bars in bar_groups:
        for bar in bars:
            top = bar.get_y() + bar.get_height()
            mid = bar.get_y() + bar.get_height() / 2
            if not np.isnan(top):
                ax.text(bar.get_x() + bar.get_width() / 2, mid,
                        f"{top:.2f}", ha="center", va="center",
                        fontsize=6, rotation=0, color="white",
                        fontweight="bold")


def add_separators(ax, n):
    for i in range(1, n):
        ax.axvline(i - 0.5, color="lightgrey", linewidth=0.8, zorder=1)


def format_yaxis(ax, ylabel, ymin=0, ymax=1.0, step=0.1):
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(ymin, ymax + step / 2, step))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def add_errorbars(ax, x_positions, values, errors, width, color):
    """Draw symmetric error caps centred on each bar top."""
    errs = np.array(errors, dtype=float)
    vals = np.array(values, dtype=float)
    valid = ~np.isnan(errs) & ~np.isnan(vals)
    if not valid.any():
        return
    ax.errorbar(
        x_positions[valid], vals[valid],
        yerr=errs[valid],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        zorder=5,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kfold", action="store_true", default=False,
                    help=(
                        "Use 5-fold aggregated probe values (mean ± CI95) "
                        "from kfold_table3_probes.csv. "
                        "Gen/logit bars remain single-fold values."
                    ))
    ap.add_argument("--out_acc", default=None,
                    help="Output path for accuracy plot (default: auto)")
    ap.add_argument("--out_auc", default=None,
                    help="Output path for AUC plot (default: auto)")
    ap.add_argument("--figsize", nargs=2, type=float, default=None,
                    metavar=("W", "H"),
                    help="Figure size in inches, e.g. --figsize 5.5 3.5")
    ap.add_argument("--format", default=None, choices=["png", "pdf", "svg"],
                    dest="fmt",
                    help="Output file format (default: png). Use pdf for Overleaf/LaTeX.")
    args = ap.parse_args()

    kfold  = args.kfold
    suffix = "_kfold" if kfold else ""
    ext    = f".{args.fmt}" if args.fmt else ".png"
    out1   = Path(args.out_acc) if args.out_acc else Path(f"hidden_knowledge_bar_acc{suffix}{ext}")
    out2   = Path(args.out_auc) if args.out_auc else Path(f"hidden_knowledge_bar_auc{suffix}{ext}")

    # ── Load surface tables (always single-fold) ──────────────────────────────
    t1 = read_csv_as_dict(DATA_DIR / "summary_table1_gen_logit.csv")
    t2 = read_csv_as_dict(DATA_DIR / "summary_table2_base_probes.csv")
    t3 = read_csv_as_dict(DATA_DIR / "summary_table3_method_probes.csv")

    labels     = [lbl for _, lbl in MODELS]
    N          = len(MODELS)
    x          = np.arange(N)
    gen_accs   = [fget(t1[k], "gen_acc")   for k, _ in MODELS]
    logit_accs = [fget(t1[k], "logit_acc") for k, _ in MODELS]
    logit_aucs = [fget(t1[k], "logit_auc") for k, _ in MODELS]

    # ── Load probe values ─────────────────────────────────────────────────────
    probe_accs     = []
    probe_aucs     = []
    probe_acc_errs = []
    probe_auc_errs = []

    if kfold:
        kf = read_csv_as_dict(DATA_DIR / "kfold_table3_probes.csv", key_col="model")
        for key, _ in MODELS:
            row = kf.get(_KFOLD_KEY[key], {})
            probe_accs.append(fget(row, "fl_lr_acc_mean"))
            probe_aucs.append(fget(row, "fl_lr_auc_mean"))
            probe_acc_errs.append(fget(row, "fl_lr_acc_ci95"))
            probe_auc_errs.append(fget(row, "fl_lr_auc_ci95"))
    else:
        for key, _ in MODELS:
            probe_row = t2[key] if key == "Base" else t3[key]
            probe_accs.append(fget(probe_row, "fl_lr_acc"))
            probe_aucs.append(fget(probe_row, "fl_lr_auc"))
        probe_acc_errs = [float("nan")] * N
        probe_auc_errs = [float("nan")] * N

    probe_accs     = np.array(probe_accs)
    probe_aucs     = np.array(probe_aucs)
    probe_acc_errs = np.array(probe_acc_errs)
    probe_auc_errs = np.array(probe_auc_errs)

    err_label = "±95 % CI (5-fold CV)" if kfold else None
    _figsize = tuple(args.figsize) if args.figsize else (13, 5.5)

    # ── Plot 1: Accuracy ──────────────────────────────────────────────────────
    width  = 0.26
    offset = [-width, 0, width]

    fig1, ax1 = plt.subplots(figsize=_figsize)

    b_gen   = ax1.bar(x + offset[0], gen_accs,   width, color=COLORS["gen"],   zorder=3)
    b_logit = ax1.bar(x + offset[1], logit_accs, width, color=COLORS["logit"], zorder=3)
    b_probe = ax1.bar(x + offset[2], probe_accs, width, color=COLORS["probe"], zorder=3)
    add_errorbars(ax1, x + offset[2], probe_accs, probe_acc_errs, width, COLORS["probe"])

    ax1.axhline(0.5, color="black", linewidth=1.2, linestyle="--", zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    format_yaxis(ax1, "Accuracy")
    title1 = "Hidden Knowledge After Unlearning — Bio (WMDP)\nSurface behaviour vs. internal hidden-state probe  [Accuracy]"
    if kfold:
        title1 += "\n(probe = 5-fold CV mean ± 95 % CI)"
    ax1.set_title(title1, fontsize=9, pad=8)

    handles1 = [
        mpatches.Patch(color=COLORS["gen"],   label="Generation acc (surface)"),
        mpatches.Patch(color=COLORS["logit"], label="Logit acc (surface)"),
        mpatches.Patch(color=COLORS["probe"], label="Probe acc — full-layer LR (internal)"),
        plt.Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Chance (50%)"),
    ]
    if err_label:
        handles1.append(plt.Line2D([0], [0], color="black", linewidth=1.2,
                                   marker="|", markersize=8, label=err_label))
    ax1.legend(handles=handles1, loc="upper right", fontsize=7, framealpha=0.9)
    add_separators(ax1, N)

    fig1.tight_layout()
    fig1.savefig(out1, dpi=300, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close(fig1)

    # ── Plot 2: AUC ───────────────────────────────────────────────────────────
    AUC_YMIN = 0.4
    width2   = 0.32
    offset2  = [-width2 / 2, width2 / 2]

    fig2, ax2 = plt.subplots(figsize=_figsize)

    b_lauc = ax2.bar(x + offset2[0], np.array(logit_aucs) - AUC_YMIN, width2,
                     bottom=AUC_YMIN, color=COLORS["logit"], zorder=3)
    b_pauc = ax2.bar(x + offset2[1], probe_aucs - AUC_YMIN, width2,
                     bottom=AUC_YMIN, color=COLORS["probe"], zorder=3)
    add_errorbars(ax2, x + offset2[1], probe_aucs, probe_auc_errs, width2, COLORS["probe"])

    ax2.axhline(0.5, color="black", linewidth=1.2, linestyle="--", zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    format_yaxis(ax2, "AUC (ROC)", ymin=0.4, ymax=0.8, step=0.05)
    title2 = "Hidden Knowledge After Unlearning — Bio (WMDP)\nSurface behaviour vs. internal hidden-state probe  [AUC]"
    if kfold:
        title2 += "\n(probe = 5-fold CV mean ± 95 % CI)"
    ax2.set_title(title2, fontsize=9, pad=8)

    handles2 = [
        mpatches.Patch(color=COLORS["logit"], label="Logit AUC (surface)"),
        mpatches.Patch(color=COLORS["probe"], label="Probe AUC — full-layer LR (internal)"),
        plt.Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Chance (50%)"),
    ]
    if err_label:
        handles2.append(plt.Line2D([0], [0], color="black", linewidth=1.2,
                                   marker="|", markersize=8, label=err_label))
    ax2.legend(handles=handles2, loc="upper right", fontsize=7, framealpha=0.9)
    add_separators(ax2, N)

    fig2.tight_layout()
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
