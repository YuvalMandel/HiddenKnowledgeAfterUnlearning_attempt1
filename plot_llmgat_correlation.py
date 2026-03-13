#!/usr/bin/env python3
"""
plot_llmgat_correlation.py

Scatter plot correlating LLM-GAT attack robustness scores with a
"hidden knowledge" gap score derived from our probe / generation results.

Hidden knowledge  =  probe_score  −  baseline_col
  (positive ⟹ the model retains more knowledge in hidden states than it
   reveals through generation / logit scoring)

Two plot modes (--plot_y):
  scores (default) — two side-by-side subplots:
                       Left  (blue) = "WMDP, Best Input Attack"
                       Right (red)  = "WMDP, Best Tamp. Attack"
  gap              — single plot; Y = |Tamp. Attack − Input Attack|

Probe selection (--probe_type, --probe_clf, --probe_metric):
  type   : pl (per-layer best) | ml (multi-layer) | vote | avg
  clf    : LR | RF | AdaBoost
  metric : acc | true | false | f1 | auc | prec | rec

Usage:
    python plot_llmgat_correlation.py
    python plot_llmgat_correlation.py --probe_type pl --probe_clf RF --probe_metric f1
    python plot_llmgat_correlation.py --plot_y gap
    python plot_llmgat_correlation.py --out my_plot.png
"""

import argparse
import csv
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR          = Path("data")
PROBE_CSV_DEFAULT = "summary_table3_method_probes.csv"

METHOD_MAP = {
    "Grad Diff":  "GradDiff",
    "RMU":        "RMU",
    "RMU + LAT":  "RMU-LAT",
    "RepNoise":   "RepNoise",
    "ELM":        "ELM",
    "RR":         "RR",
    "TAR":        "TAR",
    "PB&J":       "PB&J",
}

SHORT_NAME = {
    "Grad Diff":  "GradDiff",
    "RMU":        "RMU",
    "RMU + LAT":  "RMU-LAT",
    "RepNoise":   "RepNoise",
    "ELM":        "ELM",
    "RR":         "RR",
    "TAR":        "TAR",
    "PB&J":       "PB&J",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv_as_dict(path: Path, key_col: str = "method") -> dict:
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row[key_col]] = row
    return result


def probe_col_name(probe_type: str, probe_clf: str, probe_metric: str) -> str:
    clf    = probe_clf.lower().replace("adaboost", "ada")
    metric = probe_metric.lower()
    if metric == "false":
        metric = "fals"
    return f"{probe_type}_{clf}_{metric}"


def get_probe_val(row: dict, col: str) -> float:
    try:
        return float(row.get(col) or "nan")
    except (ValueError, TypeError):
        return float("nan")


def get_baseline(row: dict, preferred_col: str) -> float:
    try:
        v = float(row.get(preferred_col) or "nan")
    except ValueError:
        v = float("nan")
    if np.isnan(v) and preferred_col != "gen_acc":
        try:
            v = float(row.get("gen_acc") or "nan")
        except ValueError:
            v = float("nan")
    return v


def annotate_r(ax, x, y, color, label, y_offset):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return
    r, p = stats.pearsonr(x[mask], y[mask])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    ax.annotate(
        f"{label}: r = {r:.2f}{sig}  (p = {p:.3f})",
        xy=(0.03, y_offset), xycoords="axes fraction",
        fontsize=8.5, color=color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85),
    )


def fit_line(ax, x, y, color):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return
    m, b, *_ = stats.linregress(x[mask], y[mask])
    xr = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xr, m * xr + b, color=color, linewidth=1.4, linestyle="--", alpha=0.6, zorder=1)


def scatter_with_labels(ax, x, y, short, color):
    for xi, yi, name in zip(x, y, short):
        if np.isnan(xi) or np.isnan(yi):
            continue
        ax.scatter(xi, yi, color=color, s=80, zorder=3,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(name, xy=(xi, yi), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, color="#222222")


def style_ax(ax, x, y, xlabel, ylabel):
    valid_x = x[~np.isnan(x)]
    valid_y = y[~np.isnan(y)]
    if len(valid_x) and len(valid_y):
        pad_x = max((valid_x.max() - valid_x.min()) * 0.15, 0.02)
        pad_y = max((valid_y.max() - valid_y.min()) * 0.15, 0.02)
        ax.set_xlim(valid_x.min() - pad_x, valid_x.max() + pad_x)
        ax.set_ylim(valid_y.min() - pad_y, valid_y.max() + pad_y)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir",     default=str(DATA_DIR))
    parser.add_argument("--probe_table",  default=PROBE_CSV_DEFAULT,
                        help="Probe summary CSV for unlearned methods "
                             "(default: summary_table3_method_probes.csv)")
    parser.add_argument("--probe_type",   default="ml",
                        choices=["pl", "ml", "vote", "avg"],
                        help="Probe type (default: ml)")
    parser.add_argument("--probe_clf",    default="LR",
                        choices=["LR", "RF", "AdaBoost"],
                        help="Classifier (default: LR)")
    parser.add_argument("--probe_metric", default="auc",
                        choices=["acc", "true", "false", "f1", "auc", "prec", "rec"],
                        help="Probe metric (default: auc)")
    parser.add_argument("--gen_col",      default="logit_auc",
                        help="Baseline score column from table1 subtracted from probe score "
                             "(default: logit_auc, falls back to gen_acc if empty)")
    parser.add_argument("--plot_y",       default="scores",
                        choices=["gap", "scores"],
                        help="Y-axis: 'scores' (default) = two subplots, one per attack; "
                             "'gap' = single plot with |Tamp−Input|")
    parser.add_argument("--out",          default=None)
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    llmgat_path = data_dir / "LLM-GAT_summary_table.csv"
    gen_path    = data_dir / "summary_table1_gen_logit.csv"
    probe_path  = data_dir / args.probe_table

    for p in [llmgat_path, gen_path, probe_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    pcol = probe_col_name(args.probe_type, args.probe_clf, args.probe_metric)
    print(f"  Probe column  : {pcol}")
    print(f"  Baseline col  : {args.gen_col} (fallback: gen_acc)")
    print(f"  Probe table   : {args.probe_table}")

    llmgat_rows = load_csv_as_dict(llmgat_path, key_col="Method")
    gen_rows    = load_csv_as_dict(gen_path,    key_col="method")
    probe_rows  = load_csv_as_dict(probe_path,  key_col="method")

    methods, hk_scores, ia_scores, ta_scores = [], [], [], []

    for llmgat_name, our_name in METHOD_MAP.items():
        if llmgat_name not in llmgat_rows:
            continue
        lg  = llmgat_rows[llmgat_name]
        gen = gen_rows.get(our_name)
        prb = probe_rows.get(our_name)

        if gen is None:
            print(f"  [skip] {llmgat_name}: missing gen row in table1")
            continue
        if prb is None:
            print(f"  [skip] {llmgat_name}: missing probe row in {args.probe_table}")
            continue

        baseline  = get_baseline(gen, args.gen_col)
        probe_val = get_probe_val(prb, pcol)
        hk        = probe_val - baseline

        try:
            ia = float(lg.get("WMDP, Best Input Attack") or "nan")
            ta = float(lg.get("WMDP, Best Tamp. Attack") or "nan")
        except ValueError:
            ia = ta = float("nan")

        methods.append(llmgat_name)
        hk_scores.append(hk)
        ia_scores.append(ia)
        ta_scores.append(ta)

        print(f"  {llmgat_name:20s}  baseline={baseline:.3f}  probe={probe_val:.3f}"
              f"  hk={hk:+.3f}  input={ia:.3f}  tamp={ta:.3f}")

    if not methods:
        raise RuntimeError("No methods matched.")

    hk    = np.array(hk_scores, dtype=float)
    ia    = np.array(ia_scores,  dtype=float)
    ta    = np.array(ta_scores,  dtype=float)
    short = [SHORT_NAME[n] for n in methods]

    probe_lbl = f"{args.probe_type.upper()} / {args.probe_clf} / {args.probe_metric}"
    xlabel    = f"Hidden Knowledge Gap  (probe {args.probe_metric} − {args.gen_col})"
    tag       = f"{args.probe_type}_{args.probe_clf}_{args.probe_metric}"
    probe_stem = Path(args.probe_table).stem.replace("summary_", "")

    # -----------------------------------------------------------------------
    # scores mode — two side-by-side subplots
    # -----------------------------------------------------------------------
    if args.plot_y == "scores":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"WMDP Attack Score vs. Hidden Knowledge Gap\n"
            f"probe: {probe_lbl}   |   hidden knowledge = probe score − {args.gen_col}",
            fontsize=11,
        )

        C_INPUT = "#1f77b4"   # blue
        C_TAMP  = "#d62728"   # red

        for ax, y_arr, color, atk_lbl in [
            (axes[0], ia, C_INPUT, "Best Input Attack"),
            (axes[1], ta, C_TAMP,  "Best Tamp. Attack"),
        ]:
            fit_line(ax, hk, y_arr, color)
            scatter_with_labels(ax, hk, y_arr, short, color)
            annotate_r(ax, hk, y_arr, color, atk_lbl, 0.93)
            style_ax(ax, hk, y_arr, xlabel, f"WMDP {atk_lbl}")
            ax.set_title(f"WMDP, {atk_lbl}", fontsize=10)

        plt.tight_layout()
        out = Path(args.out) if args.out else (
            data_dir / f"correlation_hk_vs_attack_scores_{probe_stem}_{tag}.png"
        )

    # -----------------------------------------------------------------------
    # gap mode — single plot
    # -----------------------------------------------------------------------
    else:
        delta = np.abs(ta - ia)
        COLOR = "#2ca02c"   # green

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(
            f"Hidden Knowledge Gap vs. |Attack Vulnerability Gap|\n"
            f"probe: {probe_lbl}   |   hidden knowledge = probe score − {args.gen_col}",
            fontsize=11,
        )

        fit_line(ax, hk, delta, COLOR)
        scatter_with_labels(ax, hk, delta, short, COLOR)
        annotate_r(ax, hk, delta, COLOR, "|Tamp. − Input|", 0.93)
        style_ax(ax, hk, delta, xlabel,
                 "|Attack Vulnerability Gap|  |Tamp. Attack − Input Attack|")

        plt.tight_layout()
        out = Path(args.out) if args.out else (
            data_dir / f"correlation_hk_vs_attack_gap_{probe_stem}_{tag}.png"
        )

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
