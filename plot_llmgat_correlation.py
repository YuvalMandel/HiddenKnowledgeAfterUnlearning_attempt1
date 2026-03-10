#!/usr/bin/env python3
"""
plot_llmgat_correlation.py

Scatter plots correlating LLM-GAT attack robustness scores with a
"hidden knowledge" gap score derived from our probe / generation results.

Hidden knowledge  =  best_probe_acc  −  gen_valid_acc
  (positive ⟹ the model retains more knowledge in hidden states than it
   reveals through generation; negative ⟹ model appears to know less
   internally than it shows externally)

Two y-axes (one subplot each):
  Left  — "WMDP, Best Input Attack"
  Right — "WMDP, Best Tamp. Attack"

Usage:
    python plot_llmgat_correlation.py
    python plot_llmgat_correlation.py --data_dir data --checkpoint_dir checkpoints
    python plot_llmgat_correlation.py --probe_table summary_table2_base_probes.csv
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

DATA_DIR       = Path("data")
LLMGAT_CSV     = DATA_DIR / "LLM-GAT_summary_table.csv"
GEN_CSV        = DATA_DIR / "summary_table1_gen_logit.csv"
# Default probe table: method probes (Table 3).  Override with --probe_table.
PROBE_CSV_DEFAULT = "summary_table3_method_probes.csv"

# Mapping: LLM-GAT method name → our summary CSV method name
METHOD_MAP = {
    "Grad Diff":          "GradDiff",
    "RMU":                "RMU",
    "RMU + LAT":          "RMU-LAT",
    "RepNoise":           "RepNoise",
    "ELM":                "ELM",
    "RR":                 "RR",
    "TAR":                "TAR",
    "PB&J":               "PB&J",
    "Llama3 8B Instruct": "Base",
}

# Probe accuracy columns to consider for "best probe acc"
# (per-layer best, multi-layer, vote ensemble, avg ensemble — all CLFs)
PROBE_ACC_COLS = (
    [f"pl_{c}_acc"   for c in ("lr", "rf", "ada")]
    + [f"ml_{c}_acc"   for c in ("lr", "rf", "ada")]
    + [f"vote_{c}_acc" for c in ("lr", "rf", "ada")]
    + [f"avg_{c}_acc"  for c in ("lr", "rf", "ada")]
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv_as_dict(path: Path, key_col: str = "method") -> dict:
    """Return {row[key_col]: {col: value, ...}, ...}."""
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row[key_col]] = row
    return result


def best_probe_acc(row: dict) -> float:
    """Max numeric value across all probe accuracy columns in a CSV row."""
    vals = []
    for col in PROBE_ACC_COLS:
        v = row.get(col, "")
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            pass
    return max(vals) if vals else float("nan")


def annotate_r(ax, x, y):
    """Add Pearson r and p-value annotation to an axis."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return
    r, p = stats.pearsonr(x[mask], y[mask])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    ax.annotate(
        f"r = {r:.2f}{sig}  (p = {p:.3f})",
        xy=(0.05, 0.93), xycoords="axes fraction",
        fontsize=9, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8),
    )


def fit_line(ax, x, y, color):
    """Draw OLS regression line, ignoring NaNs."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return
    m, b, *_ = stats.linregress(x[mask], y[mask])
    xr = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xr, m * xr + b, color=color, linewidth=1.2, linestyle="--", alpha=0.6, zorder=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir",     default=str(DATA_DIR))
    parser.add_argument("--probe_table",  default=PROBE_CSV_DEFAULT,
                        help="Probe summary CSV filename inside data_dir "
                             "(default: summary_table3_method_probes.csv)")
    parser.add_argument("--gen_col",      default="gen_valid_acc",
                        help="Generation accuracy column (default: gen_valid_acc)")
    parser.add_argument("--out",          default=None,
                        help="Output PNG path (default: auto-generated)")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    llmgat    = data_dir / "LLM-GAT_summary_table.csv"
    gen_csv   = data_dir / "summary_table1_gen_logit.csv"
    probe_csv = data_dir / args.probe_table

    for p in [llmgat, gen_csv, probe_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # Load tables
    llmgat_rows = load_csv_as_dict(llmgat, key_col="Method")
    gen_rows    = load_csv_as_dict(gen_csv, key_col="method")
    probe_rows  = load_csv_as_dict(probe_csv, key_col="method")

    # Build per-method data
    methods, hk_scores, input_att, tamp_att = [], [], [], []

    for llmgat_name, our_name in METHOD_MAP.items():
        if llmgat_name not in llmgat_rows:
            continue
        lg  = llmgat_rows[llmgat_name]
        gen = gen_rows.get(our_name)
        prb = probe_rows.get(our_name)

        if gen is None or prb is None:
            print(f"  [skip] {llmgat_name}: missing gen or probe row in summary CSVs")
            continue

        try:
            gen_acc   = float(gen.get(args.gen_col) or "nan")
        except ValueError:
            gen_acc   = float("nan")
        probe_acc = best_probe_acc(prb)
        hk        = probe_acc - gen_acc       # hidden knowledge gap

        try:
            ia = float(lg.get("WMDP, Best Input Attack") or "nan")
            ta = float(lg.get("WMDP, Best Tamp. Attack") or "nan")
        except ValueError:
            ia, ta = float("nan"), float("nan")

        methods.append(llmgat_name)
        hk_scores.append(hk)
        input_att.append(ia)
        tamp_att.append(ta)

        print(f"  {llmgat_name:25s}  gen={gen_acc:.3f}  probe={probe_acc:.3f}"
              f"  hk={hk:+.3f}  input_att={ia:.2f}  tamp_att={ta:.2f}")

    if not methods:
        raise RuntimeError("No methods matched — check METHOD_MAP and CSV contents.")

    hk  = np.array(hk_scores, dtype=float)
    ia  = np.array(input_att,  dtype=float)
    ta  = np.array(tamp_att,   dtype=float)

    # Shorten display names
    short = [n.replace("Llama3 8B Instruct", "Base").replace("RMU + LAT", "RMU-LAT")
             for n in methods]

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    COLOR_IA = "#1f77b4"   # blue  — input attack
    COLOR_TA = "#d62728"   # red   — tampered attack

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Hidden Knowledge Gap vs. LLM-GAT Attack Robustness\n"
        "(hidden knowledge = best probe acc. − gen. acc.)",
        fontsize=12,
    )

    for ax, y, color, y_label in [
        (axes[0], ia, COLOR_IA, "WMDP, Best Input Attack"),
        (axes[1], ta, COLOR_TA, "WMDP, Best Tamp. Attack"),
    ]:
        fit_line(ax, hk, y, color)

        for xi, yi, name in zip(hk, y, short):
            if np.isnan(xi) or np.isnan(yi):
                continue
            ax.scatter(xi, yi, color=color, s=80, zorder=3, edgecolors="white", linewidths=0.6)
            ax.annotate(
                name,
                xy=(xi, yi),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color="#222222",
            )

        annotate_r(ax, hk, y)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)
        ax.set_xlabel("Hidden Knowledge Gap  (probe acc. − gen. acc.)", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(y_label, fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    plt.tight_layout()

    probe_stem = Path(args.probe_table).stem.replace("summary_", "")
    out = Path(args.out) if args.out else (
        data_dir / f"correlation_hk_vs_attacks_{probe_stem}_{args.gen_col}.png"
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
