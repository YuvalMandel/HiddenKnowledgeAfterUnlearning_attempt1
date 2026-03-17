#!/usr/bin/env python3
"""
Plot Representational Erasure (RE) and Directional Rotation - Forward (DR_fwd)
across checkpoints and methods, computed from sweep CSVs.

Notation:
  APP_base : base probe  → base model   [summary_table2, Base row]       constant
  APP_post : method probe→ method model [sweep CSV, mp_ columns]          per ck
  ApP      : base probe  → method model [sweep CSV, bp_ columns]          per ck

  RE       = (APP_base - APP_post) / (APP_base - 0.5)
               ≈ 0: no erasure  |  ≈ 1: full erasure

  DR_fwd   = (APP_post - ApP) / (APP_post - 0.5)
               ≈ 0: base axis still tracks post model  |  ≈ 1: strong misalignment

NOTE: DR_backward (method probe → base model) is NOT stored in sweep CSVs —
it would require re-running each checkpoint's probe on base model hidden states.
DR_backward is available only at ck8 via summary_table5_cross_probes.csv.
Use plot_RE_DR.py for the full symmetric DR at ck8.

Usage examples:
  # Line plot: RE and DR_fwd per checkpoint, all methods, mb/LR
  python plot_RE_DR_sweep.py

  # Scatter RE vs DR_fwd at ck4 and ck8, ibnp/RF
  python plot_RE_DR_sweep.py --band ibnp --clf rf --checkpoints 4,8 --plot_type scatter

  # Only RE, specific methods
  python plot_RE_DR_sweep.py --methods GradDiff,RMU,ELM --metrics_plot RE
"""

import argparse
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]

_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def safe_name(n: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", n)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_app_base(data_dir: Path, band: str, clf: str, metric: str) -> float:
    """Load APP_base (base probe → base model) from summary_table2."""
    path = data_dir / "summary_table2_base_probes.csv"
    if not path.exists():
        print(f"ERROR: {path} not found.", file=sys.stderr)
        sys.exit(1)
    t2  = pd.read_csv(path, index_col=0)
    col = f"{band}_{clf}_{metric}"
    if col not in t2.columns:
        print(f"ERROR: column '{col}' not in {path}.", file=sys.stderr)
        print(f"  Available columns: {list(t2.columns)}", file=sys.stderr)
        sys.exit(1)
    val = float(t2.loc["Base", col])
    print(f"APP_base ({col}): {val:.4f}")
    return val


def load_sweep(data_dir: Path, methods: list,
               band: str, clf: str, metric: str) -> pd.DataFrame:
    """
    Load per-checkpoint APP_post and ApP from sweep CSVs.
    Returns DataFrame: method, checkpoint, APP_post, ApP
    """
    mp_col = f"mp_{clf}_{band}_{metric}"
    bp_col = f"bp_{clf}_{band}_{metric}"
    frames = []
    for method in methods:
        sn   = safe_name(method)
        path = data_dir / f"sweep_{sn}" / f"{sn}_sweep.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping '{method}'", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        for col in (mp_col, bp_col):
            if col not in df.columns:
                print(f"  WARNING: column '{col}' missing in {path} — skipping '{method}'",
                      file=sys.stderr)
                break
        else:
            frames.append(pd.DataFrame({
                "method":     method,
                "checkpoint": df["checkpoint"].astype(int),
                "APP_post":   df[mp_col].astype(float),
                "ApP":        df[bp_col].astype(float),
            }))
    if not frames:
        print("ERROR: No sweep data loaded.", file=sys.stderr)
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def compute_metrics(df: pd.DataFrame, APP_base: float) -> pd.DataFrame:
    df = df.copy()
    df["RE"] = (APP_base - df["APP_post"]) / (APP_base - 0.5)
    denom    = df["APP_post"] - 0.5
    df["DR_fwd"] = np.where(denom.abs() < 1e-6, np.nan,
                            (df["APP_post"] - df["ApP"]) / denom)
    return df


def prepend_base_ck0(df: pd.DataFrame, APP_base: float) -> pd.DataFrame:
    """Add checkpoint=0 rows for the base model (RE=0, DR_fwd=0 by definition)."""
    rows = [{"method": m, "checkpoint": 0,
             "APP_post": APP_base, "ApP": APP_base,
             "RE": 0.0, "DR_fwd": 0.0}
            for m in df["method"].unique()]
    return pd.concat([pd.DataFrame(rows), df], ignore_index=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_line(df: pd.DataFrame, methods: list, checkpoints: list,
              metrics_plot: list, title: str | None,
              out_path: Path, figsize: tuple):
    """One subplot per metric, one line per method, x-axis = checkpoint."""
    n   = len(metrics_plot)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), squeeze=False)

    for ax, met in zip(axes[0], metrics_plot):
        for i, method in enumerate(methods):
            sub = (df[(df["method"] == method) & df["checkpoint"].isin(checkpoints)]
                   .sort_values("checkpoint"))
            if sub.empty:
                continue
            ax.plot(sub["checkpoint"], sub[met],
                    marker=_MARKERS[i % len(_MARKERS)],
                    color=_COLORS[i % len(_COLORS)],
                    label=method, linewidth=1.8, markersize=5)

        ax.axhline(0, color="grey",       lw=0.8, ls="--", alpha=0.5)
        ax.axhline(1, color="steelblue",  lw=0.7, ls=":",  alpha=0.4, label="= 1")
        ax.set_xlabel("Checkpoint", fontsize=10)
        ax.set_ylabel(met, fontsize=10)
        ax.set_title(met, fontsize=11)
        xticks = sorted(df[df["checkpoint"].isin(checkpoints)]["checkpoint"].unique())
        ax.set_xticks(xticks)
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(alpha=0.25, linewidth=0.6)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, methods: list, checkpoints: list,
                 title: str | None, out_path: Path, figsize: tuple):
    """One subplot per checkpoint, RE on y, DR_fwd on x, one dot per method."""
    n   = len(checkpoints)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), squeeze=False)

    for ax, ck in zip(axes[0], checkpoints):
        sub = df[df["checkpoint"] == ck]
        for i, method in enumerate(methods):
            row = sub[sub["method"] == method]
            if row.empty:
                continue
            ax.scatter(row["DR_fwd"], row["RE"],
                       color=_COLORS[i % len(_COLORS)], s=200,
                       marker=_MARKERS[i % len(_MARKERS)],
                       zorder=5, edgecolors="k", linewidths=0.6, label=method)
            ax.annotate(method,
                        xy=(float(row["DR_fwd"]), float(row["RE"])),
                        xytext=(6, 4), textcoords="offset points", fontsize=8)

        ax.axhline(0, color="grey",       lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="grey",       lw=0.8, ls="--", alpha=0.5)
        ax.axhline(1, color="steelblue",  lw=0.7, ls=":",  alpha=0.4, label="RE=1")
        ax.axvline(1, color="darkorange", lw=0.7, ls=":",  alpha=0.4, label="DR=1")
        ax.set_xlabel("DR_fwd", fontsize=10)
        ax.set_ylabel("RE",     fontsize=10)
        ax.set_title(f"ck{ck}" if ck > 0 else "Base", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.25, linewidth=0.6)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods",      default="all",
                    help="Methods: comma-separated or 'all' (default: all)")
    ap.add_argument("--checkpoints",  default="all",
                    help="Checkpoints: comma-separated or 'all' (default: all)")
    ap.add_argument("--band",         default="mb",
                    help="Probe band: mb,ibnp,fl,ib,ibe,eb,pl (default: mb)")
    ap.add_argument("--clf",          default="lr",
                    help="Classifier: lr,rf,adaboost (default: lr)")
    ap.add_argument("--metric",       default="acc",
                    help="Metric: acc,auc,f1 (default: acc)")
    ap.add_argument("--metrics_plot", default="RE,DR_fwd",
                    help="Computed metrics to plot: RE,DR_fwd or either (default: RE,DR_fwd)")
    ap.add_argument("--plot_type",    default="line", choices=["line", "scatter"],
                    help="line: checkpoints on x, one line per method  |  "
                         "scatter: RE vs DR_fwd, one subplot per checkpoint (default: line)")
    ap.add_argument("--no_base",      action="store_true",
                    help="Do not prepend base model as checkpoint 0")
    ap.add_argument("--title",        default=None)
    ap.add_argument("--out",          default=None,
                    help="Output file (default: auto-generated)")
    ap.add_argument("--figsize",      default="8,5",
                    help="Figure size per panel W,H in inches (default: 8,5)")
    ap.add_argument("--data_dir",     default=None,
                    help="Path to data/ directory (default: ./data)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent / "data"
    figsize  = tuple(float(x) for x in args.figsize.split(","))
    methods  = (ALL_METHODS if args.methods == "all"
                else [m.strip() for m in args.methods.split(",")])
    metrics_plot = [m.strip() for m in args.metrics_plot.split(",")]

    # ── Load data ────────────────────────────────────────────────────────────
    APP_base = load_app_base(data_dir, args.band, args.clf, args.metric)
    raw      = load_sweep(data_dir, methods, args.band, args.clf, args.metric)
    df       = compute_metrics(raw, APP_base)

    avail_ck = sorted(df["checkpoint"].unique())
    checkpoints = (avail_ck if args.checkpoints == "all"
                   else [int(c) for c in args.checkpoints.split(",")])

    if not args.no_base:
        df = prepend_base_ck0(df, APP_base)
        if 0 not in checkpoints:
            checkpoints = [0] + checkpoints

    # ── Output path ──────────────────────────────────────────────────────────
    auto = (f"RE_DR_sweep_{args.band}_{args.clf}_{args.metric}"
            f"_{args.plot_type}.png")
    out_path = Path(args.out) if args.out else Path(auto)

    title = (args.title or
             f"RE / DR_fwd  |  band={args.band}  clf={args.clf}  metric={args.metric}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if args.plot_type == "line":
        plot_line(df, methods, checkpoints, metrics_plot, title, out_path, figsize)
    else:
        plot_scatter(df, methods, checkpoints, title, out_path, figsize)


if __name__ == "__main__":
    main()
