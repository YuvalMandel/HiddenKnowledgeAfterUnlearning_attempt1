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
ALL_BANDS   = ["pl", "mb", "fl", "ib", "ibe", "eb", "ibnp"]

BAND_LABELS = {
    "pl": "Per-Layer", "mb": "Mid-Band", "fl": "Full-Layer",
    "ib": "Init-Band", "ibe": "Init+Emb", "eb": "End-Band", "ibnp": "IB-NoPCA",
}

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
    Load per-checkpoint APP_post, ApP, and APp (if xp_ columns present) from sweep CSVs.
    Returns DataFrame: method, checkpoint, APP_post, ApP [, APp]
    APp = method probe → base model (xp_ columns); enables full DR = (DR_fwd + DR_bwd)/2.
    """
    mp_col = f"mp_{clf}_{band}_{metric}"
    bp_col = f"bp_{clf}_{band}_{metric}"
    xp_col = f"xp_{clf}_{band}_{metric}"
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
            row = {
                "method":     method,
                "checkpoint": df["checkpoint"].astype(int),
                "APP_post":   df[mp_col].astype(float),
                "ApP":        df[bp_col].astype(float),
            }
            if xp_col in df.columns:
                row["APp"] = df[xp_col].astype(float)
            frames.append(pd.DataFrame(row))
    if not frames:
        print("ERROR: No sweep data loaded.", file=sys.stderr)
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def compute_metrics(df: pd.DataFrame, APP_base: float) -> pd.DataFrame:
    """Compute RE, DR_fwd (and DR_bwd/DR if APp column present)."""
    df = df.copy()
    df["RE"] = (APP_base - df["APP_post"]) / (APP_base - 0.5)
    denom    = df["APP_post"] - 0.5
    df["DR_fwd"] = np.where(denom.abs() < 1e-6, np.nan,
                            (df["APP_post"] - df["ApP"]) / denom)
    if "APp" in df.columns:
        df["DR_bwd"] = (APP_base - df["APp"]) / (APP_base - 0.5)
        df["DR"]     = (df["DR_fwd"] + df["DR_bwd"]) / 2.0
    return df


def prepend_base_ck0(df: pd.DataFrame, APP_base: float) -> pd.DataFrame:
    """Add checkpoint=0 rows for the base model (RE=0, DR_fwd=0 by definition)."""
    rows = [{"method": m, "checkpoint": 0,
             "APP_post": APP_base, "ApP": APP_base,
             "RE": 0.0, "DR_fwd": 0.0}
            for m in df["method"].unique()]
    return pd.concat([pd.DataFrame(rows), df], ignore_index=True)


def load_all_bands_pipeline(data_dir: Path, methods: list, bands: list,
                            clf: str, metric: str) -> pd.DataFrame:
    """
    Load RE, DR_fwd, DR_bwd, and full DR from pipeline summary tables for multiple bands.
    t2 = summary_table2_base_probes.csv   (base probe  → base/method models)
    t3 = summary_table3_method_probes.csv (method probe → method models)
    t5 = summary_table5_cross_probes.csv  (method probe → base model)
    Returns DataFrame: method, band, checkpoint=8, RE, DR_fwd, DR_bwd, DR
    """
    t2 = pd.read_csv(data_dir / "summary_table2_base_probes.csv",   index_col=0)
    t3 = pd.read_csv(data_dir / "summary_table3_method_probes.csv", index_col=0)
    t5 = pd.read_csv(data_dir / "summary_table5_cross_probes.csv",  index_col=0)

    frames = []
    for band in bands:
        col = f"{band}_{clf}_{metric}"
        missing = [name for name, tbl in [("t2", t2), ("t3", t3), ("t5", t5)]
                   if col not in tbl.columns]
        if missing:
            print(f"  WARNING: '{col}' missing in {missing} — skipping band '{band}'",
                  file=sys.stderr)
            continue
        APP_base = float(t2.loc["Base", col])
        rows = []
        for method in methods:
            if method not in t3.index:
                continue
            APP_post = float(t3.loc[method, col])
            ApP      = float(t2.loc[method, col]) if method in t2.index else float("nan")
            APp      = float(t5.loc[method, col]) if method in t5.index else float("nan")

            RE     = (APP_base - APP_post) / (APP_base - 0.5)
            denom_fwd = APP_post - 0.5
            DR_fwd = ((APP_post - ApP) / denom_fwd
                      if abs(denom_fwd) > 1e-6 else float("nan"))
            DR_bwd = (APP_base - APp) / (APP_base - 0.5)
            DR     = (DR_fwd + DR_bwd) / 2.0

            rows.append({"method": method, "band": band, "checkpoint": 8,
                         "APP_post": APP_post, "ApP": ApP, "APp": APp,
                         "RE": RE, "DR_fwd": DR_fwd, "DR_bwd": DR_bwd, "DR": DR})
        frames.append(pd.DataFrame(rows))

    if not frames:
        print("ERROR: No pipeline data loaded for any band.", file=sys.stderr)
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def load_all_bands_sweep(data_dir: Path, methods: list, bands: list,
                         clf: str, metric: str, checkpoint: int) -> pd.DataFrame:
    """
    Load RE and DR_fwd at a single checkpoint for multiple bands from sweep CSVs.
    Returns DataFrame: method, band, RE, DR_fwd
    """
    t2_path = data_dir / "summary_table2_base_probes.csv"
    t2 = pd.read_csv(t2_path, index_col=0)

    frames = []
    for band in bands:
        base_col = f"{band}_{clf}_{metric}"
        if base_col not in t2.columns:
            print(f"  WARNING: column '{base_col}' not in summary_table2 — skipping band '{band}'",
                  file=sys.stderr)
            continue
        APP_base = float(t2.loc["Base", base_col])
        raw = load_sweep(data_dir, methods, band, clf, metric)
        raw = raw[raw["checkpoint"] == checkpoint]
        if raw.empty:
            continue
        df = compute_metrics(raw, APP_base)
        df["band"] = band
        frames.append(df)

    if not frames:
        print("ERROR: No sweep data loaded for any band.", file=sys.stderr)
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_line(df: pd.DataFrame, methods: list, checkpoints: list,
              metrics_plot: list, title: str | None,
              out_path: Path, figsize: tuple, hue_col: str = "method"):
    """
    One subplot per metric, x-axis = checkpoint.
    hue_col="method" : one line per method (default)
    hue_col="band"   : one line per probe band (averaged across methods)
    """
    n   = len(metrics_plot)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), squeeze=False)

    if hue_col == "band" and "band" in df.columns:
        # preserve canonical band ordering
        hue_vals  = [b for b in ALL_BANDS if b in df["band"].unique()]
        label_fn  = lambda v: BAND_LABELS.get(v, v)
    else:
        hue_vals  = methods
        label_fn  = lambda v: v

    for ax, met in zip(axes[0], metrics_plot):
        if met not in df.columns:
            ax.set_title(f"{met}  (not available)", fontsize=11)
            continue
        for i, hval in enumerate(hue_vals):
            if hue_col == "band" and "band" in df.columns:
                sub = df[(df["band"] == hval) & df["checkpoint"].isin(checkpoints)]
                # average across methods per checkpoint
                sub = (sub.groupby("checkpoint", as_index=False)[met]
                          .mean().sort_values("checkpoint"))
            else:
                sub = (df[(df["method"] == hval) & df["checkpoint"].isin(checkpoints)]
                       .sort_values("checkpoint"))
            if sub.empty:
                continue
            ax.plot(sub["checkpoint"], sub[met],
                    marker=_MARKERS[i % len(_MARKERS)],
                    color=_COLORS[i % len(_COLORS)],
                    label=label_fn(hval), linewidth=1.8, markersize=5)

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


def plot_bar(df: pd.DataFrame, methods: list, bands: list,
             metrics_plot: list, title: str | None,
             out_path: Path, figsize: tuple):
    """
    Bar chart: x-axis = probe band, grouped bars per method, one subplot per metric.
    df must have columns: method, band, RE, DR_fwd
    """
    n   = len(metrics_plot)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), squeeze=False)

    for ax, met in zip(axes[0], metrics_plot):
        avail_bands = [b for b in bands if b in df["band"].unique()]
        n_b = len(avail_bands)
        n_m = len(methods)
        bar_w = min(0.8 / n_m, 0.3)
        x     = np.arange(n_b)

        for i, method in enumerate(methods):
            sub  = df[df["method"] == method]
            vals = []
            for band in avail_bands:
                row = sub[sub["band"] == band][met]
                vals.append(float(row.iloc[0]) if len(row) > 0 else float("nan"))
            offset = (i - n_m / 2 + 0.5) * bar_w
            ax.bar(x + offset, vals, bar_w * 0.9,
                   label=method, color=_COLORS[i % len(_COLORS)],
                   alpha=0.85, edgecolor="white", linewidth=0.5)

        ax.axhline(0, color="grey",      lw=0.8, ls="--", alpha=0.5)
        ax.axhline(1, color="steelblue", lw=0.7, ls=":",  alpha=0.4, label="= 1")
        ax.set_xticks(x)
        ax.set_xticklabels([BAND_LABELS.get(b, b) for b in avail_bands],
                           rotation=20, ha="right", fontsize=9)
        ax.set_xlabel("Probe Band", fontsize=10)
        ax.set_ylabel(met, fontsize=10)
        ax.set_title(met, fontsize=11)
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

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
    ap.add_argument("--metrics_plot", default="RE,DR",
                    help="Computed metrics to plot: any of RE,DR,DR_fwd,DR_bwd "
                         "(default: RE,DR). DR requires pipeline tables; "
                         "sweep/line mode only has DR_fwd)")
    ap.add_argument("--plot_type",    default="line", choices=["line", "scatter", "bar"],
                    help="line: checkpoints on x, one line per method  |  "
                         "scatter: RE vs DR_fwd, one subplot per checkpoint  |  "
                         "bar: probe bands on x, grouped by method (default: line)")
    ap.add_argument("--bands",        default=None,
                    help="Probe bands: comma-separated or 'all'. "
                         "In bar mode: x-axis bands (default: all). "
                         "In line mode: one colored line per band, averaged across methods.")
    ap.add_argument("--sweep",        action="store_true",
                    help="Bar mode: load from sweep CSVs instead of pipeline tables")
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

    # ── Bar mode (multi-band) ────────────────────────────────────────────────
    if args.plot_type == "bar":
        bands = (ALL_BANDS if (args.bands or "all") == "all"
                 else [b.strip() for b in args.bands.split(",")])
        if args.sweep:
            ck   = (8 if args.checkpoints == "all"
                    else int(args.checkpoints.split(",")[-1]))
            df   = load_all_bands_sweep(data_dir, methods, bands, args.clf, args.metric, ck)
            src  = f"sweep-ck{ck}"
        else:
            df   = load_all_bands_pipeline(data_dir, methods, bands, args.clf, args.metric)
            ck   = 8
            src  = "pipeline-ck8"
        auto  = f"RE_DR_bar_{src}_{args.clf}_{args.metric}.png"
        out_path = Path(args.out) if args.out else Path(auto)
        title = (args.title or
                 f"RE / DR_fwd by Probe Band  |  {src}  clf={args.clf}  metric={args.metric}")
        plot_bar(df, methods, bands, metrics_plot, title, out_path, figsize)
        return

    # ── Multi-band line mode (--bands given for line/scatter) ─────────────────
    if args.bands and args.plot_type in ("line", "scatter"):
        bands = (ALL_BANDS if args.bands == "all"
                 else [b.strip() for b in args.bands.split(",")])
        frames = []
        for band in bands:
            try:
                APP_base_b = load_app_base(data_dir, band, args.clf, args.metric)
            except SystemExit:
                continue
            try:
                raw_b = load_sweep(data_dir, methods, band, args.clf, args.metric)
            except SystemExit:
                continue
            df_b = compute_metrics(raw_b, APP_base_b)
            df_b["band"] = band
            if not args.no_base:
                base_rows = [{"method": m, "checkpoint": 0, "band": band,
                              "APP_post": APP_base_b, "ApP": APP_base_b,
                              "RE": 0.0, "DR_fwd": 0.0}
                             for m in df_b["method"].unique()]
                df_b = pd.concat([pd.DataFrame(base_rows), df_b], ignore_index=True)
            frames.append(df_b)
        if not frames:
            print("ERROR: no data loaded for any band.", file=sys.stderr)
            sys.exit(1)
        df = pd.concat(frames, ignore_index=True)

        # warn if DR/DR_bwd requested but xp_ columns absent (old sweep CSVs)
        for met in metrics_plot:
            if met not in df.columns:
                print(f"  WARNING: metric '{met}' not in sweep data. "
                      f"DR/DR_bwd require xp_ columns — re-run sweep to populate them.",
                      file=sys.stderr)

        avail_ck = sorted(df["checkpoint"].unique())
        checkpoints = (avail_ck if args.checkpoints == "all"
                       else [int(c) for c in args.checkpoints.split(",")])
        if not args.no_base and 0 not in checkpoints:
            checkpoints = [0] + checkpoints

        bands_str = "_".join(bands)
        auto = (f"RE_DR_sweep_bands_{bands_str}_{args.clf}_{args.metric}"
                f"_{args.plot_type}.png")
        out_path = Path(args.out) if args.out else Path(auto)
        title = (args.title or
                 f"RE / DR_fwd by Band  |  methods={','.join(methods)}"
                 f"  clf={args.clf}  metric={args.metric}")
        if args.plot_type == "line":
            plot_line(df, methods, checkpoints, metrics_plot, title, out_path, figsize,
                      hue_col="band")
        else:
            plot_scatter(df, methods, checkpoints, title, out_path, figsize)
        return

    # ── Line / scatter mode (single band) ────────────────────────────────────
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
