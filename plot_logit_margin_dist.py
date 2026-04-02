#!/usr/bin/env python3
"""
plot_logit_margin_dist.py

Visualise the distribution of the bio logit margin
  tf_margin = true_logit - false_logit
for the base model, chosen unlearning methods, and their sweep checkpoints 1-8.

Data sources
------------
  Base model : checkpoints/base_bio_logit_test.csv  (columns incl. tf_margin)
  Sweep ck   : checkpoints/sweep_{safe_method}/ck{N}/partial.json
                 key "logit_scores" → list of [true_logit, false_logit] rows
  Final ck   : checkpoints/{safe_method}_bio_logit_test.csv
                 (only loaded when --final is passed)

Usage examples
--------------
  # All 8 methods × all 8 sweep checkpoints, KDE:
  python plot_logit_margin_dist.py

  # Only ELM and RMU, checkpoints 1 4 8:
  python plot_logit_margin_dist.py --methods ELM RMU --checkpoints 1 4 8

  # Faceted (one sub-panel per method, base shown on each):
  python plot_logit_margin_dist.py --facet

  # Include the "final checkpoint" CSV alongside sweep checkpoints:
  python plot_logit_margin_dist.py --final --methods ELM

  # Violin plot (one violin per method, stacked checkpoints):
  python plot_logit_margin_dist.py --plot_type violin

  # Histogram instead of KDE:
  python plot_logit_margin_dist.py --plot_type hist --methods GradDiff ELM
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]
N_CHECKPOINTS = 8

# One visually distinct base colour per method (tab10 palette extended).
_METHOD_BASE_COLORS = {
    "GradDiff": "#1f77b4",
    "RMU":      "#ff7f0e",
    "RMU-LAT":  "#2ca02c",
    "RepNoise": "#d62728",
    "ELM":      "#9467bd",
    "RR":       "#8c564b",
    "TAR":      "#e377c2",
    "PB&J":     "#7f7f7f",
}

BASE_COLOR   = "#000000"   # base model — black
FINAL_LSTYLE = "--"        # line style for final-checkpoint curves


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_name(method: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", method)


def _blend(hex_color: str, white_frac: float) -> str:
    """Blend a hex colour toward white by `white_frac` ∈ [0, 1]."""
    r, g, b = mcolors.to_rgb(hex_color)
    r2 = r + (1 - r) * white_frac
    g2 = g + (1 - g) * white_frac
    b2 = b + (1 - b) * white_frac
    return mcolors.to_hex((r2, g2, b2))


def _ck_colors(base_hex: str, n: int) -> list:
    """Return n colours for checkpoints 1..n: ck1 darkest → ck_n lightest."""
    # ck1 = 0% white (full colour), ck_n = 60% white
    return [_blend(base_hex, 0.60 * (i / max(n - 1, 1))) for i in range(n)]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_base(data_dir: Path):
    """Load tf_margin array from base_bio_logit_test.csv."""
    csv_path = data_dir / "base_bio_logit_test.csv"
    if not csv_path.exists():
        print(f"[warn] Base logit CSV not found: {csv_path}", file=sys.stderr)
        return None
    import csv
    margins = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                margins.append(float(row["tf_margin"]))
            except (KeyError, ValueError):
                pass
    if not margins:
        print(f"[warn] Base logit CSV is empty: {csv_path}", file=sys.stderr)
        return None
    return np.array(margins)


def load_sweep_ck(data_dir: Path, method: str, ck: int):
    """Load tf_margin from sweep partial.json for (method, ck)."""
    partial_path = data_dir / f"sweep_{safe_name(method)}" / f"ck{ck}" / "partial.json"
    if not partial_path.exists():
        return None
    with open(partial_path) as f:
        partial = json.load(f)
    scores = partial.get("logit_scores")
    if not scores:
        return None
    margins = [float(row[0]) - float(row[1]) for row in scores]
    return np.array(margins)


def load_final_ck(data_dir: Path, method: str):
    """Load tf_margin from {sn}_bio_logit_test.csv (final-checkpoint CSV)."""
    import csv
    sn = safe_name(method)
    csv_path = data_dir / f"{sn}_bio_logit_test.csv"
    if not csv_path.exists():
        return None
    margins = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                margins.append(float(row["tf_margin"]))
            except (KeyError, ValueError):
                pass
    return np.array(margins) if margins else None


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _kde_plot(ax, margins, color, lw, ls, alpha, label, bw_adjust=1.0, xmin=None):
    """Draw a KDE curve onto ax."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(margins, bw_method="scott")
    kde.set_bandwidth(kde.factor * bw_adjust)
    x_lo = margins.min() - 0.5
    if xmin is not None:
        x_lo = max(x_lo, xmin)
    xs = np.linspace(x_lo, margins.max() + 0.5, 512)
    ax.plot(xs, kde(xs), color=color, lw=lw, ls=ls, alpha=alpha, label=label)


def _hist_plot(ax, margins, color, alpha, label, bins=60):
    ax.hist(margins, bins=bins, density=True,
            color=color, alpha=max(alpha * 0.7, 0.25), label=label,
            histtype="stepfilled", linewidth=0)


def _violin_pos(method_idx, ck_idx, n_methods, n_cks):
    """Compute x-position for a violin: methods separated, checkpoints clustered."""
    spacing = n_cks + 1.5
    return method_idx * spacing + ck_idx


# ── Single-axes plot ──────────────────────────────────────────────────────────

def make_plot_single(
    data: dict,           # {label: (margins_array, color, lw, ls, alpha)}
    plot_type: str,
    out_path: Path,
    bw_adjust: float,
    title: str,
    xlabel: str = "Logit margin  (true − false)",
    xmin: float = None,
):
    """All distributions on one axes."""
    fig, ax = plt.subplots(figsize=(12, 5))

    if plot_type == "violin":
        # Violin needs different layout — fall back to grouped violins
        _make_violin_axes(ax, data, xlabel=xlabel)
    else:
        for label, (margins, color, lw, ls, alpha) in data.items():
            if plot_type == "kde":
                _kde_plot(ax, margins, color, lw, ls, alpha, label, bw_adjust, xmin=xmin)
            else:
                _hist_plot(ax, margins, color, alpha, label)

    if xmin is None:
        ax.axvline(0, color="gray", lw=0.8, ls=":")
    else:
        ax.set_xlim(left=xmin)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13)

    _add_legend(ax, n_cols=max(1, len(data) // 20))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def _make_violin_axes(ax, data: dict, xlabel: str = "Logit margin  (true − false)"):
    """Draw violin plots: x = index, grouped by series order in data."""
    positions = list(range(len(data)))
    parts = ax.violinplot(
        [v[0] for v in data.values()],
        positions=positions,
        showmedians=True,
        widths=0.8,
    )
    colors = [v[1] for v in data.values()]
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    parts["cbars"].set_color("black")
    parts["cmaxes"].set_color("black")
    parts["cmins"].set_color("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(list(data.keys()), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(xlabel)


def _add_legend(ax, n_cols=1):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels, fontsize=6.5, ncol=n_cols,
              loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, framealpha=0.8)


# ── Faceted plot (one panel per method) ───────────────────────────────────────

def make_plot_facet(
    base_margins,
    method_data: dict,    # {method: {ck_label: (margins, color, lw, ls, alpha)}}
    plot_type: str,
    out_path: Path,
    bw_adjust: float,
    xlabel: str = "Logit margin  (true − false)",
    xmin: float = None,
):
    methods = list(method_data.keys())
    n = len(methods)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              sharex=False, sharey=False)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax_idx, method in enumerate(methods):
        ax = axes_flat[ax_idx]
        # Draw base reference
        if base_margins is not None:
            if plot_type == "kde":
                _kde_plot(ax, base_margins, BASE_COLOR, lw=2.0, ls="-",
                          alpha=0.9, label="Base", bw_adjust=bw_adjust, xmin=xmin)
            elif plot_type == "hist":
                _hist_plot(ax, base_margins, BASE_COLOR, alpha=0.5, label="Base")
            elif plot_type == "violin":
                pass  # violin facet not implemented; silently skip
        # Draw each checkpoint
        for ck_label, (margins, color, lw, ls, alpha) in method_data[method].items():
            if plot_type == "kde":
                _kde_plot(ax, margins, color, lw, ls, alpha, ck_label, bw_adjust, xmin=xmin)
            elif plot_type == "hist":
                _hist_plot(ax, margins, color, alpha, ck_label)
        if xmin is None:
            ax.axvline(0, color="gray", lw=0.8, ls=":")
        else:
            ax.set_xlim(left=xmin)
        ax.set_title(method, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        _add_legend(ax, n_cols=1)

    # Hide empty panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Bio logit margin distributions (KDE) — per method", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot bio logit margin distributions across methods and checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        metavar="M",
        help=f"Methods to include (default: all). Choices: {', '.join(ALL_METHODS)}",
    )
    parser.add_argument(
        "--checkpoints", nargs="+", type=int, default=None,
        metavar="N",
        help="Sweep checkpoint indices to include (1-8, default: all).",
    )
    parser.add_argument(
        "--final", action="store_true",
        help="Also include the 'final checkpoint' CSV for each method "
             "(checkpoints/{safe_method}_bio_logit_test.csv).",
    )
    parser.add_argument(
        "--abs", action="store_true",
        help="Plot |true_logit - false_logit| (confidence magnitude) instead of the signed margin.",
    )
    parser.add_argument(
        "--no_base", action="store_true",
        help="Omit the base model distribution.",
    )
    parser.add_argument(
        "--plot_type", choices=["kde", "hist", "violin"], default="kde",
        help="Plot type (default: kde).",
    )
    parser.add_argument(
        "--bw_adjust", type=float, default=1.0,
        help="KDE bandwidth multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--facet", action="store_true",
        help="One sub-panel per method (base shown on each).",
    )
    parser.add_argument(
        "--data_dir", default="checkpoints",
        help="Root directory for checkpoints (default: checkpoints/).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output PNG path (auto-generated if not set).",
    )
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    methods     = args.methods if args.methods else ALL_METHODS
    checkpoints = args.checkpoints if args.checkpoints else list(range(1, N_CHECKPOINTS + 1))

    # Validate
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        parser.error(f"Unknown method(s): {bad}. Choose from: {ALL_METHODS}")
    bad_cks = [c for c in checkpoints if not (1 <= c <= N_CHECKPOINTS)]
    if bad_cks:
        parser.error(f"Checkpoint indices out of range 1-{N_CHECKPOINTS}: {bad_cks}")

    # Auto output name
    if args.out is None:
        mstr  = "all" if len(methods) == len(ALL_METHODS) else "_".join(safe_name(m) for m in methods)
        ckstr = "all" if checkpoints == list(range(1, N_CHECKPOINTS + 1)) else "_".join(str(c) for c in checkpoints)
        abs_tag = "_abs" if args.abs else ""
        suffix = f"_{args.plot_type}{abs_tag}" + ("_facet" if args.facet else "")
        out_path = Path(f"plots/logit_margin_dist_m{mstr}_ck{ckstr}{suffix}.png")
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    margin_transform = np.abs if args.abs else (lambda x: x)
    base_margins = None if args.no_base else load_base(data_dir)
    if base_margins is not None:
        base_margins = margin_transform(base_margins)

    # method_data[method][ck_label] = (margins, color, lw, ls, alpha)
    method_data: dict = {}
    n_cks = len(checkpoints)
    for method in methods:
        base_hex = _METHOD_BASE_COLORS.get(method, "#333333")
        ck_colors = _ck_colors(base_hex, n_cks)
        entries = {}

        for ci, ck in enumerate(sorted(checkpoints)):
            margins = load_sweep_ck(data_dir, method, ck)
            if margins is None:
                print(f"[skip] {method} ck{ck}: no data", file=sys.stderr)
                continue
            margins = margin_transform(margins)
            col   = ck_colors[ci]
            lw    = 1.2 + 0.4 * (ci / max(n_cks - 1, 1))   # thin→thick
            alpha = 0.55 + 0.40 * (ci / max(n_cks - 1, 1))  # transparent→opaque
            entries[f"{method} ck{ck}"] = (margins, col, lw, "-", alpha)

        if args.final:
            margins = load_final_ck(data_dir, method)
            if margins is not None:
                margins = margin_transform(margins)
                entries[f"{method} final"] = (margins, base_hex, 2.0, FINAL_LSTYLE, 0.9)
            else:
                print(f"[skip] {method} final: no CSV found", file=sys.stderr)

        if entries:
            method_data[method] = entries
        else:
            print(f"[skip] {method}: no checkpoint data found", file=sys.stderr)

    if not method_data and base_margins is None:
        print("No data to plot. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── Print summary ─────────────────────────────────────────────────────────
    total_series = (1 if base_margins is not None else 0) + sum(
        len(v) for v in method_data.values()
    )
    print(f"Plotting {total_series} distributions "
          f"({'faceted' if args.facet else 'single axes'}, {args.plot_type})")

    # ── Shared label strings ──────────────────────────────────────────────────
    if args.abs:
        xlabel = "|true − false|  (confidence magnitude)"
        margin_label = "confidence magnitude"
        xmin = 0.0
    else:
        xlabel = "Logit margin  (true − false)"
        margin_label = "margin"
        xmin = None

    # ── Build flat data dict for single-axes ──────────────────────────────────
    if not args.facet:
        flat_data = {}
        if base_margins is not None:
            flat_data["Base"] = (base_margins, BASE_COLOR, 2.5, "-", 1.0)
        for method, entries in method_data.items():
            flat_data.update(entries)

        ck_str = (f"ck1–{N_CHECKPOINTS}" if checkpoints == list(range(1, N_CHECKPOINTS + 1))
                  else "ck" + ",".join(str(c) for c in checkpoints))
        title = (f"Bio logit {margin_label} distributions — "
                 f"{', '.join(methods[:4])}{'…' if len(methods) > 4 else ''} ({ck_str})")

        make_plot_single(flat_data, args.plot_type, out_path, args.bw_adjust, title, xlabel, xmin)
    else:
        make_plot_facet(base_margins, method_data, args.plot_type, out_path, args.bw_adjust, xlabel, xmin)


if __name__ == "__main__":
    main()
