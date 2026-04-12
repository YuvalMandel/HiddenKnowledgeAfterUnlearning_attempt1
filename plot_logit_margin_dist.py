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


def load_test_labels(pairs_csv: Path) -> list:
    """Return ordered list of expected labels ("True"/"False") for the test split.

    Reads data/wmdp_tf_pairs.csv, filters rows where split=="test", and returns
    labels in file order — which matches the row_id ordering in the logit CSVs.
    """
    import csv as _csv
    labels = []
    with open(pairs_csv, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if row["split"] == "test":
                labels.append(row["label"])   # "True" or "False"
    return labels


def compute_correct_mask(raw_margins: np.ndarray, labels: list) -> np.ndarray:
    """Boolean mask: True where the model's prediction matches the expected label.

    Uses the *raw* (signed) margin to decide the predicted label:
      pred = "True" if tf_margin > 0 else "False"
    Then correct = (pred == expected).

    Always call this on raw margins BEFORE any abs transform.
    """
    if len(raw_margins) != len(labels):
        raise ValueError(
            f"Margin array length ({len(raw_margins)}) != labels length ({len(labels)}). "
            "Make sure the pairs CSV and the logit data come from the same test split."
        )
    return np.array(
        [("True" if m > 0 else "False") == lab for m, lab in zip(raw_margins, labels)],
        dtype=bool,
    )


# ── Math helpers ──────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid: P(True) = 1 / (1 + exp(-margin))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _confidence(x: np.ndarray) -> np.ndarray:
    """Calibrated confidence magnitude: |P(True) - 0.5| * 2  ∈ [0, 1].
    0 = indifferent (P=0.5), 1 = certain (P=0 or 1)."""
    return np.abs(_sigmoid(x) - 0.5) * 2.0


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _kde_plot(ax, margins, color, lw, ls, alpha, label, bw_adjust=1.0,
              xmin=None, xmax=None):
    """Draw a KDE curve onto ax."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(margins, bw_method="scott")
    kde.set_bandwidth(kde.factor * bw_adjust)
    x_lo = margins.min() - 0.5
    x_hi = margins.max() + 0.5
    if xmin is not None:
        x_lo = max(x_lo, xmin)
    if xmax is not None:
        x_hi = min(x_hi, xmax)
    xs = np.linspace(x_lo, x_hi, 512)
    ax.plot(xs, kde(xs), color=color, lw=lw, ls=ls, alpha=alpha, label=label)


def _hist_plot(ax, margins, color, alpha, label, bins=60):
    ax.hist(margins, bins=bins, density=True,
            color=color, alpha=max(alpha * 0.7, 0.25), label=label,
            histtype="stepfilled", linewidth=0)


def _ecdf_plot(ax, margins, color, lw, ls, alpha, label):
    """Empirical CDF: step function from 0→1 as margin increases."""
    xs = np.sort(margins)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs, ys, color=color, lw=lw, ls=ls, alpha=alpha, label=label, where="post")


def _boxplot_on_ax(ax, data: dict, xlabel: str):
    """One box per series; colored by method."""
    labels = list(data.keys())
    arrays = [v[0] for v in data.values()]
    colors = [v[1] for v in data.values()]
    bps = ax.boxplot(arrays, patch_artist=True,
                     medianprops=dict(color="black", lw=2),
                     flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, col in zip(bps["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(xlabel, fontsize=10)


# ── Core per-axes draw helper ─────────────────────────────────────────────────

def _draw_on_ax(ax, data, plot_type, bw_adjust, xmin, xlabel, xmax=None):
    """Draw all series in data onto ax; set reference line, xlim, and axis labels."""
    if plot_type == "violin":
        _make_violin_axes(ax, data, xlabel=xlabel)
    elif plot_type == "boxplot":
        _boxplot_on_ax(ax, data, xlabel)
    else:
        for label, (margins, color, lw, ls, alpha) in data.items():
            if len(margins) < 2:
                print(f"  [warn] '{label}': only {len(margins)} point(s), skipping",
                      file=sys.stderr)
                continue
            if plot_type == "kde":
                _kde_plot(ax, margins, color, lw, ls, alpha, label, bw_adjust,
                          xmin=xmin, xmax=xmax)
            elif plot_type == "ecdf":
                _ecdf_plot(ax, margins, color, lw, ls, alpha, label)
            else:
                _hist_plot(ax, margins, color, alpha, label)

    if plot_type not in ("violin", "boxplot"):
        if xmin is None and xmax is None:
            ax.axvline(0, color="gray", lw=0.8, ls=":")
        if xmin is not None:
            ax.set_xlim(left=xmin)
        if xmax is not None:
            ax.set_xlim(right=xmax)

    ylabel = "Cumulative probability" if plot_type == "ecdf" else "Density"
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if plot_type == "ecdf":
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")


# ── Single-axes plot ──────────────────────────────────────────────────────────

def make_plot_single(
    data: dict,           # {label: (margins_array, color, lw, ls, alpha)}
    plot_type: str,
    out_path: Path,
    bw_adjust: float,
    title: str,
    xlabel: str = "Logit margin  (true − false)",
    xmin: float = None,
    xmax: float = None,
):
    """All distributions on one axes."""
    fig, ax = plt.subplots(figsize=(12, 5))
    _draw_on_ax(ax, data, plot_type, bw_adjust, xmin, xlabel, xmax=xmax)
    ax.set_title(title, fontsize=13)
    _add_legend(ax, n_cols=max(1, len(data) // 20))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Grid plot (1×2, 2×1, or 2×2 panels) ─────────────────────────────────────

def make_plot_grid(
    panels: list,     # [(panel_title, data_dict), ...]  — length == nrows * ncols
    nrows: int,
    ncols: int,
    plot_type: str,
    out_path: Path,
    bw_adjust: float,
    suptitle: str,
    xlabel: str = "Logit margin  (true − false)",
    xmin: float = None,
    xmax: float = None,
):
    """Render a nrows × ncols grid of distribution panels."""
    # Compute a shared x-range across all panels so comparisons are valid.
    # For KDE/hist/ecdf the data lives on the x-axis; violin/boxplot use y.
    if plot_type not in ("violin", "boxplot"):
        g_xmin, g_xmax = xmin, xmax
        if g_xmin is None or g_xmax is None:
            all_vals = []
            for _, pdata in panels:
                for margins, *_ in pdata.values():
                    if len(margins):
                        all_vals.append(float(margins.min()))
                        all_vals.append(float(margins.max()))
            if all_vals:
                if g_xmin is None:
                    g_xmin = min(all_vals) - 0.5
                if g_xmax is None:
                    g_xmax = max(all_vals) + 0.5
    else:
        g_xmin, g_xmax = xmin, xmax

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(9 * ncols, 5 * nrows),
                             squeeze=False,
                             sharey=True)
    axes_flat = axes.flatten()
    # sharey hides y-tick labels on all but the leftmost column by default;
    # restore them so every panel is self-contained.
    for ax in axes_flat:
        ax.tick_params(labelleft=True)

    for ax, (panel_title, data) in zip(axes_flat, panels):
        _draw_on_ax(ax, data, plot_type, bw_adjust, g_xmin, xlabel, xmax=g_xmax)
        ax.set_title(panel_title, fontsize=11)
        _add_legend(ax, n_cols=max(1, len(data) // 20))

    # Hide any unused axes (shouldn't happen with correct panel count, but defensive)
    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=13)
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
    xmax: float = None,
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
        panel_data = {}
        if base_margins is not None:
            panel_data["Base"] = (base_margins, BASE_COLOR, 2.0, "-", 0.9)
        panel_data.update(method_data[method])
        _draw_on_ax(ax, panel_data, plot_type, bw_adjust, xmin, xlabel, xmax=xmax)
        ax.set_title(method, fontsize=11)
        _add_legend(ax, n_cols=1)

    # Hide empty panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Bio logit margin distributions (KDE) — per method", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Statistics export ────────────────────────────────────────────────────────

def _flt(x):
    """Round float to 6 sig-figs, return None-safe."""
    return round(float(x), 6) if x is not None and not np.isnan(x) else None


def compute_series_stats(label: str,
                         raw_margins: np.ndarray,
                         labels: list | None,
                         base_raw: np.ndarray | None) -> dict:
    """Comprehensive statistics for one series (always uses raw signed margins).

    All derived transforms (prob, conf, abs) are computed internally here so
    the export is independent of the current display mode.

    Parameters
    ----------
    label      : series name, e.g. "ELM ck3"
    raw_margins: 1-D array of (true_logit - false_logit) values
    labels     : list of "True"/"False" expected labels (same length), or None
    base_raw   : raw margins of the base model for KS / Wasserstein, or None
    """
    from scipy import stats as sp_stats

    n = len(raw_margins)
    probs = _sigmoid(raw_margins)          # P(True)  ∈ [0, 1]
    confs = _confidence(raw_margins)       # |P-0.5|*2 ∈ [0, 1]

    s: dict = {
        "series":  label,
        "n_total": n,
        # ── Raw margin ────────────────────────────────────────────────────────
        "margin_mean":   _flt(np.mean(raw_margins)),
        "margin_median": _flt(np.median(raw_margins)),
        "margin_std":    _flt(np.std(raw_margins)),
        "margin_p10":    _flt(np.percentile(raw_margins, 10)),
        "margin_p25":    _flt(np.percentile(raw_margins, 25)),
        "margin_p75":    _flt(np.percentile(raw_margins, 75)),
        "margin_p90":    _flt(np.percentile(raw_margins, 90)),
        # ── Probability space (P(True) = sigmoid(margin)) ────────────────────
        "prob_true_mean":   _flt(np.mean(probs)),
        "prob_true_median": _flt(np.median(probs)),
        "prob_true_std":    _flt(np.std(probs)),
        # ── Confidence magnitude (|P - 0.5| * 2) ────────────────────────────
        "conf_mean":   _flt(np.mean(confs)),
        "conf_median": _flt(np.median(confs)),
        "conf_std":    _flt(np.std(confs)),
        # ── Fraction predicted "True" (margin > 0) ───────────────────────────
        "frac_pred_true": _flt(np.mean(raw_margins > 0)),
    }

    # ── Splits on gold label + correctness ───────────────────────────────────
    if labels is not None and len(labels) == n:
        gold_true  = np.array([l == "True" for l in labels])
        pred_true  = raw_margins > 0
        correct    = pred_true == gold_true   # per-model correctness

        s["accuracy"]    = _flt(correct.mean())
        s["n_correct"]   = int(correct.sum())
        s["n_incorrect"] = int((~correct).sum())

        # Margin by correctness
        for name, mask in [("correct", correct), ("incorrect", ~correct)]:
            sub = raw_margins[mask]
            s[f"margin_{name}_mean"]   = _flt(sub.mean())   if len(sub) else None
            s[f"margin_{name}_median"] = _flt(np.median(sub)) if len(sub) else None
            s[f"margin_{name}_std"]    = _flt(sub.std())    if len(sub) else None
            s[f"prob_{name}_mean"]     = _flt(_sigmoid(sub).mean()) if len(sub) else None
            s[f"conf_{name}_mean"]     = _flt(_confidence(sub).mean()) if len(sub) else None

        # Margin by gold label
        s["n_gold_true"]  = int(gold_true.sum())
        s["n_gold_false"] = int((~gold_true).sum())
        for name, mask in [("gold_true", gold_true), ("gold_false", ~gold_true)]:
            sub = raw_margins[mask]
            s[f"margin_{name}_mean"]   = _flt(sub.mean())   if len(sub) else None
            s[f"margin_{name}_median"] = _flt(np.median(sub)) if len(sub) else None
            s[f"margin_{name}_std"]    = _flt(sub.std())    if len(sub) else None
            s[f"prob_{name}_mean"]     = _flt(_sigmoid(sub).mean()) if len(sub) else None
            s[f"conf_{name}_mean"]     = _flt(_confidence(sub).mean()) if len(sub) else None

        # 2×2 breakdown (correct/incorrect × gold_true/gold_false)
        for corr_name, corr_mask in [("correct", correct), ("incorrect", ~correct)]:
            for gold_name, gold_mask in [("gold_true", gold_true), ("gold_false", ~gold_true)]:
                combo = corr_mask & gold_mask
                sub   = raw_margins[combo]
                key   = f"{corr_name}_{gold_name}"
                s[f"n_{key}"]              = int(combo.sum())
                s[f"margin_{key}_mean"]    = _flt(sub.mean())            if len(sub) else None
                s[f"margin_{key}_median"]  = _flt(np.median(sub))        if len(sub) else None
                s[f"prob_{key}_mean"]      = _flt(_sigmoid(sub).mean())  if len(sub) else None
                s[f"conf_{key}_mean"]      = _flt(_confidence(sub).mean()) if len(sub) else None

    # ── vs base: KS statistic & Wasserstein distance ─────────────────────────
    if base_raw is not None and label != "Base":
        try:
            ks = sp_stats.ks_2samp(raw_margins, base_raw)
            s["ks_stat_vs_base"] = _flt(ks.statistic)
            s["ks_pval_vs_base"] = _flt(ks.pvalue)
        except Exception:
            s["ks_stat_vs_base"] = s["ks_pval_vs_base"] = None
        try:
            from scipy.stats import wasserstein_distance
            s["wasserstein_vs_base"] = _flt(wasserstein_distance(raw_margins, base_raw))
        except Exception:
            s["wasserstein_vs_base"] = None

    return s


def export_stats(raw_margins_map: dict,
                 labels: list | None,
                 prefix: str) -> None:
    """Compute stats for every series and write {prefix}.csv and {prefix}.json.

    Parameters
    ----------
    raw_margins_map : {series_label: raw_np_array}
    labels          : ordered list of "True"/"False" gold labels (same length as arrays)
    prefix          : output path prefix, e.g. "stats/elm_ck8"
    """
    import csv as _csv

    base_raw = raw_margins_map.get("Base")
    all_stats = []
    for lbl, raw in raw_margins_map.items():
        st = compute_series_stats(lbl, raw, labels, base_raw)
        all_stats.append(st)

    # ── JSON (full nested stats) ──────────────────────────────────────────────
    json_path = Path(f"{prefix}.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Stats JSON → {json_path}")

    # ── CSV (flat, one row per series) ───────────────────────────────────────
    csv_path = Path(f"{prefix}.csv")
    # Collect all unique keys in order
    fieldnames: list = []
    seen: set = set()
    for st in all_stats:
        for k in st:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_stats)
    print(f"Stats CSV  → {csv_path}")


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
    # ── Transform (mutually exclusive) ───────────────────────────────────────
    xform = parser.add_mutually_exclusive_group()
    xform.add_argument(
        "--abs", action="store_true",
        help="Display |margin|  (absolute confidence, raw logit scale).",
    )
    xform.add_argument(
        "--prob", action="store_true",
        help="Display P(True) = sigmoid(margin) ∈ [0,1]  (probability space).",
    )
    xform.add_argument(
        "--conf", action="store_true",
        help="Display |P(True) − 0.5| × 2 ∈ [0,1]  "
             "(calibrated confidence: 0=random, 1=certain).",
    )
    parser.add_argument(
        "--split_correct", action="store_true",
        help="Split panels by whether each model predicted correctly (per-model). "
             "Alone: 2 panels (correct | incorrect). "
             "With --split_gold: 2×2 grid.",
    )
    parser.add_argument(
        "--split_gold", action="store_true",
        help="Split panels by the gold label (True | False). "
             "Alone: 2 panels (gold True | gold False). "
             "With --split_correct: 2×2 grid.",
    )
    parser.add_argument(
        "--pairs_csv", default="data/wmdp_tf_pairs.csv",
        help="Path to wmdp_tf_pairs.csv (needed for --split_correct / --split_gold). "
             "Default: data/wmdp_tf_pairs.csv",
    )
    parser.add_argument(
        "--no_base", action="store_true",
        help="Omit the base model distribution.",
    )
    parser.add_argument(
        "--plot_type",
        choices=["kde", "hist", "ecdf", "violin", "boxplot"],
        default="kde",
        help="Plot type (default: kde).  "
             "kde/hist/ecdf show density/CDF.  "
             "violin/boxplot show spread per series.",
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
        "--export_stats", default=None, metavar="PREFIX",
        help="Compute and export comprehensive statistics to PREFIX.csv and PREFIX.json "
             "(e.g. --export_stats stats/elm_margins).  "
             "Includes: mean/median/std/percentiles, accuracy, correctness splits, "
             "gold-label splits, 2×2 breakdowns, KS test and Wasserstein vs base.  "
             "If data/wmdp_tf_pairs.csv is reachable, label-based stats are always included.",
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Skip generating the PNG (useful with --export_stats).",
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
        xform_tag  = ("_prob" if args.prob else "_conf" if args.conf
                      else "_abs" if args.abs else "")
        split_tag  = ("_splitCG" if (args.split_correct and args.split_gold)
                      else "_splitC" if args.split_correct
                      else "_splitG" if args.split_gold
                      else "")
        facet_tag  = "_facet" if args.facet else ""
        suffix = f"_{args.plot_type}{xform_tag}{split_tag}{facet_tag}"
        out_path = Path(f"plots/logit_margin_dist_m{mstr}_ck{ckstr}{suffix}.png")
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.prob:
        margin_transform = _sigmoid
    elif args.conf:
        margin_transform = _confidence
    elif args.abs:
        margin_transform = np.abs
    else:
        margin_transform = lambda x: x

    # raw_margins_map keeps the pre-transform margins for correctness splitting.
    # flat_data / method_data store post-transform (display) margins.
    raw_margins_map: dict = {}   # {series_label: raw_np_array}

    raw_base = None if args.no_base else load_base(data_dir)
    base_margins = margin_transform(raw_base) if raw_base is not None else None
    if raw_base is not None:
        raw_margins_map["Base"] = raw_base

    # method_data[method][ck_label] = (margins, color, lw, ls, alpha)
    method_data: dict = {}
    n_cks = len(checkpoints)
    for method in methods:
        base_hex = _METHOD_BASE_COLORS.get(method, "#333333")
        ck_colors = _ck_colors(base_hex, n_cks)
        entries = {}

        for ci, ck in enumerate(sorted(checkpoints)):
            raw = load_sweep_ck(data_dir, method, ck)
            if raw is None:
                print(f"[skip] {method} ck{ck}: no data", file=sys.stderr)
                continue
            lbl   = f"{method} ck{ck}"
            raw_margins_map[lbl] = raw
            col   = ck_colors[ci]
            lw    = 1.2 + 0.4 * (ci / max(n_cks - 1, 1))   # thin→thick
            alpha = 0.55 + 0.40 * (ci / max(n_cks - 1, 1))  # transparent→opaque
            entries[lbl] = (margin_transform(raw), col, lw, "-", alpha)

        if args.final:
            raw = load_final_ck(data_dir, method)
            if raw is not None:
                lbl = f"{method} final"
                raw_margins_map[lbl] = raw
                entries[lbl] = (margin_transform(raw), base_hex, 2.0, FINAL_LSTYLE, 0.9)
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
    if args.prob:
        xlabel       = "P(True) = σ(margin)  [0, 1]"
        margin_label = "P(True)"
        xmin, xmax   = 0.0, 1.0
    elif args.conf:
        xlabel       = "Confidence = |P(True) − 0.5| × 2  [0, 1]"
        margin_label = "confidence"
        xmin, xmax   = 0.0, 1.0
    elif args.abs:
        xlabel       = "|margin|  (|true − false|)"
        margin_label = "|margin|"
        xmin, xmax   = 0.0, None
    else:
        xlabel       = "Logit margin  (true − false)"
        margin_label = "margin"
        xmin, xmax   = None, None

    # ── Build flat data dict ───────────────────────────────────────────────────
    flat_data = {}
    if base_margins is not None:
        flat_data["Base"] = (base_margins, BASE_COLOR, 2.5, "-", 1.0)
    for method, entries in method_data.items():
        flat_data.update(entries)

    ck_str = (f"ck1–{N_CHECKPOINTS}" if checkpoints == list(range(1, N_CHECKPOINTS + 1))
              else "ck" + ",".join(str(c) for c in checkpoints))
    title = (f"Bio logit {margin_label} distributions — "
             f"{', '.join(methods[:4])}{'…' if len(methods) > 4 else ''} ({ck_str})")

    # ── Load labels for splits and/or stats export ────────────────────────────
    need_labels = args.split_correct or args.split_gold or args.export_stats
    labels: list | None = None
    if need_labels:
        pairs_csv = Path(args.pairs_csv)
        if pairs_csv.exists():
            labels = load_test_labels(pairs_csv)
        elif args.split_correct or args.split_gold:
            print(f"[error] {pairs_csv} not found (needed for --split_correct / --split_gold).",
                  file=sys.stderr)
            sys.exit(1)
        else:
            print(f"[warn] {pairs_csv} not found — stats will omit label-based columns.",
                  file=sys.stderr)

    # ── Export statistics ─────────────────────────────────────────────────────
    if args.export_stats:
        export_stats(raw_margins_map, labels, args.export_stats)

    if args.no_plot:
        return

    # ── Splitting (correct and/or gold label) ────────────────────────────────
    need_split = args.split_correct or args.split_gold
    if need_split:
        gold_mask = np.array([lab == "True" for lab in labels])  # same for all series

        # Per-series correctness masks (only computed when --split_correct)
        correct_masks: dict = {}
        if args.split_correct:
            for lbl, (disp, *_) in flat_data.items():
                raw = raw_margins_map.get(lbl)
                if raw is None or len(raw) != len(labels):
                    print(f"[warn] '{lbl}': length mismatch, treating as all-correct.",
                          file=sys.stderr)
                    correct_masks[lbl] = np.ones(len(disp), dtype=bool)
                else:
                    correct_masks[lbl] = compute_correct_mask(raw, labels)

        def _subset(combined_mask_fn):
            """Build a flat_data dict filtered by a per-label mask function."""
            d = {}
            for lbl, (disp, color, lw, ls, alpha) in flat_data.items():
                raw = raw_margins_map.get(lbl)
                if raw is None or len(raw) != len(labels):
                    d[lbl] = (disp, color, lw, ls, alpha)
                else:
                    m = combined_mask_fn(lbl)
                    d[lbl] = (disp[m], color, lw, ls, alpha)
            return d

        if args.split_correct and args.split_gold:
            # 2×2: correct/incorrect × gold True/False
            # Correct + Gold True  → model predicted True,  gold True  (TP)
            # Correct + Gold False → model predicted False, gold False (TN)
            # Incorrect + Gold True  → model predicted False, gold True  (FN)
            # Incorrect + Gold False → model predicted True,  gold False (FP)
            panels = [
                ("TP — correct\ngold=True,  model predicted True",
                 _subset(lambda l: correct_masks[l] & gold_mask)),
                ("TN — correct\ngold=False,  model predicted False",
                 _subset(lambda l: correct_masks[l] & ~gold_mask)),
                ("FN — incorrect\ngold=True,  model predicted False",
                 _subset(lambda l: ~correct_masks[l] & gold_mask)),
                ("FP — incorrect\ngold=False,  model predicted True",
                 _subset(lambda l: ~correct_masks[l] & ~gold_mask)),
            ]
            make_plot_grid(panels, 2, 2, args.plot_type, out_path, args.bw_adjust,
                           title, xlabel, xmin, xmax)
        elif args.split_correct:
            panels = [
                ("Correct  (model answer = gold answer)\nfraction varies per model",
                 _subset(lambda l: correct_masks[l])),
                ("Incorrect  (model answer \u2260 gold answer)\nfraction varies per model",
                 _subset(lambda l: ~correct_masks[l])),
            ]
            make_plot_grid(panels, 1, 2, args.plot_type, out_path, args.bw_adjust,
                           title, xlabel, xmin, xmax)
        else:
            panels = [
                ("Gold = True  (correct answer is True)\n50% of questions by construction",
                 _subset(lambda l: gold_mask)),
                ("Gold = False  (correct answer is False)\n50% of questions by construction",
                 _subset(lambda l: ~gold_mask)),
            ]
            make_plot_grid(panels, 1, 2, args.plot_type, out_path, args.bw_adjust,
                           title, xlabel, xmin, xmax)
    elif not args.facet:
        make_plot_single(flat_data, args.plot_type, out_path, args.bw_adjust,
                         title, xlabel, xmin, xmax)
    else:
        make_plot_facet(base_margins, method_data, args.plot_type, out_path,
                        args.bw_adjust, xlabel, xmin, xmax)


if __name__ == "__main__":
    main()
