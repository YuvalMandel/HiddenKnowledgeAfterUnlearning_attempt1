#!/usr/bin/env python3
"""
plot_sweep_bars.py — Flexible grouped bar plot for sweep results.

Each row in a sweep CSV is one (method, checkpoint) pair.
This script melts the wide CSV into long format, filters by any combination
of dimensions, then plots grouped bars with full control over which dimension
goes on the x-axis and which becomes the bar color.

Dimensions
----------
  method        — unlearning method (GradDiff, RMU, …)
  checkpoint    — checkpoint number (1–8)
  dataset       — bio | cyber
  probe_source  — bp (base probes) | mp (method probes) | gen | logit | mcq
  clf           — LR | rf | adaboost   (N/A for gen/logit/mcq)
  probe_type    — pl | ml | fl | ib | ibe | eb | vote | avg | gen | logit | mcq
  metric        — acc | auc | f1 | true | fals | prec | rec | gib | valid_acc

Usage examples
--------------
  # Compare methods at ck8, LR, mid-band, bio accuracy
  python plot_sweep_bars.py \\
      --checkpoints 8 --clf LR --probe_type ml --metric acc \\
      --group_by method

  # Compare probe types for GradDiff ck8, bio, LR
  python plot_sweep_bars.py \\
      --methods GradDiff --checkpoints 8 --clf LR --metric acc \\
      --probe_type pl,ml,fl,ib,ibe,eb --group_by probe_type

  # Compare methods × probe_type for ck8, LR, bio AUC
  python plot_sweep_bars.py \\
      --checkpoints 8 --clf LR --probe_type ml,fl,ib --metric auc \\
      --group_by method --color_by probe_type

  # Compare checkpoints for GradDiff, LR, ml, bio
  python plot_sweep_bars.py \\
      --methods GradDiff --clf LR --probe_type ml --metric acc \\
      --group_by checkpoint

  # Compare all classifiers for GradDiff ck8, ml, bio
  python plot_sweep_bars.py \\
      --methods GradDiff --checkpoints 8 --probe_type ml --metric acc \\
      --clf all --group_by clf
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────────────────────

ALL_METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J",
               "Llama3-8B"]
# "Base" loads from base_results.json (instruct base, before unlearning)

PROBE_SRCS  = {"bp", "mp"}
PROBE_CLFS  = {"lr", "rf", "adaboost"}
PROBE_BANDS = {"pl", "ml", "mb", "vote", "avg", "fl", "ib", "ibe", "eb", "ibnp"}
NON_PROBE   = {"gen", "logit", "mcq"}
SKIP_COLS   = {"lyr"}          # best-layer integer, not a plottable metric

BAND_LABELS = {
    "pl":    "Per-Layer",   "ml":    "Mid-Band",   "mb":    "Mid-Band",
    "fl":    "Full-Layer",  "ib":    "Init-Band",
    "ibe":   "Init+Emb",    "eb":    "End-Band",   "ibnp":  "IB-NoPCA",
    "vote":  "Vote-Ens",    "avg":   "Avg-Ens",
    "gen":   "Generation",  "logit": "Logit",     "mcq": "MCQ",
}
CLF_LABELS  = {"lr": "LR", "rf": "RF", "adaboost": "AdaBoost", "N/A": "—"}
SRC_LABELS  = {"bp": "Base-Probe", "mp": "Method-Probe",
               "gen": "Gen", "logit": "Logit", "mcq": "MCQ"}
DIM_LABELS  = {
    "method":       "Method",
    "checkpoint":   "Checkpoint",
    "clf":          "Classifier",
    "probe_type":   "Probe Type",
    "probe_source": "Probe Source",
    "dataset":      "Dataset",
    "metric":       "Metric",
}
METRIC_LABELS = {
    "acc": "Accuracy",       "auc": "AUC",       "f1": "F1",
    "true": "True-Acc",      "fals": "False-Acc",
    "prec": "Precision",     "rec": "Recall",
    "gib": "Gibberish Rate", "valid_acc": "Valid Accuracy",
}

# ── Safe name (matches hidden_knowledge_after_unlearning.py) ───────────────

def safe_name(n: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", n)


# ── CSV loading ────────────────────────────────────────────────────────────

def load_all_sweep_csvs(data_dir: Path, methods_filter: str) -> pd.DataFrame:
    """Load per-method sweep CSVs and concatenate with a 'method' column."""
    if methods_filter == "all":
        method_list = ALL_METHODS
    else:
        method_list = [m.strip() for m in methods_filter.split(",")]

    frames = []
    for method in method_list:
        sn   = safe_name(method)
        path = data_dir / f"sweep_{sn}" / f"{sn}_sweep.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping '{method}'", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df.insert(0, "method", method)
        frames.append(df)

    if not frames:
        print("ERROR: No sweep CSVs found.", file=sys.stderr)
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


# ── Wide → Long melt ───────────────────────────────────────────────────────

def melt_sweep_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt wide sweep DataFrame into long format with columns:
      method, checkpoint, dataset, probe_source, clf, probe_type, metric, value
    """
    id_cols = {"method", "checkpoint", "model_id"}
    records = []

    for _, row in df.iterrows():
        base = {"method": row["method"], "checkpoint": int(row["checkpoint"])}

        for col, val in row.items():
            if col in id_cols:
                continue
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue

            # Strip optional cyber_ prefix
            if col.startswith("cyber_"):
                dataset = "cyber"
                rest    = col[len("cyber_"):]
            else:
                dataset = "bio"
                rest    = col

            parsed = _parse_col(rest)
            if parsed is None:
                continue
            probe_source, clf, probe_type, metric = parsed
            if metric in SKIP_COLS:
                continue

            records.append({
                **base,
                "dataset":      dataset,
                "probe_source": probe_source,
                "clf":          clf,
                "probe_type":   probe_type,
                "metric":       metric,
                "value":        val,
            })

    return pd.DataFrame(records)


def _parse_col(rest: str):
    """
    Parse a (non-cyber-prefixed) column name.
    Returns (probe_source, clf, probe_type, metric) or None.
    """
    # Non-probe: gen_*, logit_*, mcq_*
    for ptype in NON_PROBE:
        prefix = f"{ptype}_"
        if rest.startswith(prefix):
            metric = rest[len(prefix):]
            return (ptype, "N/A", ptype, metric)

    # Probe: {src}_{clf}_{band}_{metric}
    parts = rest.split("_")
    if len(parts) < 4:
        return None
    src, clf, band = parts[0], parts[1], parts[2]
    if src not in PROBE_SRCS or clf not in PROBE_CLFS or band not in PROBE_BANDS:
        return None
    band = "ml" if band == "mb" else band   # normalise CSV abbrev → canonical
    metric = "_".join(parts[3:])
    return (src, clf, band, metric)


# ── Pipeline JSON → long format ────────────────────────────────────────────

# Maps JSON probe-band key → CSV band abbreviation
_BAND_KEY_MAP = {
    "per_layer":      "pl",
    "mid_band":       "ml",
    "multi_layer":    "ml",   # legacy name
    "vote_ensemble":  "vote",
    "avg_ensemble":   "avg",
    "full_layer":     "fl",
    "init_band":      "ib",
    "init_band_emb":  "ibe",
    "end_band":       "eb",
    "ib_no_pca":      "ibnp",
}
# Maps JSON CLF key → internal lowercase
_CLF_KEY_MAP = {"LR": "lr", "RF": "rf", "AdaBoost": "adaboost"}
# Maps JSON metric key → internal name
_METRIC_KEY_MAP = {
    "accuracy":       "acc",
    "true_accuracy":  "true",
    "false_accuracy": "fals",
    "precision":      "prec",
    "recall":         "rec",
    "f1":             "f1",
    "auc":            "auc",
    "accuracy_valid": "valid_acc",
    "gibberish_rate": "gib",
    "mean_margin":    "margin",
}


def _probe_stats_to_records(aps: dict, base: dict,
                             dataset: str, probe_source: str) -> list:
    """Convert an all_probe_stats dict to a list of long-format record dicts."""
    records = []
    for band_key, band_abbr in _BAND_KEY_MAP.items():
        clf_dict = aps.get(band_key)
        if not clf_dict:
            continue
        for clf_key, clf_abbr in _CLF_KEY_MAP.items():
            stats = clf_dict.get(clf_key)
            if not stats:
                continue
            for metric_key, metric_abbr in _METRIC_KEY_MAP.items():
                if metric_key not in stats:
                    continue
                try:
                    val = float(stats[metric_key])
                except (TypeError, ValueError):
                    continue
                records.append({**base,
                                 "dataset":      dataset,
                                 "probe_source": probe_source,
                                 "clf":          clf_abbr,
                                 "probe_type":   band_abbr,
                                 "metric":       metric_abbr,
                                 "value":        val})
    return records


def _gen_stats_to_records(gs: dict, base: dict,
                           dataset: str, probe_source: str) -> list:
    """Convert a gen/logit stats dict to long-format records."""
    records = []
    for metric_key, metric_abbr in _METRIC_KEY_MAP.items():
        if metric_key not in gs:
            continue
        try:
            val = float(gs[metric_key])
        except (TypeError, ValueError):
            continue
        records.append({**base,
                         "dataset":      dataset,
                         "probe_source": probe_source,
                         "clf":          "N/A",
                         "probe_type":   probe_source,
                         "metric":       metric_abbr,
                         "value":        val})
    return records


def _load_method_json(ck_dir: Path, method: str) -> list:
    """Load one method results.json and return long-format records."""
    import json
    sn   = safe_name(method)
    path = ck_dir / f"{sn}_results.json"
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping '{method}'", file=sys.stderr)
        return []
    with open(path) as f:
        r = json.load(f)

    base = {"method": method, "checkpoint": 8}
    records = []

    # Bio
    for key, src in (("gen", "gen"), ("logit", "logit"), ("mcq", "mcq")):
        if key in r:
            records += _gen_stats_to_records(r[key], base, "bio", src)
    if "all_base_probe_stats" in r:
        records += _probe_stats_to_records(r["all_base_probe_stats"],  base, "bio", "bp")
    if "all_method_probe_stats" in r:
        records += _probe_stats_to_records(r["all_method_probe_stats"], base, "bio", "mp")

    # Cyber — use "og" subset as representative
    og = (r.get("cyber_subsets") or {}).get("og") or {}
    for key, src in (("gen", "gen"), ("logit", "logit")):
        if og.get(key):
            records += _gen_stats_to_records(og[key], base, "cyber", src)
    if og.get("base_probes"):
        records += _probe_stats_to_records(og["base_probes"],   base, "cyber", "bp")
    if og.get("method_probes"):
        records += _probe_stats_to_records(og["method_probes"], base, "cyber", "mp")

    return records


def _load_base_json(ck_dir: Path) -> list:
    """Load base_results.json (instruct base model before unlearning)."""
    import json
    path = ck_dir / "base_results.json"
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping 'Base'", file=sys.stderr)
        return []
    with open(path) as f:
        r = json.load(f)

    base = {"method": "Base", "checkpoint": 8}
    records = []

    # Bio — different key names than method JSONs
    for key, src in (("gen_stats", "gen"), ("logit_stats", "logit"), ("mcq_stats", "mcq")):
        if key in r:
            records += _gen_stats_to_records(r[key], base, "bio", src)
    if "all_probe_stats" in r:
        # Base model has one probe set; expose as both bp and mp for compatibility
        records += _probe_stats_to_records(r["all_probe_stats"], base, "bio", "bp")
        records += _probe_stats_to_records(r["all_probe_stats"], base, "bio", "mp")

    # Cyber — base uses "probes" key (not "base_probes"/"method_probes")
    og = (r.get("cyber_subsets") or {}).get("og") or {}
    for key, src in (("gen", "gen"), ("logit", "logit")):
        if og.get(key):
            records += _gen_stats_to_records(og[key], base, "cyber", src)
    if og.get("probes"):
        records += _probe_stats_to_records(og["probes"], base, "cyber", "bp")
        records += _probe_stats_to_records(og["probes"], base, "cyber", "mp")

    return records


def load_base_as_ck0(ck_dir: Path, methods: list) -> pd.DataFrame:
    """Load base model results as checkpoint-0 rows, cloned for each method."""
    records = _load_base_json(ck_dir)
    # Keep only mp rows to avoid duplicate bp/mp entries
    records = [r for r in records if r.get("probe_source") == "mp"]
    out = []
    for method in methods:
        for r in records:
            out.append({**r, "method": method, "checkpoint": 0})
    return pd.DataFrame(out) if out else pd.DataFrame()


def load_from_pipeline(ck_dir: Path, methods_filter: str) -> pd.DataFrame:
    """
    Load pipeline results (checkpoint-8 models) from checkpoints/*.json.
    Supports special names "Base" and "Llama3-8B" in addition to ALL_METHODS.
    Returns the same long-format DataFrame as melt_sweep_df().
    """
    if methods_filter == "all":
        method_list = ["Base"] + ALL_METHODS
    else:
        method_list = [m.strip() for m in methods_filter.split(",")]

    records = []
    for method in method_list:
        if method == "Base":
            records += _load_base_json(ck_dir)
        else:
            records += _load_method_json(ck_dir, method)

    if not records:
        print("ERROR: No pipeline results found.", file=sys.stderr)
        sys.exit(1)

    return pd.DataFrame(records)


# ── Filtering ──────────────────────────────────────────────────────────────

def _parse_list(s: str, all_vals) -> list:
    if s == "all":
        return sorted(all_vals, key=str)
    return [v.strip() for v in s.split(",")]


def _normalise_clf(raw: list) -> list:
    """Accept LR/RF/AdaBoost (case-insensitive) and return lowercase internal names."""
    mapping = {"lr": "lr", "rf": "rf", "adaboost": "adaboost",
               "LR": "lr", "RF": "rf", "AdaBoost": "adaboost",
               "N/A": "N/A"}
    return [mapping.get(c, c.lower()) for c in raw]


def filter_long_df(long_df: pd.DataFrame, *,
                   methods, checkpoints, clfs, probe_types,
                   probe_sources, metrics, datasets) -> pd.DataFrame:
    # N/A clf (gen/logit/mcq rows) always passes the clf filter
    clf_mask = long_df["clf"].isin(clfs) | (long_df["clf"] == "N/A")
    mask = (
        long_df["method"].isin(methods)
        & long_df["checkpoint"].isin(checkpoints)
        & clf_mask
        & long_df["probe_type"].isin(probe_types)
        & long_df["probe_source"].isin(probe_sources)
        & long_df["metric"].isin(metrics)
        & long_df["dataset"].isin(datasets)
    )
    return long_df[mask].copy()


# ── Plotting ───────────────────────────────────────────────────────────────

_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def _dim_label(dim: str, val) -> str:
    """Human-readable label for a dimension value."""
    if dim == "checkpoint":
        return f"ck{int(val)}"
    if dim == "probe_type":
        return BAND_LABELS.get(str(val), str(val))
    if dim == "clf":
        return CLF_LABELS.get(str(val), str(val))
    if dim == "probe_source":
        return SRC_LABELS.get(str(val), str(val))
    return str(val)


def _make_combo_col(df, dims: list, col_name: str):
    """Add a combined label column from multiple dimension columns (e.g. 'LR + Mid-Band')."""
    parts = [df[d].apply(lambda v, d=d: _dim_label(d, v)) for d in dims]
    result = parts[0]
    for p in parts[1:]:
        result = result + " + " + p
    df = df.copy()
    df[col_name] = result
    return df


def _natural_sort_key(x):
    """Sort key that handles numeric strings naturally."""
    try:
        return (0, int(x))
    except (ValueError, TypeError):
        return (1, str(x))


def plot_bars(long_df: pd.DataFrame, *,
              group_by: list,
              color_by: list,
              metrics: list[str],
              title: str | None,
              out_path: Path,
              figsize: tuple,
              show_values: bool,
              sort_by: str,
              chance_line: float | None = 0.5):
    """
    Aggregate (mean over all non-grouped dims) and draw a grouped bar chart.
    group_by / color_by are lists of dimension names; multi-dim combos are
    joined with ' + ' to form composite group/hue labels.
    One subplot per metric if multiple metrics are requested.
    """
    GRP = "__group__"
    HUE = "__hue__"

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics,
                             figsize=(figsize[0] * n_metrics, figsize[1]),
                             squeeze=False)

    for ax, metric in zip(axes[0], metrics):
        sub = long_df[long_df["metric"] == metric].copy()
        if sub.empty:
            ax.set_title(f"No data for metric={metric}")
            continue

        sub = _make_combo_col(sub, group_by, GRP)
        if color_by:
            sub = _make_combo_col(sub, color_by, HUE)

        agg_dims = [GRP] + ([HUE] if color_by else [])
        agg = sub.groupby(agg_dims, as_index=False)["value"].mean()

        # Determine groups and hues
        groups = sorted(agg[GRP].unique(), key=_natural_sort_key)
        if sort_by == "value":
            means  = agg.groupby(GRP)["value"].mean()
            groups = sorted(groups, key=lambda g: means.get(g, 0), reverse=True)

        hues = sorted(agg[HUE].unique(), key=_natural_sort_key) if color_by else [None]

        n_g   = len(groups)
        n_h   = len(hues)
        bar_w = min(0.8 / n_h, 0.35)
        x     = np.arange(n_g)

        for i, hue in enumerate(hues):
            sub_h = agg[agg[HUE] == hue] if color_by and hue is not None else agg

            vals = []
            for g in groups:
                row = sub_h[sub_h[GRP] == g]["value"]
                vals.append(float(row.iloc[0]) if len(row) > 0 else float("nan"))

            offset = (i - n_h / 2 + 0.5) * bar_w
            bars   = ax.bar(x + offset, vals, bar_w * 0.9,
                            label=hue,
                            color=_COLORS[i % len(_COLORS)],
                            alpha=0.85, edgecolor="white", linewidth=0.5)

            if show_values:
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.003,
                                f"{v:.3f}", ha="center", va="bottom",
                                fontsize=6.5, rotation=45)

        # Axes formatting
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        metric_lbl = METRIC_LABELS.get(metric, metric)
        ax.set_ylabel(metric_lbl, fontsize=10)
        grp_label = " + ".join(DIM_LABELS.get(d, d) for d in group_by)
        ax.set_xlabel(grp_label, fontsize=10)

        # Y-axis floor near data minimum (never below 0)
        vmin = sub["value"].min()
        vmax = sub["value"].max()
        pad  = max(0.02, (vmax - vmin) * 0.15)
        ax.set_ylim(bottom=max(0.0, vmin - pad),
                    top=min(1.0, vmax + pad + (0.07 if show_values else 0.02)))

        if chance_line is not None and 0 < chance_line < 1:
            ax.axhline(chance_line, color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.5, label=f"Chance ({chance_line})")

        ax_title = title if title else _auto_title(sub, group_by, color_by, metric)
        if n_metrics == 1:
            ax.set_title(ax_title, fontsize=11)
        else:
            ax.set_title(metric_lbl, fontsize=10)

        if color_by or (chance_line is not None):
            legend_title = " + ".join(DIM_LABELS.get(d, d) for d in color_by) if color_by else None
            ax.legend(title=legend_title,
                      bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8,
                      title_fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)

    if n_metrics > 1 and title:
        fig.suptitle(title, fontsize=12, y=1.01)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def _auto_title(df, group_by: list, color_by: list, metric):
    grp_lbl = " + ".join(DIM_LABELS.get(d, d) for d in group_by)
    parts = [f"{METRIC_LABELS.get(metric, metric)} by {grp_lbl}"]
    if color_by:
        hue_lbl = " + ".join(DIM_LABELS.get(d, d) for d in color_by)
        parts.append(f"× {hue_lbl}")
    # Append fixed-context labels for dimensions with a single value
    active = set(group_by) | set(color_by)
    for dim in ("dataset", "probe_type", "clf", "probe_source", "method", "checkpoint"):
        if dim in active or dim not in df.columns:
            continue
        vals = [v for v in df[dim].unique() if v != "N/A"]
        if len(vals) == 1:
            parts.append(f"[{_dim_label(dim, vals[0])}]")
    return " ".join(parts)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Filter arguments
    ap.add_argument("--methods",      default="all",
                    help="Method(s): GradDiff,RMU,… or 'all'")
    ap.add_argument("--checkpoints",  default="all",
                    help="Checkpoint(s): 1,4,8 or 'all' (integers 1–8) or 'latest' "
                         "(reads pipeline results.json for ck8 instead of sweep CSVs)")
    ap.add_argument("--clf",          default="all",
                    help="Classifier(s): LR,RF,AdaBoost or 'all'")
    ap.add_argument("--probe_type",   default="ml",
                    help="Probe type(s): pl,ml,fl,ib,ibe,eb,vote,avg,gen,logit,mcq or 'all'")
    ap.add_argument("--probe_source", default="bp",
                    help="Probe source(s): bp,mp or 'all'")
    ap.add_argument("--metric",       default="acc",
                    help="Metric(s): acc,auc,f1,true,fals,prec,rec,gib,valid_acc or 'all'")
    ap.add_argument("--dataset",      default="bio",
                    help="Dataset(s): bio,cyber or 'all'")

    # Layout arguments
    _DIMS = "method,checkpoint,clf,probe_type,probe_source,dataset,metric"
    ap.add_argument("--group_by",     default="method",
                    help=f"Dimension(s) for x-axis groups, comma-separated. Options: {_DIMS}")
    ap.add_argument("--color_by",     default=None,
                    help=f"Dimension(s) for bar colors, comma-separated (optional). Options: {_DIMS}")
    ap.add_argument("--sort_by",      default="name",
                    choices=["name","value"],
                    help="Sort x-axis groups by name or by mean value (descending)")

    # Output arguments
    ap.add_argument("--title",        default=None,   help="Custom plot title")
    ap.add_argument("--out",          default="sweep_bars.png",
                    help="Output file (png/pdf/svg)")
    ap.add_argument("--figsize",      default="12,5",
                    help="Figure size as W,H inches (per-metric panel)")
    ap.add_argument("--show_values",  action="store_true",
                    help="Print numeric values above each bar")
    ap.add_argument("--no_chance",    action="store_true",
                    help="Suppress the 0.5 chance-level dashed line")
    ap.add_argument("--no_base",      action="store_true",
                    help="Do not prepend base model as checkpoint 0 (sweep mode only)")
    ap.add_argument("--data_dir",     default=None,
                    help="Path to data/ directory (default: auto-detect from script location)")
    ap.add_argument("--ck_dir",       default=None,
                    help="Path to checkpoints/ directory for --checkpoints latest "
                         "(default: auto-detect from script location)")

    args = ap.parse_args()

    # Resolve directories
    script_root = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else script_root / "data"
    ck_dir   = Path(args.ck_dir)   if args.ck_dir   else script_root / "checkpoints"

    figsize = tuple(float(x) for x in args.figsize.split(","))

    # ── Load & melt ──────────────────────────────────────────────────────
    use_pipeline = (args.checkpoints.strip().lower() == "latest")

    if use_pipeline:
        print(f"Loading pipeline results from {ck_dir}/…  (checkpoint 8 only)")
        if not ck_dir.exists():
            print(f"ERROR: checkpoints directory not found: {ck_dir}", file=sys.stderr)
            sys.exit(1)
        long_df = load_from_pipeline(ck_dir, args.methods)
        print(f"  Loaded {long_df['method'].nunique()} method(s) from pipeline.")
    else:
        if not data_dir.exists():
            print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading sweep CSVs from {data_dir}/sweep_*/…")
        wide_df = load_all_sweep_csvs(data_dir, args.methods)
        print(f"  Loaded {len(wide_df)} rows across {wide_df['method'].nunique()} method(s).")
        long_df = melt_sweep_df(wide_df)

    print(f"  Long-format rows: {len(long_df)}")

    # ── Discover available values ─────────────────────────────────────────
    avail = {
        "methods":       sorted(long_df["method"].unique()),
        "checkpoints":   sorted(long_df["checkpoint"].unique()),
        "clfs":          sorted(long_df["clf"].unique()),
        "probe_types":   sorted(long_df["probe_type"].unique()),
        "probe_sources": sorted(long_df["probe_source"].unique()),
        "metrics":       sorted(long_df["metric"].unique()),
        "datasets":      sorted(long_df["dataset"].unique()),
    }

    # ── Parse filter lists ────────────────────────────────────────────────
    methods     = _parse_list(args.methods, avail["methods"])
    if use_pipeline:
        checkpoints = [8]
    else:
        # Prepend base model as checkpoint 0
        if not args.no_base and ck_dir.exists():
            base_ck0 = load_base_as_ck0(ck_dir, methods)
            if not base_ck0.empty:
                long_df = pd.concat([base_ck0, long_df], ignore_index=True)
                print(f"  Added base model as checkpoint 0 ({len(base_ck0)} rows).")
        checkpoints = [int(c) for c in _parse_list(args.checkpoints, avail["checkpoints"])]
        # Always include ck0 if base was added
        if 0 not in checkpoints:
            checkpoints = [0] + checkpoints
    clfs_raw      = _parse_list(args.clf,          avail["clfs"])
    clfs          = _normalise_clf(clfs_raw)
    probe_types   = _parse_list(args.probe_type,   avail["probe_types"])
    probe_sources = _parse_list(args.probe_source, avail["probe_sources"])
    metrics       = _parse_list(args.metric,       avail["metrics"])
    datasets      = _parse_list(args.dataset,      avail["datasets"])

    # Auto-add non-probe sources when their probe_type is requested
    for pt in probe_types:
        if pt in NON_PROBE and pt not in probe_sources:
            probe_sources.append(pt)

    # ── Filter ────────────────────────────────────────────────────────────
    fdf = filter_long_df(
        long_df,
        methods=methods, checkpoints=checkpoints, clfs=clfs,
        probe_types=probe_types, probe_sources=probe_sources,
        metrics=metrics, datasets=datasets,
    )

    if fdf.empty:
        print("\nERROR: No data after filtering.", file=sys.stderr)
        print(f"  Available probe_types : {avail['probe_types']}", file=sys.stderr)
        print(f"  Available metrics     : {avail['metrics']}", file=sys.stderr)
        print(f"  Available clfs        : {avail['clfs']}", file=sys.stderr)
        sys.exit(1)

    group_by = [d.strip() for d in args.group_by.split(",")]
    color_by = [d.strip() for d in args.color_by.split(",")] if args.color_by else []

    print(f"  Rows after filter: {len(fdf)}")
    print(f"  group_by={group_by}  color_by={color_by}")
    print(f"  metrics={metrics}  probe_types={probe_types}")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_bars(
        fdf,
        group_by=group_by,
        color_by=color_by,
        metrics=metrics,
        title=args.title,
        out_path=Path(args.out),
        figsize=figsize,
        show_values=args.show_values,
        sort_by=args.sort_by,
        chance_line=None if args.no_chance else 0.5,
    )


if __name__ == "__main__":
    main()
