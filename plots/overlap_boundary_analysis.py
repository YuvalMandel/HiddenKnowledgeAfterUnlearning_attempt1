#!/usr/bin/env python3
"""
Overlap / "coin-toss boundary" analysis between suppressed and forgotten questions.

Defines, per method, an overlap interval in mid-layer internal-trajectory AUC:
  overlap_low  = max(Q10_suppressed, Q10_forgotten)
  overlap_high = min(Q90_suppressed, Q90_forgotten)

Then, within that overlap subset, compares suppressed vs forgotten on:
  - final hidden-knowledge gap (K_internal_ck8 - K_external_ck8)
  - final internal/external K
  - mid/late trajectory deltas and shapes (when present)

Run from repo root:
  python plots/overlap_boundary_analysis.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover
    mannwhitneyu = None

from hk_utils import (
    METHODS,
    REPO,
    compute_k_ext,
    compute_prefilter_mask,
    get_parquet_multi_single,
    get_method_ck8_labels,
    load_correct_idx,
    load_scores_df,
    load_split_indices,
    method_fname,
)


IN_TRAJ_BY_Q = REPO / "plots" / "ckpt_layer_trajectory_metrics" / "trajectory_metrics_by_method_subset.csv"
OUT_DIR = REPO / "plots" / "overlap_boundary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"suppressed": "#d62728", "forgotten": "#7f7f7f"}


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = df.astype(object).where(pd.notnull(df), "").values.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join([header, sep] + body) + "\n"


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = np.nanvar(a, ddof=1)
    vb = np.nanvar(b, ddof=1)
    if not np.isfinite(va) or not np.isfinite(vb):
        return np.nan
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if sp == 0:
        return np.nan
    return float((np.nanmean(a) - np.nanmean(b)) / sp)


def _mw_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    if mannwhitneyu is None or len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def load_trajectory_wide() -> pd.DataFrame:
    if not IN_TRAJ_BY_Q.exists():
        raise FileNotFoundError(
            f"Missing input: {IN_TRAJ_BY_Q}\n"
            "Run: python plots/quantify_ckpt_layer_trajectories.py"
        )
    df = pd.read_csv(IN_TRAJ_BY_Q)
    needed = {"method", "subset", "layer_band", "question_idx"}
    if not needed.issubset(set(df.columns)):
        raise KeyError(f"Missing required columns in {IN_TRAJ_BY_Q}. Found: {sorted(df.columns)}")

    score_col = "auc" if "auc" in df.columns else "mid_layer_internal_trajectory_auc"
    df = df.rename(columns={score_col: "auc"})

    metrics = ["auc", "base", "ck8", "delta_ck8_minus_base", "slope_per_ckpt", "min_checkpoint_value"]
    present_metrics = [m for m in metrics if m in df.columns]
    long = df[["method", "subset", "question_idx", "layer_band"] + present_metrics].copy()

    wide = long.pivot_table(
        index=["method", "subset", "question_idx"],
        columns="layer_band",
        values=present_metrics,
        aggfunc="first",
    )
    wide.columns = [f"{lb}_{m}" for m, lb in wide.columns.to_list()]
    wide = wide.reset_index()

    # Convenience aliases used by the task brief.
    rename_map = {
        "mid_auc": "mid_layer_internal_trajectory_auc",
        "mid_ck8": "ck8_mid_k_internal",
        "late_ck8": "ck8_late_k_internal",
        "late_base": "base_late_k_internal",
    }
    for src, dst in rename_map.items():
        if src in wide.columns and dst not in wide.columns:
            wide = wide.rename(columns={src: dst})
    return wide


def load_ck8_k_by_question() -> pd.DataFrame:
    correct_idx = load_correct_idx()
    te, orig_to_pos = load_split_indices()
    scores_df = load_scores_df()
    parquet_multi_single = get_parquet_multi_single(scores_df)

    filt_mask = compute_prefilter_mask(te, orig_to_pos, scores_df)
    filt_te = te[filt_mask]
    filt_orig_to_pos = {int(qi): i for i, qi in enumerate(filt_te)}

    all_rows = []
    for method in METHODS:
        labels, mk_int, _k_int_source = get_method_ck8_labels(
            method, correct_idx, filt_te, filt_orig_to_pos, parquet_multi_single
        )
        if labels is None:
            continue
        ext_path = REPO / "inside_out_ext" / f"{method_fname(method)}_ck8_bio_ext.npy"
        if not ext_path.exists():
            continue
        mk_ext = compute_k_ext(np.load(ext_path), correct_idx, filt_te)
        df = pd.DataFrame(
            {
                "method": method,
                "question_idx": filt_te.astype(int),
                "subset": labels.astype(str),
                "K_internal_ck8": mk_int.astype(np.float64),
                "K_external_ck8": mk_ext.astype(np.float64),
            }
        )
        df["HKGap"] = df["K_internal_ck8"] - df["K_external_ck8"]
        # Pre-filter implies base external and internal are 1.0.
        df["K_external_base"] = 1.0
        df["external_drop"] = df["K_external_base"] - df["K_external_ck8"]
        all_rows.append(df)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if out.empty:
        raise RuntimeError("No ck8 K rows computed; check ck8 ext files and parquet inputs.")
    return out


@dataclass(frozen=True)
class OverlapInterval:
    method: str
    overlap_low: float
    overlap_high: float
    n_suppressed_total: int
    n_forgotten_total: int
    n_suppressed_overlap: int
    n_forgotten_overlap: int


def compute_overlap_intervals(df: pd.DataFrame, mid_auc_col: str) -> pd.DataFrame:
    rows: list[OverlapInterval] = []
    for method in METHODS:
        sub = df[(df["method"] == method) & (df["subset"].isin(["suppressed", "forgotten"]))].copy()
        if sub.empty or mid_auc_col not in sub.columns:
            continue
        s = sub[sub["subset"] == "suppressed"][mid_auc_col].dropna().to_numpy(dtype=np.float64)
        f = sub[sub["subset"] == "forgotten"][mid_auc_col].dropna().to_numpy(dtype=np.float64)
        if len(s) < 5 or len(f) < 5:
            continue
        q10s, q90s = float(np.quantile(s, 0.10)), float(np.quantile(s, 0.90))
        q10f, q90f = float(np.quantile(f, 0.10)), float(np.quantile(f, 0.90))
        low = max(q10s, q10f)
        high = min(q90s, q90f)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = np.nan, np.nan
            s_in = 0
            f_in = 0
        else:
            s_in = int(np.sum((s >= low) & (s <= high)))
            f_in = int(np.sum((f >= low) & (f <= high)))
        rows.append(
            OverlapInterval(
                method=method,
                overlap_low=low,
                overlap_high=high,
                n_suppressed_total=int(len(s)),
                n_forgotten_total=int(len(f)),
                n_suppressed_overlap=s_in,
                n_forgotten_overlap=f_in,
            )
        )
    out = pd.DataFrame([r.__dict__ for r in rows])
    if out.empty:
        raise RuntimeError("No overlap intervals computed; check mid AUC availability and subset labels.")
    out["overlap_width"] = out["overlap_high"] - out["overlap_low"]
    return out


def plot_mid_auc_vs_hkgap_by_method(df: pd.DataFrame, intervals: pd.DataFrame, mid_auc_col: str) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 12.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, method in enumerate(METHODS):
        ax = axes[i]
        sub = df[(df["method"] == method) & (df["subset"].isin(["suppressed", "forgotten"]))].copy()
        if sub.empty:
            ax.set_title(f"{method} (no data)", fontsize=10)
            ax.axis("off")
            continue

        row = intervals[intervals["method"] == method]
        if not row.empty and np.isfinite(float(row["overlap_low"].iloc[0])) and np.isfinite(float(row["overlap_high"].iloc[0])):
            low = float(row["overlap_low"].iloc[0])
            high = float(row["overlap_high"].iloc[0])
            ax.axvspan(low, high, color="#bbbbbb", alpha=0.22, linewidth=0)

        for subset in ["suppressed", "forgotten"]:
            ssub = sub[sub["subset"] == subset]
            ax.scatter(
                ssub[mid_auc_col].to_numpy(dtype=np.float64),
                ssub["HKGap"].to_numpy(dtype=np.float64),
                s=12,
                alpha=0.55,
                color=COLORS[subset],
                edgecolors="none",
                rasterized=True,
            )
        ax.axhline(0.0, color="#222222", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(method, fontsize=10)
        ax.grid(True, linewidth=0.35, alpha=0.25)

    for ax in axes[-2:]:
        ax.set_xlabel("mid-layer internal trajectory AUC")
    for ax in axes[::2]:
        ax.set_ylabel("HKGap (K_internal_ck8 − K_external_ck8)")
    fig.suptitle("Mid-layer AUC vs HKGap (suppressed vs forgotten)", fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mid_auc_vs_hkgap_by_method.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT_DIR / "mid_auc_vs_hkgap_by_method.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_boundary_hkgap_strip(df: pd.DataFrame, intervals: pd.DataFrame, mid_auc_col: str) -> None:
    # Build per-method overlap-only dataset.
    rows = []
    for _, r in intervals.iterrows():
        method = str(r["method"])
        low = r["overlap_low"]
        high = r["overlap_high"]
        if not np.isfinite(low) or not np.isfinite(high):
            continue
        sub = df[(df["method"] == method) & (df["subset"].isin(["suppressed", "forgotten"]))].copy()
        sub = sub[(sub[mid_auc_col] >= low) & (sub[mid_auc_col] <= high)].copy()
        if sub.empty:
            continue
        rows.append(sub)
    over = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if over.empty:
        return

    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    x_positions = {m: i for i, m in enumerate(METHODS)}

    rng = np.random.default_rng(42)
    for subset, dx in [("suppressed", -0.18), ("forgotten", 0.18)]:
        ssub = over[over["subset"] == subset].copy()
        xs = np.array([x_positions[m] for m in ssub["method"]], dtype=np.float64)
        xs = xs + dx + rng.uniform(-0.06, 0.06, size=len(xs))
        ax.scatter(xs, ssub["HKGap"].to_numpy(dtype=np.float64), s=14, alpha=0.55, color=COLORS[subset], edgecolors="none")

    ax.axhline(0.0, color="#222222", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=0, fontsize=9)
    ax.set_ylabel("HKGap (overlap subset)")
    ax.set_title("HKGap within mid-AUC overlap interval (strip)", fontsize=11)
    ax.grid(True, axis="y", linewidth=0.35, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "boundary_hkgap_strip_by_method.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT_DIR / "boundary_hkgap_strip_by_method.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overlap_diff_heatmap(method: str, summary_rows: pd.DataFrame) -> None:
    # summary_rows filtered to one method, one row per metric
    if summary_rows.empty:
        return
    metrics = summary_rows["metric"].tolist()
    diffs = summary_rows["mean_suppressed"].to_numpy(dtype=np.float64) - summary_rows["mean_forgotten"].to_numpy(dtype=np.float64)
    ds = summary_rows["cohen_d"].to_numpy(dtype=np.float64)

    mat = np.vstack([diffs, ds])
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(metrics)), 2.6))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-np.nanmax(np.abs(mat)), vmax=np.nanmax(np.abs(mat)))

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["mean diff", "Cohen d"])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{method}: suppressed − forgotten (within overlap)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    fig.tight_layout()
    safe = method_fname(method)
    fig.savefig(OUT_DIR / f"overlap_diff_heatmap_{safe}.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"overlap_diff_heatmap_{safe}.pdf", bbox_inches="tight")
    plt.close(fig)


def boundary_comparison_tables(df: pd.DataFrame, intervals: pd.DataFrame, mid_auc_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_pref = [
        ("HKGap", "HKGap"),
        ("K_internal_ck8", "K_internal_ck8"),
        ("K_external_ck8", "K_external_ck8"),
        ("external_drop", "external_drop"),
        ("mid_delta_ck8_minus_base", "mid_delta_ck8_minus_base"),
        ("late_delta_ck8_minus_base", "late_delta_ck8_minus_base"),
        ("mid_ck8", "mid_ck8"),
        ("late_ck8", "late_ck8"),
        ("mid_slope_per_ckpt", "mid_slope_per_ckpt"),
        ("late_slope_per_ckpt", "late_slope_per_ckpt"),
        ("mid_min_checkpoint_value", "mid_min_checkpoint_value"),
        ("late_min_checkpoint_value", "late_min_checkpoint_value"),
    ]

    rows = []
    for _, r in intervals.iterrows():
        method = str(r["method"])
        low = r["overlap_low"]
        high = r["overlap_high"]
        if not np.isfinite(low) or not np.isfinite(high):
            continue
        sub = df[(df["method"] == method) & (df["subset"].isin(["suppressed", "forgotten"]))].copy()
        sub = sub[(sub[mid_auc_col] >= low) & (sub[mid_auc_col] <= high)].copy()
        if sub.empty:
            continue

        for public_name, col in metrics_pref:
            if col not in sub.columns:
                continue
            s = sub[sub["subset"] == "suppressed"][col].dropna().to_numpy(dtype=np.float64)
            f = sub[sub["subset"] == "forgotten"][col].dropna().to_numpy(dtype=np.float64)
            if len(s) < 2 or len(f) < 2:
                continue
            rows.append(
                {
                    "method": method,
                    "metric": public_name,
                    "n_suppressed": int(len(s)),
                    "n_forgotten": int(len(f)),
                    "mean_suppressed": float(np.mean(s)),
                    "mean_forgotten": float(np.mean(f)),
                    "diff_supp_minus_forg": float(np.mean(s) - np.mean(f)),
                    "cohen_d": _cohen_d(s, f),
                    "p_value_mwu_2sided": _mw_pvalue(s, f),
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    # A compact method-level summary: pick strongest |d| metric per method.
    best_rows = []
    for method in METHODS:
        m = summary[summary["method"] == method].copy()
        if m.empty:
            continue
        m = m[np.isfinite(m["cohen_d"])].copy()
        if m.empty:
            continue
        m = m.sort_values("cohen_d", key=lambda s: s.abs(), ascending=False)
        best = m.iloc[0].to_dict()
        best_rows.append(best)
    best_df = pd.DataFrame(best_rows)
    return summary, best_df


def main() -> None:
    traj = load_trajectory_wide()
    kdf = load_ck8_k_by_question()

    # Merge: prefer trajectory subset labels, but keep K-derived labels as a check.
    merged = traj.merge(
        kdf[["method", "subset", "question_idx", "K_internal_ck8", "K_external_ck8", "HKGap", "external_drop"]],
        on=["method", "subset", "question_idx"],
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("Trajectory/K merge is empty; check that subset labels and question_idx match.")

    mid_auc_col = "mid_layer_internal_trajectory_auc" if "mid_layer_internal_trajectory_auc" in merged.columns else "mid_auc"
    if mid_auc_col not in merged.columns:
        raise KeyError("Could not find a mid AUC column in merged data.")

    intervals = compute_overlap_intervals(merged, mid_auc_col=mid_auc_col)
    intervals = intervals.sort_values("method").reset_index(drop=True)

    intervals.to_csv(OUT_DIR / "overlap_region_counts.csv", index=False)

    # Boundary comparisons.
    summary, best = boundary_comparison_tables(merged, intervals, mid_auc_col=mid_auc_col)
    summary.to_csv(OUT_DIR / "overlap_boundary_summary.csv", index=False)
    (OUT_DIR / "overlap_boundary_summary.md").write_text(
        "# Overlap Boundary Summary (within overlap interval)\n\n"
        + _df_to_markdown_table(summary.round(6)) if not summary.empty else "# Overlap Boundary Summary\n\n(no rows)\n",
        encoding="utf-8",
    )

    # Visuals.
    plot_mid_auc_vs_hkgap_by_method(merged, intervals, mid_auc_col=mid_auc_col)
    plot_boundary_hkgap_strip(merged, intervals, mid_auc_col=mid_auc_col)
    if not summary.empty:
        for method in METHODS:
            plot_overlap_diff_heatmap(method, summary_rows=summary[summary["method"] == method].copy())

    # Human-readable summary.
    lines = [
        "# Overlap Boundary Analysis Summary",
        "",
        "Overlap definition: overlap_low=max(Q10_supp, Q10_forg); overlap_high=min(Q90_supp, Q90_forg).",
        "",
        "## Counts",
        "",
        _df_to_markdown_table(intervals.round(6)),
        "",
        "## Strongest metric inside overlap (by |Cohen d|)",
        "",
    ]
    if best.empty:
        lines.append("(no overlap-boundary comparisons available)\n")
    else:
        lines.append(_df_to_markdown_table(best.round(6)))
        lines.append("")
        lines.append("Interpretation hint: positive `diff_supp_minus_forg` means suppressed > forgotten.")
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("Saved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
