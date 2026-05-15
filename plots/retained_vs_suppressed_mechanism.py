#!/usr/bin/env python3
"""
Task E: Retained vs suppressed mechanism analysis.

Hypothesis: both subsets preserve internal knowledge (full-layer probe), but
suppressed shows stronger output-access collapse.

Per-question metrics:
  - K_int trajectory  (mean K_int over ck1-ck8, full-layer probe)
  - External drop     (K_ext_base - K_ext_ck8)
  - K_int - K_ext gap @ ck8  (K_int_ck8 - K_ext_ck8)

Outputs (plots/retained_vs_suppressed_mechanism/):
  retained_vs_suppressed_mechanism.csv
  retained_vs_suppressed_mechanism.md
  per_method_{method}.pdf / .png
  retained_vs_suppressed_summary_bars.pdf / .png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "plots"))

from hk_utils import (
    METHODS, load_correct_idx, load_split_indices, load_scores_df,
    compute_prefilter_mask, get_parquet_full_single, get_method_ck8_labels, method_fname,
)

TRAJ_BY_Q = REPO / "plots" / "ckpt_layer_trajectory_metrics" / "trajectory_metrics_by_method_subset.csv"
OUT_DIR = REPO / "plots" / "retained_vs_suppressed_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_RET  = "#1f77b4"
COL_SUPP = "#d62728"
SUBSETS  = ["retained", "suppressed"]

METRICS = {
    "k_int_traj":     "K_int trajectory\n(full-layer, mean ck1-ck8)",
    "ext_drop":       "External drop\n(K_ext base - K_ext ck8)",
    "int_ext_gap_ck8":"K_int - K_ext gap @ ck8",
}


def mwu_greater(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    u, p = mannwhitneyu(a, b, alternative="greater")
    return float(u), float(p)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def sig_stars(p):
    if np.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def load_per_question_table():
    traj = pd.read_csv(TRAJ_BY_Q)
    traj = traj[traj["subset"].isin(SUBSETS)].copy()
    full = traj[traj["layer_band"] == "full"][["method", "subset", "question_idx", "auc", "ck8"]].copy()
    full = full.rename(columns={"auc": "k_int_traj", "ck8": "k_int_ck8"})
    return full


def load_external_k():
    correct_idx = load_correct_idx()
    te, orig_to_pos = load_split_indices()
    df = load_scores_df()
    parquet_full = get_parquet_full_single(df)
    filt_mask = compute_prefilter_mask(te, orig_to_pos, df)
    filt_te = te[filt_mask]
    filt_o2p = {int(qi): i for i, qi in enumerate(filt_te)}

    base_ext = parquet_full[parquet_full["model_id"] == "base"][["question_idx", "k_external"]]
    base_ext = base_ext.rename(columns={"k_external": "k_ext_base"})

    rows = []
    for method in METHODS:
        labels, k_ext_ck8_arr, _ = get_method_ck8_labels(
            method, correct_idx, filt_te, filt_o2p, parquet_full
        )
        if labels is None:
            continue
        for qi, kext in zip(filt_te, k_ext_ck8_arr):
            rows.append({"method": method, "question_idx": int(qi), "k_ext_ck8": float(kext)})

    ext_ck8 = pd.DataFrame(rows)
    ext = ext_ck8.merge(base_ext, on="question_idx", how="left")
    ext["ext_drop"] = ext["k_ext_base"] - ext["k_ext_ck8"]
    return ext


def compute_stats(per_q):
    stat_rows = []
    for method in METHODS:
        m = per_q[per_q["method"] == method]
        ret  = m[m["subset"] == "retained"]
        supp = m[m["subset"] == "suppressed"]

        for metric, label in METRICS.items():
            ra = ret[metric].dropna().to_numpy()
            sa = supp[metric].dropna().to_numpy()
            alt = "greater" if metric in ("ext_drop", "int_ext_gap_ck8") else "less"
            u, p = (mannwhitneyu(sa, ra, alternative=alt)
                    if len(sa) >= 2 and len(ra) >= 2 else (np.nan, np.nan))
            stat_rows.append({
                "method": method,
                "metric": metric,
                "n_retained":   len(ra),
                "n_suppressed": len(sa),
                "mean_retained":   round(float(ra.mean()), 4) if len(ra) else np.nan,
                "mean_suppressed": round(float(sa.mean()), 4) if len(sa) else np.nan,
                "diff_supp_minus_ret": round(float(sa.mean() - ra.mean()), 4) if len(sa) and len(ra) else np.nan,
                "cohens_d": round(cohens_d(sa, ra), 3) if len(sa) >= 2 and len(ra) >= 2 else np.nan,
                "p_value": float(p) if not np.isnan(p) else np.nan,
                "sig": sig_stars(float(p)) if not np.isnan(p) else "",
            })
    return pd.DataFrame(stat_rows)


def box_pair(ax, ret_vals, supp_vals, title, ylabel=None):
    bp = ax.boxplot(
        [ret_vals, supp_vals],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="white", lw=2),
        whiskerprops=dict(lw=1.2),
        capprops=dict(lw=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    bp["boxes"][0].set_facecolor(COL_RET)
    bp["boxes"][1].set_facecolor(COL_SUPP)
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_alpha(0.75)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["retained", "suppressed"], fontsize=8)
    ax.set_title(title, fontsize=8, pad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _, p = mwu_greater(supp_vals, ret_vals)
    stars = sig_stars(p)
    y_top = max(np.percentile(ret_vals, 95) if len(ret_vals) else 0,
                np.percentile(supp_vals, 95) if len(supp_vals) else 0)
    ax.text(1.5, y_top * 1.02 + 0.02, stars, ha="center", va="bottom", fontsize=9,
            color="black" if stars != "ns" else "#888888")


def plot_per_method(per_q):
    metric_keys = list(METRICS.keys())
    metric_labels = list(METRICS.values())

    for method in METHODS:
        m = per_q[per_q["method"] == method]
        ret  = m[m["subset"] == "retained"]
        supp = m[m["subset"] == "suppressed"]

        fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
        fig.suptitle(
            f"{method}  —  retained (n={len(ret)}) vs suppressed (n={len(supp)})",
            fontsize=11, y=1.01,
        )

        for ax, key, lbl in zip(axes, metric_keys, metric_labels):
            ra = ret[key].dropna().to_numpy()
            sa = supp[key].dropna().to_numpy()
            box_pair(ax, ra, sa, lbl)

        fig.tight_layout()
        stem = OUT_DIR / f"per_method_{method_fname(method)}"
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=180)
        plt.close(fig)


def plot_summary_bars(stats_df):
    metric_keys = list(METRICS.keys())
    metric_labels_short = ["K_int traj (full)", "Ext drop", "K_int-K_ext gap @ ck8"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=False)
    fig.suptitle("Retained vs Suppressed mechanism — mean difference (suppressed − retained)",
                 fontsize=11)

    for ax, key, lbl in zip(axes, metric_keys, metric_labels_short):
        sub = stats_df[stats_df["metric"] == key].copy()
        sub = sub[sub["method"].isin(METHODS)].copy()
        sub = sub.sort_values("diff_supp_minus_ret", ascending=True)

        colors = [COL_SUPP if d > 0 else COL_RET for d in sub["diff_supp_minus_ret"]]
        y = np.arange(len(sub))
        ax.barh(y, sub["diff_supp_minus_ret"], color=colors, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["method"], fontsize=9)
        ax.set_title(lbl, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for i, (_, row) in enumerate(sub.iterrows()):
            stars = row["sig"]
            if stars and stars != "ns":
                x = row["diff_supp_minus_ret"]
                offset = 0.003 if x >= 0 else -0.003
                ha = "left" if x >= 0 else "right"
                ax.text(x + offset, y[i], stars, va="center", ha=ha,
                        fontsize=8, color="black", fontweight="bold")

    fig.tight_layout()
    stem = OUT_DIR / "retained_vs_suppressed_summary_bars"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_md(stats_df):
    lines = [
        "# Retained vs Suppressed — Mechanism Analysis",
        "",
        "Hypothesis: both subsets preserve internal K_int (full-layer probe), but suppressed "
        "shows stronger output-access collapse.",
        "",
        "p-values: one-sided Mann-Whitney U (direction: suppressed > retained for "
        "ext_drop and int_ext_gap; retained > suppressed for k_int_traj).",
        "",
    ]
    for metric, label in METRICS.items():
        lines.append(f"## {label.replace(chr(10), ' ')}")
        lines.append("")
        sub = stats_df[stats_df["metric"] == metric][[
            "method", "n_retained", "n_suppressed",
            "mean_retained", "mean_suppressed", "diff_supp_minus_ret",
            "cohens_d", "p_value", "sig"
        ]]
        header = "| " + " | ".join(sub.columns) + " |"
        sep    = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
        lines.append(header)
        lines.append(sep)
        for _, r in sub.iterrows():
            lines.append("| " + " | ".join(str(round(v, 4)) if isinstance(v, float) else str(v) for v in r) + " |")
        lines.append("")

    (OUT_DIR / "retained_vs_suppressed_mechanism.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    print("Loading trajectory data...")
    per_q = load_per_question_table()

    print("Loading external K data...")
    ext = load_external_k()

    per_q = per_q.merge(ext[["method", "question_idx", "k_ext_base", "k_ext_ck8", "ext_drop"]],
                        on=["method", "question_idx"], how="left")
    per_q["int_ext_gap_ck8"] = per_q["k_int_ck8"] - per_q["k_ext_ck8"]

    print("Computing stats...")
    stats_df = compute_stats(per_q)
    stats_df.to_csv(OUT_DIR / "retained_vs_suppressed_mechanism.csv", index=False)

    print("Writing markdown report...")
    write_md(stats_df)

    print("Plotting per-method figures...")
    plot_per_method(per_q)

    print("Plotting summary bars...")
    plot_summary_bars(stats_df)

    print(f"\nSaved all outputs to {OUT_DIR}")

    print("\n-- Key means by metric --")
    for metric in METRICS:
        sub = stats_df[stats_df["metric"] == metric][
            ["method", "mean_retained", "mean_suppressed", "diff_supp_minus_ret", "sig"]
        ]
        print(f"\n{metric}")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
