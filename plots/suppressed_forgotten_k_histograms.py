#!/usr/bin/env python3
"""
Suppressed vs Forgotten K-distribution histograms.

Run from repo root:
  python plots/suppressed_forgotten_k_histograms.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover
    mannwhitneyu = None

from hk_utils import REPO, EXT_DIR, METHODS, method_fname, load_correct_idx, compute_k_ext


OUT_DIR = REPO / "plots" / "suppressed_forgotten_k_histograms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAJ_BY_Q = REPO / "plots" / "ckpt_layer_trajectory_metrics" / "trajectory_metrics_by_method_subset.csv"
STATS_PATH = REPO / "plots" / "ckpt_layer_trajectory_metrics" / "suppressed_vs_forgotten_stats.csv"
POOLED_FEATURES = REPO / "plots" / "base_feature_prediction" / "pooled_features.csv"

BINS = np.linspace(0.0, 1.0, 21)
ALPHA = 0.5
COL_SUP = "#d62728"
COL_FOR = "#7f7f7f"


def load_mid_auc_df():
    if not TRAJ_BY_Q.exists():
        raise FileNotFoundError(f"Missing trajectory file: {TRAJ_BY_Q}")
    df = pd.read_csv(TRAJ_BY_Q)
    needed = {"method", "subset", "question_idx", "layer_band"}
    if not needed.issubset(set(df.columns)):
        raise KeyError(f"Missing required columns in {TRAJ_BY_Q}. Found: {sorted(df.columns)}")
    score_col = "auc" if "auc" in df.columns else "mid_layer_internal_trajectory_auc"
    sub = df[df["layer_band"] == "mid"].copy()
    sub = sub.rename(columns={score_col: "mid_auc"})
    sub = sub[["method", "subset", "question_idx", "mid_auc"]].copy()
    return sub


def load_stats_lookup():
    if not STATS_PATH.exists():
        return {}
    sdf = pd.read_csv(STATS_PATH)
    if "metric" in sdf.columns:
        sdf = sdf[sdf["metric"] == "mid_auc"].copy()
    out = {}
    for _, r in sdf.iterrows():
        out[str(r["method"])] = {
            "p_value": float(r["p_value_greater"]) if "p_value_greater" in sdf.columns else np.nan,
            "delta": float(r["mean_diff_supp_minus_forg"]) if "mean_diff_supp_minus_forg" in sdf.columns else np.nan,
            "n_suppressed": int(r["n_suppressed"]) if "n_suppressed" in sdf.columns else None,
            "n_forgotten": int(r["n_forgotten"]) if "n_forgotten" in sdf.columns else None,
        }
    return out


def compute_p_if_needed(s_vals, f_vals):
    if mannwhitneyu is None or len(s_vals) < 2 or len(f_vals) < 2:
        return np.nan
    return float(mannwhitneyu(s_vals, f_vals, alternative="greater").pvalue)


def hist_overlap_coefficient(s_vals, f_vals):
    hs, _ = np.histogram(s_vals, bins=BINS, density=True)
    hf, _ = np.histogram(f_vals, bins=BINS, density=True)
    bw = BINS[1] - BINS[0]
    return float(np.sum(np.minimum(hs, hf)) * bw)


def count_peaks(vals):
    h, _ = np.histogram(vals, bins=BINS, density=True)
    if len(h) < 3:
        return 0
    peaks = 0
    thr = 0.12 * np.nanmax(h) if np.nanmax(h) > 0 else 0.0
    for i in range(1, len(h) - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] >= thr:
            peaks += 1
    return peaks


def classify_pattern(s_vals, f_vals):
    delta = float(np.mean(s_vals) - np.mean(f_vals))
    overlap = hist_overlap_coefficient(s_vals, f_vals)
    q50 = float(np.median(s_vals) - np.median(f_vals))
    q90 = float(np.quantile(s_vals, 0.9) - np.quantile(f_vals, 0.9))
    s_peaks = count_peaks(s_vals)
    if s_peaks >= 2:
        return "bimodal/mixed"
    if delta >= 0.08 and overlap <= 0.70:
        return "broad distributional shift"
    if delta > 0 and q50 < 0.04 and q90 >= 0.10:
        return "tail-driven separation"
    return "heavy overlap"


def add_hist_common(ax, s_vals, f_vals, title, xlab, p_val=np.nan):
    s_mean = float(np.mean(s_vals))
    f_mean = float(np.mean(f_vals))
    delta = s_mean - f_mean

    ax.hist(s_vals, bins=BINS, density=True, alpha=ALPHA, color=COL_SUP, label=f"suppressed (n={len(s_vals)})")
    ax.hist(f_vals, bins=BINS, density=True, alpha=ALPHA, color=COL_FOR, label=f"forgotten (n={len(f_vals)})")
    ax.axvline(s_mean, linestyle="--", linewidth=1.5, color=COL_SUP)
    ax.axvline(f_mean, linestyle="--", linewidth=1.5, color=COL_FOR)

    txt = (
        f"n_suppressed = {len(s_vals)}\n"
        f"n_forgotten = {len(f_vals)}\n"
        f"Δ = {delta:.3f}\n"
        f"p = {p_val:.3g}" if np.isfinite(p_val) else
        f"n_suppressed = {len(s_vals)}\n"
        f"n_forgotten = {len(f_vals)}\n"
        f"Δ = {delta:.3f}\n"
        "p = n/a"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", alpha=0.15),
    )
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Question density")
    ax.legend(fontsize=8)


def load_external_ck8_lookup():
    """
    Returns dict[(method, question_idx)] -> external_k_ck8
    Uses pooled_features if present; otherwise computes from ext arrays.
    """
    out = {}
    if POOLED_FEATURES.exists():
        pdf = pd.read_csv(POOLED_FEATURES)
        ext_col = None
        for c in ["ck8_external_k", "k_external_ck8", "k_external"]:
            if c in pdf.columns:
                ext_col = c
                break
        if ext_col is not None and {"method", "question_idx"}.issubset(set(pdf.columns)):
            for _, r in pdf[["method", "question_idx", ext_col]].dropna().iterrows():
                out[(str(r["method"]), int(r["question_idx"]))] = float(r[ext_col])
            if out:
                return out

    correct_idx = load_correct_idx()
    for method in METHODS:
        mfile = method_fname(method)
        p = EXT_DIR / f"{mfile}_ck8_bio_ext.npy"
        if not p.exists():
            continue
        ext = np.load(p)
        qidx = np.arange(ext.shape[0], dtype=int)
        k = compute_k_ext(ext, correct_idx, qidx)
        for qi, kv in zip(qidx, k):
            out[(method, int(qi))] = float(kv)
    return out


def plot_method_hist(df_m, method, stats_lookup, ext_lookup):
    sub = df_m[df_m["subset"].isin(["suppressed", "forgotten"])].copy()
    s = sub[sub["subset"] == "suppressed"]["mid_auc"].dropna().to_numpy()
    f = sub[sub["subset"] == "forgotten"]["mid_auc"].dropna().to_numpy()
    if len(s) == 0 or len(f) == 0:
        return None

    st = stats_lookup.get(method, {})
    p_val = st.get("p_value", np.nan)
    if not np.isfinite(p_val):
        p_val = compute_p_if_needed(s, f)

    fig, ax = plt.subplots(figsize=(7, 5))
    add_hist_common(
        ax,
        s,
        f,
        f"{method}: Suppressed vs Forgotten — Mid-layer Internal Trajectory AUC",
        "Mid-layer internal K trajectory AUC",
        p_val=p_val,
    )
    fig.tight_layout()
    stem = OUT_DIR / f"{method_fname(method)}_mid_auc_hist"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Optional 2-panel with external K at ck8
    sub = sub.copy()
    sub["external_k_ck8"] = [ext_lookup.get((method, int(qi)), np.nan) for qi in sub["question_idx"].tolist()]
    s_ext = sub[sub["subset"] == "suppressed"]["external_k_ck8"].dropna().to_numpy()
    f_ext = sub[sub["subset"] == "forgotten"]["external_k_ck8"].dropna().to_numpy()
    if len(s_ext) > 0 and len(f_ext) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        add_hist_common(
            axes[0],
            s,
            f,
            "Mid-layer internal trajectory AUC",
            "Mid-layer internal K trajectory AUC",
            p_val=p_val,
        )
        add_hist_common(
            axes[1],
            s_ext,
            f_ext,
            "External K at ck8",
            "External K at ck8",
            p_val=np.nan,
        )
        fig.suptitle(f"{method}: Internal Survival vs External Suppression", fontsize=12)
        fig.tight_layout()
        fig.subplots_adjust(top=0.86)
        stem2 = OUT_DIR / f"{method_fname(method)}_internal_external_2panel"
        fig.savefig(f"{stem2}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem2}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    return {
        "method": method,
        "n_suppressed": int(len(s)),
        "n_forgotten": int(len(f)),
        "mean_suppressed": float(np.mean(s)),
        "mean_forgotten": float(np.mean(f)),
        "delta": float(np.mean(s) - np.mean(f)),
        "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        "pattern": classify_pattern(s, f),
        "overlap_coef": hist_overlap_coefficient(s, f),
    }


def plot_pooled(df):
    sub = df[df["subset"].isin(["suppressed", "forgotten"])].copy()
    s = sub[sub["subset"] == "suppressed"]["mid_auc"].dropna().to_numpy()
    f = sub[sub["subset"] == "forgotten"]["mid_auc"].dropna().to_numpy()
    if len(s) == 0 or len(f) == 0:
        return
    p_val = compute_p_if_needed(s, f)
    fig, ax = plt.subplots(figsize=(7, 5))
    add_hist_common(
        ax,
        s,
        f,
        "Pooled: Suppressed vs Forgotten — Mid-layer Internal Trajectory AUC",
        "Mid-layer internal K trajectory AUC",
        p_val=p_val,
    )
    fig.tight_layout()
    stem = OUT_DIR / "pooled_mid_auc_hist"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows):
    lines = ["# Suppressed vs Forgotten Histogram Summary", ""]
    if not rows:
        lines.append("- No methods had both suppressed and forgotten rows.")
        (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        return
    lines.append("| Method | n_suppressed | n_forgotten | mean_suppressed | mean_forgotten | Δ | p-value | Pattern |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        ptxt = f"{r['p_value']:.3g}" if np.isfinite(r["p_value"]) else "n/a"
        lines.append(
            f"| {r['method']} | {r['n_suppressed']} | {r['n_forgotten']} | "
            f"{r['mean_suppressed']:.3f} | {r['mean_forgotten']:.3f} | "
            f"{r['delta']:.3f} | {ptxt} | {r['pattern']} |"
        )
    lines.append("")
    lines.append("Pattern labels used:")
    lines.append("- broad distributional shift")
    lines.append("- tail-driven separation")
    lines.append("- bimodal/mixed")
    lines.append("- heavy overlap")
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    df = load_mid_auc_df()
    stats_lookup = load_stats_lookup()
    ext_lookup = load_external_ck8_lookup()

    rows = []
    for method in METHODS:
        df_m = df[df["method"] == method].copy()
        if df_m.empty:
            continue
        row = plot_method_hist(df_m, method, stats_lookup, ext_lookup)
        if row is not None:
            rows.append(row)

    # pooled
    plot_pooled(df)

    # summary
    write_summary(rows)
    print("Saved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
