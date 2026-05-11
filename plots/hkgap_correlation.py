#!/usr/bin/env python3
"""
Method-level correlation between:
  (A) suppressed-minus-forgotten mid-layer internal-trajectory AUC gap
and several variants of the final hidden-knowledge gap after unlearning.

Run from repo root:
  python plots/hkgap_correlation.py
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
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover
    pearsonr = None
    spearmanr = None

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
OUT_DIR = REPO / "plots" / "hkgap_correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rankdata_ties_avg(x: np.ndarray) -> np.ndarray:
    """Minimal rankdata with average ranks for ties; 1-based ranks."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return np.nan
    if pearsonr is not None:
        return float(pearsonr(x, y).statistic)
    # fallback: numpy corrcoef
    c = np.corrcoef(x, y)
    return float(c[0, 1])


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return np.nan
    if spearmanr is not None:
        return float(spearmanr(x, y).statistic)
    rx = _rankdata_ties_avg(x)
    ry = _rankdata_ties_avg(y)
    return _pearsonr(rx, ry)


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = df.astype(object).where(pd.notnull(df), "").values.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join([header, sep] + body) + "\n"


def load_mid_auc_by_question() -> pd.DataFrame:
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
    sub = df[df["layer_band"] == "mid"].copy()
    sub = sub.rename(columns={score_col: "mid_auc"})
    return sub[["method", "subset", "question_idx", "mid_auc"]].copy()


@dataclass(frozen=True)
class MethodPoint:
    method: str
    n_total: int
    n_suppressed: int
    n_forgotten: int
    prop_suppressed: float
    mean_mid_auc_suppressed: float
    mean_mid_auc_forgotten: float
    mid_auc_gap: float
    mean_hkgap_all: float
    mean_hkgap_suppressed: float
    mean_hkgap_forgotten: float
    hkgap_suppressed_minus_forgotten: float
    hidden_knowledge_mass: float
    mean_k_internal_ck8_all: float
    mean_k_external_ck8_all: float
    mean_k_internal_ck8_suppressed: float
    mean_k_external_ck8_suppressed: float
    mean_k_internal_ck8_forgotten: float
    mean_k_external_ck8_forgotten: float


def compute_method_points(mid_auc_df: pd.DataFrame) -> pd.DataFrame:
    correct_idx = load_correct_idx()
    te, orig_to_pos = load_split_indices()
    scores_df = load_scores_df()
    parquet_multi_single = get_parquet_multi_single(scores_df)

    filt_mask = compute_prefilter_mask(te, orig_to_pos, scores_df)
    filt_te = te[filt_mask]
    filt_orig_to_pos = {int(qi): i for i, qi in enumerate(filt_te)}

    rows: list[MethodPoint] = []
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
        hkgap = mk_int.astype(np.float64) - mk_ext.astype(np.float64)

        per_q = pd.DataFrame(
            {
                "method": method,
                "question_idx": filt_te.astype(int),
                "subset": labels.astype(str),
                "k_internal_ck8": mk_int.astype(np.float64),
                "k_external_ck8": mk_ext.astype(np.float64),
                "hkgap": hkgap.astype(np.float64),
            }
        )

        m_mid = mid_auc_df[mid_auc_df["method"] == method].copy()
        if m_mid.empty:
            continue
        m_mid = m_mid[["question_idx", "mid_auc", "subset"]].copy()
        merged = per_q.merge(m_mid, on=["question_idx", "subset"], how="left")

        n_total = int(len(merged))
        n_s = int((merged["subset"] == "suppressed").sum())
        n_f = int((merged["subset"] == "forgotten").sum())
        prop_s = float(n_s / n_total) if n_total else np.nan

        s_mid = merged.loc[merged["subset"] == "suppressed", "mid_auc"].dropna()
        f_mid = merged.loc[merged["subset"] == "forgotten", "mid_auc"].dropna()
        mean_s_mid = float(s_mid.mean()) if len(s_mid) else np.nan
        mean_f_mid = float(f_mid.mean()) if len(f_mid) else np.nan
        mid_gap = float(mean_s_mid - mean_f_mid) if np.isfinite(mean_s_mid) and np.isfinite(mean_f_mid) else np.nan

        mean_hkgap_all = float(merged["hkgap"].mean())
        mean_hkgap_s = float(merged.loc[merged["subset"] == "suppressed", "hkgap"].mean()) if n_s else np.nan
        mean_hkgap_f = float(merged.loc[merged["subset"] == "forgotten", "hkgap"].mean()) if n_f else np.nan
        hkgap_s_minus_f = (
            float(mean_hkgap_s - mean_hkgap_f)
            if np.isfinite(mean_hkgap_s) and np.isfinite(mean_hkgap_f)
            else np.nan
        )
        hk_mass = float(prop_s * mean_hkgap_s) if np.isfinite(prop_s) and np.isfinite(mean_hkgap_s) else np.nan

        mean_ki_all = float(merged["k_internal_ck8"].mean())
        mean_ke_all = float(merged["k_external_ck8"].mean())
        mean_ki_s = float(merged.loc[merged["subset"] == "suppressed", "k_internal_ck8"].mean()) if n_s else np.nan
        mean_ke_s = float(merged.loc[merged["subset"] == "suppressed", "k_external_ck8"].mean()) if n_s else np.nan
        mean_ki_f = float(merged.loc[merged["subset"] == "forgotten", "k_internal_ck8"].mean()) if n_f else np.nan
        mean_ke_f = float(merged.loc[merged["subset"] == "forgotten", "k_external_ck8"].mean()) if n_f else np.nan

        rows.append(
            MethodPoint(
                method=method,
                n_total=n_total,
                n_suppressed=n_s,
                n_forgotten=n_f,
                prop_suppressed=prop_s,
                mean_mid_auc_suppressed=mean_s_mid,
                mean_mid_auc_forgotten=mean_f_mid,
                mid_auc_gap=mid_gap,
                mean_hkgap_all=mean_hkgap_all,
                mean_hkgap_suppressed=mean_hkgap_s,
                mean_hkgap_forgotten=mean_hkgap_f,
                hkgap_suppressed_minus_forgotten=hkgap_s_minus_f,
                hidden_knowledge_mass=hk_mass,
                mean_k_internal_ck8_all=mean_ki_all,
                mean_k_external_ck8_all=mean_ke_all,
                mean_k_internal_ck8_suppressed=mean_ki_s,
                mean_k_external_ck8_suppressed=mean_ke_s,
                mean_k_internal_ck8_forgotten=mean_ki_f,
                mean_k_external_ck8_forgotten=mean_ke_f,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        raise RuntimeError("No method rows computed; check that ck8 ext files and trajectory CSV exist.")
    return df


def _scatter_with_labels(
    x: np.ndarray,
    y: np.ndarray,
    labels: Iterable[str],
    title: str,
    xlab: str,
    ylab: str,
    out_stem: str,
) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    labels = [l for i, l in enumerate(list(labels)) if bool(mask[i])]

    r = _pearsonr(x, y)
    rho = _spearmanr(x, y)
    n = int(len(x))

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(x, y, s=48, alpha=0.85, color="#1f77b4", edgecolors="none")
    for xi, yi, lab in zip(x, y, labels):
        ax.text(xi, yi, f"  {lab}", fontsize=9, va="center", ha="left")

    if n >= 2 and np.nanstd(x) > 0:
        try:
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            ax.plot(xx, m * xx + b, color="#333333", linewidth=1.2, alpha=0.7)
        except Exception:
            pass

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.text(
        0.02,
        0.98,
        f"n = {n}\nPearson r = {r:.3f}\nSpearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.12),
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return r, rho


def main() -> None:
    mid_auc_df = load_mid_auc_by_question()
    table = compute_method_points(mid_auc_df)

    cols = [
        "method",
        "n_total",
        "n_suppressed",
        "n_forgotten",
        "prop_suppressed",
        "mean_mid_auc_suppressed",
        "mean_mid_auc_forgotten",
        "mid_auc_gap",
        "mean_hkgap_all",
        "mean_hkgap_suppressed",
        "mean_hkgap_forgotten",
        "hkgap_suppressed_minus_forgotten",
        "hidden_knowledge_mass",
        "mean_k_internal_ck8_all",
        "mean_k_external_ck8_all",
        "mean_k_internal_ck8_suppressed",
        "mean_k_external_ck8_suppressed",
        "mean_k_internal_ck8_forgotten",
        "mean_k_external_ck8_forgotten",
    ]
    for c in cols:
        if c not in table.columns:
            table[c] = np.nan
    table = table[cols].copy()
    table = table.sort_values("method").reset_index(drop=True)

    table.to_csv(OUT_DIR / "method_hkgap_correlation_table.csv", index=False)
    (OUT_DIR / "method_hkgap_correlation_table.md").write_text(
        "# Method-Level HKGap Correlation Table\n\n" + _df_to_markdown_table(table.round(6)),
        encoding="utf-8",
    )

    x = table["mid_auc_gap"].to_numpy(dtype=np.float64)
    labels = table["method"].tolist()

    summary_lines = ["# HKGap Correlation Summary", "", "x-axis: suppressed–forgotten mid-layer AUC gap", ""]
    plots = [
        ("mean_hkgap_all", "HKGap (all questions)", "mid_auc_gap_vs_hkgap_all"),
        ("mean_hkgap_suppressed", "HKGap (suppressed only)", "mid_auc_gap_vs_hkgap_suppressed"),
        (
            "hkgap_suppressed_minus_forgotten",
            "HKGap (suppressed − forgotten)",
            "mid_auc_gap_vs_hkgap_suppressed_minus_forgotten",
        ),
        ("hidden_knowledge_mass", "Hidden-knowledge mass", "mid_auc_gap_vs_hidden_knowledge_mass"),
    ]

    for col, ylab, stem in plots:
        y = table[col].to_numpy(dtype=np.float64)
        r, rho = _scatter_with_labels(
            x=x,
            y=y,
            labels=labels,
            title=f"Mid-layer AUC gap vs {ylab}",
            xlab="mid_auc_gap = mean(mid_auc | suppressed) − mean(mid_auc | forgotten)",
            ylab=ylab,
            out_stem=stem,
        )
        summary_lines.append(f"- {stem}: Pearson r={r:.3f}, Spearman ρ={rho:.3f}, n=8")

    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("Saved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
