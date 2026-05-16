#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 replacement: K_int vs K_ext per method at ck8.

Shows both K_int (full-layer probe pairwise K score, CV-averaged) and
K_ext (logit-margin pairwise K score) as grouped bars, demonstrating
the hidden-knowledge gap.  Pre-filters to the 307 questions where the
base model has K_int=K_ext=1.

Outputs (plots/kint_kext_gap/):
  kint_kext_gap.pdf / .png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "plots"))

from hk_utils import (
    METHODS, load_correct_idx, load_split_indices, load_scores_df,
    compute_prefilter_mask, get_parquet_full_single, get_cv_full_df,
)

def get_per_layer_single(df):
    """Per-layer single-split data — used as fallback K_ext source."""
    return df[
        (df["domain"] == "bio")
        & (df["clf"] == "LR")
        & (df["layer_config"] == "layer_16")
        & (df["split_type"] == "single")
    ][["model_id", "question_idx", "k_external"]]

OUT_DIR = REPO / "plots" / "kint_kext_gap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "GradDiff": "GradDiff",
    "RMU": "RMU",
    "RMU-LAT": "RMU-LAT",
    "RepNoise": "RepNoise",
    "ELM": "ELM",
    "RR": "RR",
    "TAR": "TAR",
    "PB_J": "PB&J",
}

COL_KINT = "#1f77b4"
COL_KEXT = "#d62728"
CHANCE   = 0.5


def get_method_means(parquet_full_single, parquet_cv, parquet_per_layer, filt_te):
    """
    Returns per-method (K_int_mean, K_ext_mean) at ck8 over pre-filtered questions.
    K_int: from full-layer CV probe (primary K_int source; NaN if unavailable).
    K_ext: from full-layer single-split parquet; falls back to per-layer layer_16 if missing.
    """
    filt_set = set(int(q) for q in filt_te)

    results = {}

    # Base model reference
    base_full = parquet_full_single[parquet_full_single["model_id"] == "base"]
    base_filt = base_full[base_full["question_idx"].isin(filt_set)]
    results["Base"] = {
        "k_int": base_filt["k_internal"].mean(),
        "k_ext": base_filt["k_external"].mean(),
    }

    for method in METHODS:
        mid = f"{method}_ck8"
        # K_ext from full-layer single-split parquet; fallback to per-layer
        row_single = parquet_full_single[parquet_full_single["model_id"] == mid]
        row_filt = row_single[row_single["question_idx"].isin(filt_set)]

        if not row_filt.empty:
            k_ext = row_filt["k_external"].mean()
        else:
            row_pl = parquet_per_layer[parquet_per_layer["model_id"] == mid]
            row_pl_filt = row_pl[row_pl["question_idx"].isin(filt_set)]
            k_ext = row_pl_filt["k_external"].mean() if not row_pl_filt.empty else np.nan

        # K_int from CV full-layer probe; fallback to single-split full-layer
        row_cv = parquet_cv[parquet_cv["model_id"] == mid]
        row_cv_filt = row_cv[row_cv["question_idx"].isin(filt_set)]
        if not row_cv_filt.empty:
            k_int = row_cv_filt["k_internal"].mean()
        elif not row_filt.empty:
            k_int = row_filt["k_internal"].mean()
        else:
            k_int = np.nan

        results[method] = {"k_int": k_int, "k_ext": k_ext}

    return results


def plot_gap(results, n_filtered):
    methods_shown = ["Base"] + METHODS
    labels = [METHOD_LABELS.get(m, m) for m in methods_shown]
    k_int_vals = [results.get(m, {}).get("k_int", np.nan) for m in methods_shown]
    k_ext_vals = [results.get(m, {}).get("k_ext", np.nan) for m in methods_shown]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (ki, ke) in enumerate(zip(k_int_vals, k_ext_vals)):
        # K_int bar (hatched if NaN = full-layer probe unavailable)
        if not np.isnan(ki):
            ax.bar(x[i] - width / 2, ki, width, color=COL_KINT, alpha=0.85, edgecolor="white",
                   label=r"$K_\mathrm{int}$ (full-layer probe)" if i == 0 else "_")
        else:
            ax.bar(x[i] - width / 2, 0, width, color=COL_KINT, alpha=0.3, edgecolor=COL_KINT,
                   hatch="//", label=r"$K_\mathrm{int}$ (probe unavailable)" if i > 0 else "_")
            ax.text(x[i] - width / 2, 0.03, "n/a", ha="center", va="bottom", fontsize=7, color="gray")
        # K_ext bar
        if not np.isnan(ke):
            ax.bar(x[i] + width / 2, ke, width, color=COL_KEXT, alpha=0.85, edgecolor="white",
                   label=r"$K_\mathrm{ext}$ (logit margin)" if i == 0 else "_")

    ax.axhline(CHANCE, color="black", lw=0.9, ls="--", label="Chance (0.5)")
    ax.axvline(0.5, color="gray", lw=0.8, ls=":")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel(r"Mean pairwise $K$ score", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        r"$K_\mathrm{int}$ vs $K_\mathrm{ext}$ at ck8 -- hidden-knowledge gap"
        f"\n(pre-filtered {n_filtered} questions; base model shown for reference)",
        fontsize=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"kint_kext_gap.{ext}", bbox_inches="tight", dpi=180)
    print(f"Saved kint_kext_gap.pdf / .png  ->  {OUT_DIR}")
    plt.close(fig)


def main():
    correct_idx = load_correct_idx()
    te, orig_to_pos = load_split_indices()
    df = load_scores_df()

    parquet_single   = get_parquet_full_single(df)
    parquet_cv       = get_cv_full_df(df)
    parquet_per_layer = get_per_layer_single(df)

    filt_mask = compute_prefilter_mask(te, orig_to_pos, df)
    filt_te = te[filt_mask]
    n_filtered = len(filt_te)
    print(f"Pre-filtered questions: {n_filtered}")

    results = get_method_means(parquet_single, parquet_cv, parquet_per_layer, filt_te)

    print("\nMethod        K_int   K_ext   gap")
    for m in ["Base"] + METHODS:
        r = results.get(m, {})
        ki, ke = r.get("k_int", float("nan")), r.get("k_ext", float("nan"))
        gap = ki - ke if not (np.isnan(ki) or np.isnan(ke)) else float("nan")
        print(f"  {m:<12s}  {ki:.3f}   {ke:.3f}   {gap:+.3f}")

    plot_gap(results, n_filtered)


if __name__ == "__main__":
    main()
