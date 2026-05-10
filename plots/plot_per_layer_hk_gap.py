"""
plot_per_layer_hk_gap.py

Per-layer hidden knowledge gap by question category — dual subplot.

  Hidden knowledge gap = K_internal(layer) − K_external   [per question]
  K_external is constant across layers (fixed model property on test set).

Categories defined by base vs method K_external > 0.5:
  retained    : base correct  AND method correct
  suppressed  : base correct  AND method wrong
  emerged     : base wrong    AND method correct
  always_wrong: base wrong    AND method wrong

For each of 8 methods: one figure with two stacked subplots.
  Top    — post-unlearning model (ck8)  HK gap per layer
  Bottom — base model                   HK gap per layer
  (same question subsets, defined by that method's classification)

Lines = mean HK gap across 5 CV folds; shaded band = ±1 std.
Dashed black line at 0 = no hidden knowledge gap.

Output: plots/per_layer_hk_gap/per_layer_hk_gap_{method_id}.{pdf,png}
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE    = Path(__file__).resolve().parent
OUT_DIR = HERE / "per_layer_hk_gap"
OUT_DIR.mkdir(exist_ok=True)

PARQUET = HERE / "all_k_scores.parquet"

METHODS = [
    ("GradDiff_ck8", "GradDiff"),
    ("PB_J_ck8",     "PB&J"),
    ("RMU_ck8",      "RMU"),
    ("RMU-LAT_ck8",  "RMU-LAT"),
    ("RepNoise_ck8", "RepNoise"),
    ("ELM_ck8",      "ELM"),
    ("RR_ck8",       "RR"),
    ("TAR_ck8",      "TAR"),
]

CATEGORIES = [
    ("retained",     "Retained",     "#2ca02c"),
    ("suppressed",   "Suppressed",   "#d62728"),
    ("emerged",      "Emerged",      "#1f77b4"),
    ("always_wrong", "Always wrong", "#7f7f7f"),
]

N_LAYERS = 33

df_all = pd.read_parquet(PARQUET)

# Base K_external per question (constant across layers/folds — use fold 0, layer_0)
base_ke = (
    df_all[
        (df_all["model_id"]    == "base") &
        (df_all["split_type"]  == "cv") &
        (df_all["domain"]      == "bio") &
        (df_all["clf"]         == "LR") &
        (df_all["probe_type"]  == "own") &
        (df_all["layer_config"]== "layer_0") &
        (df_all["fold"]        == 0)
    ][["question_idx", "k_external"]]
    .set_index("question_idx")["k_external"]
)

# Pre-filter base CV data
base_cv = df_all[
    (df_all["model_id"]   == "base") &
    (df_all["split_type"] == "cv") &
    (df_all["domain"]     == "bio") &
    (df_all["clf"]        == "LR") &
    (df_all["probe_type"] == "own") &
    df_all["layer_config"].str.match(r"^layer_\d+$")
].copy()
base_cv["layer_idx"] = base_cv["layer_config"].str.extract(r"layer_(\d+)").astype(int)
base_cv["hk_gap"]    = base_cv["k_internal"] - base_cv["k_external"]


def layer_hk_stats(cv_df, cat_map):
    """Per-(layer, category) mean ± std of HK gap across 5 CV folds."""
    tmp = cv_df.copy()
    tmp["category"] = tmp["question_idx"].map(cat_map)
    tmp = tmp[tmp["category"].notna()]
    fold_means = (
        tmp.groupby(["fold", "layer_idx", "category"])["hk_gap"]
        .mean()
        .reset_index()
    )
    return (
        fold_means.groupby(["layer_idx", "category"])["hk_gap"]
        .agg(mean="mean", std="std")
        .reset_index()
    )


def draw_subplot(ax, stats, counts, title):
    for cat_id, cat_label, color in CATEGORIES:
        n = counts.get(cat_id, 0)
        s = stats[stats["category"] == cat_id].sort_values("layer_idx")
        if s.empty:
            continue
        xs = s["layer_idx"].values
        ys = s["mean"].values * 100
        ye = s["std"].fillna(0).values * 100
        ax.plot(xs, ys, label=f"{cat_label}  (n={n})",
                color=color, linewidth=1.6, marker="o", markersize=2.5, alpha=0.9)
        ax.fill_between(xs, ys - ye, ys + ye, color=color, alpha=0.13)

    # Zero line = no hidden knowledge gap
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.text(N_LAYERS - 0.5, 0.8, "K_int = K_ext", fontsize=7,
            color="black", alpha=0.5, ha="right")

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.set_ylabel("Hidden knowledge gap\nK_internal − K_external (%pts)", fontsize=8.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(axis="both", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


for method_id, method_label in METHODS:
    # Method K_external per question
    method_ke = (
        df_all[
            (df_all["model_id"]    == method_id) &
            (df_all["split_type"]  == "cv") &
            (df_all["domain"]      == "bio") &
            (df_all["clf"]         == "LR") &
            (df_all["probe_type"]  == "own") &
            (df_all["layer_config"]== "layer_0") &
            (df_all["fold"]        == 0)
        ][["question_idx", "k_external"]]
        .set_index("question_idx")["k_external"]
    )

    if method_ke.empty:
        print(f"[skip] {method_id} — no CV data in parquet")
        continue

    merged = (
        base_ke.rename("base_ke").to_frame()
        .join(method_ke.rename("method_ke"))
    )
    bc = merged["base_ke"]    > 0.5
    mc = merged["method_ke"]  > 0.5

    cat_map = pd.Series("always_wrong", index=merged.index)
    cat_map[bc  &  mc]  = "retained"
    cat_map[bc  & ~mc]  = "suppressed"
    cat_map[~bc &  mc]  = "emerged"

    counts = cat_map.value_counts().to_dict()

    # Method CV data with hk_gap
    method_cv = df_all[
        (df_all["model_id"]   == method_id) &
        (df_all["split_type"] == "cv") &
        (df_all["domain"]     == "bio") &
        (df_all["clf"]        == "LR") &
        (df_all["probe_type"] == "own") &
        df_all["layer_config"].str.match(r"^layer_\d+$")
    ].copy()

    if method_cv.empty:
        print(f"[skip] {method_id} — no per-layer CV data")
        continue

    method_cv["layer_idx"] = method_cv["layer_config"].str.extract(r"layer_(\d+)").astype(int)
    method_cv["hk_gap"]    = method_cv["k_internal"] - method_cv["k_external"]

    method_stats = layer_hk_stats(method_cv, cat_map)
    base_stats   = layer_hk_stats(base_cv,   cat_map)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"hspace": 0.38}
    )

    draw_subplot(ax_top, method_stats, counts,
                 f"Post-unlearning model ({method_label} ck8)")
    draw_subplot(ax_bot, base_stats,   counts,
                 "Base model (same question subsets)")

    ax_bot.set_xlabel("Layer  (0 = embedding, 1–32 = transformer)", fontsize=9)

    fig.suptitle(
        f"Per-layer hidden knowledge gap (K_internal − K_external) by category"
        f" — {method_label} · Bio · LR · 5-fold CV",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

    stem = OUT_DIR / f"per_layer_hk_gap_{method_id}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved: {stem}.png")
    plt.close(fig)

print("Done.")
