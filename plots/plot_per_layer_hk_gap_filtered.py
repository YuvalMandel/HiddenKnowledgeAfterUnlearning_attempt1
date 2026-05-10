"""
plot_per_layer_hk_gap_filtered.py

Pre-filter: only questions where base model has PERFECT K=1 both
  - K_internal = 1  (multi-layer LR probe wins all 3 comparisons)
  - K_external = 1  (output logits win all 3 comparisons)

For the filtered questions, classify by POST-UNLEARNING model into 4 new subsets:
  retained   : method K_int > 0.5  AND  method K_ext > 0.5   (internal correct, external correct)
  suppressed : method K_int > 0.5  AND  method K_ext <= 0.5  (internal correct, external wrong)
  forgotten  : method K_int <= 0.5 AND  method K_ext <= 0.5  (internal wrong,   external wrong)
  lucky      : method K_int <= 0.5 AND  method K_ext > 0.5   (internal wrong,   external correct)

Method K_int classification: multi-layer LR (parquet) where available; layer-26 LR (int_proba) for RMU.

For each of 8 methods: dual subplot figure.
  Top    — post-unlearning model (ck8)  per-layer HK gap = K_int(layer) - K_ext
  Bottom — base model                   per-layer HK gap = K_int(layer) - K_ext
           (K_ext=1 for all base questions by construction; shows per-layer internal strength)

Output: plots/per_layer_hk_gap_filtered/per_layer_hk_gap_filtered_{method_id}.{pdf,png}
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE    = Path(__file__).resolve().parent
OUT_DIR = HERE / "per_layer_hk_gap_filtered"
OUT_DIR.mkdir(exist_ok=True)

PARQUET = HERE / "all_k_scores.parquet"
EXT_DIR = HERE.parent / "inside_out_ext"

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
    ("retained",   "Retained\n(int✓ ext✓)",   "#2ca02c"),
    ("suppressed", "Suppressed\n(int✓ ext✗)", "#d62728"),
    ("forgotten",  "Forgotten\n(int✗ ext✗)",  "#7f7f7f"),
    ("lucky",      "Lucky\n(int✗ ext✓)",       "#ff7f0e"),
]

N_LAYERS = 33
KNOWS_THRESHOLD = 0.5

df_all = pd.read_parquet(PARQUET)

# ── Pre-filter: base K_int_multi=1 AND K_ext=1 ────────────────────────────────
base_single = df_all[
    (df_all["model_id"]    == "base") &
    (df_all["split_type"]  == "single") &
    (df_all["layer_config"]== "multi") &
    (df_all["clf"]         == "LR") &
    (df_all["domain"]      == "bio")
][["question_idx", "k_internal", "k_external"]].set_index("question_idx")

perfect_mask = (base_single["k_internal"] == 1.0) & (base_single["k_external"] == 1.0)
perfect_qidx = set(base_single[perfect_mask].index.tolist())
n_filtered   = len(perfect_qidx)
n_total      = len(base_single)
print(f"Base filter: K_int=1 AND K_ext=1 -> {n_filtered}/{n_total} questions ({100*n_filtered/n_total:.1f}%)")

# ── Base CV data, pre-filtered ─────────────────────────────────────────────────
base_cv = df_all[
    (df_all["model_id"]   == "base") &
    (df_all["split_type"] == "cv") &
    (df_all["domain"]     == "bio") &
    (df_all["clf"]        == "LR") &
    (df_all["probe_type"] == "own") &
    df_all["layer_config"].str.match(r"^layer_\d+$") &
    df_all["question_idx"].isin(perfect_qidx)
].copy()
base_cv["layer_idx"] = base_cv["layer_config"].str.extract(r"layer_(\d+)").astype(int)
base_cv["hk_gap"]    = base_cv["k_internal"] - base_cv["k_external"]

# Base K_external per filtered question (constant = 1 by construction, but use actual value)
base_ke_cv = (
    df_all[
        (df_all["model_id"]    == "base") &
        (df_all["split_type"]  == "cv") &
        (df_all["domain"]      == "bio") &
        (df_all["clf"]         == "LR") &
        (df_all["probe_type"]  == "own") &
        (df_all["layer_config"]== "layer_0") &
        (df_all["fold"]        == 0) &
        df_all["question_idx"].isin(perfect_qidx)
    ][["question_idx", "k_external"]]
    .set_index("question_idx")["k_external"]
)


def layer_hk_stats(cv_df, cat_map):
    tmp = cv_df.copy()
    tmp["category"] = tmp["question_idx"].map(cat_map)
    tmp = tmp[tmp["category"].notna()]
    fold_means = (
        tmp.groupby(["fold", "layer_idx", "category"])["hk_gap"]
        .mean().reset_index()
    )
    return (
        fold_means.groupby(["layer_idx", "category"])["hk_gap"]
        .agg(mean="mean", std="std").reset_index()
    )


def draw_subplot(ax, stats, counts, title):
    for cat_id, cat_label, color in CATEGORIES:
        n = counts.get(cat_id, 0)
        if n == 0:
            continue
        s = stats[stats["category"] == cat_id].sort_values("layer_idx")
        if s.empty:
            continue
        xs = s["layer_idx"].values
        ys = s["mean"].values * 100
        ye = s["std"].fillna(0).values * 100
        ax.plot(xs, ys, label=f"{cat_label}  (n={n})",
                color=color, linewidth=1.6, marker="o", markersize=2.5, alpha=0.9)
        ax.fill_between(xs, ys - ye, ys + ye, color=color, alpha=0.13)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.text(N_LAYERS - 0.5, 0.8, "K_int = K_ext", fontsize=7,
            color="black", alpha=0.5, ha="right")
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.set_ylabel("Hidden knowledge gap\nK_internal − K_external (%pts)", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)
    ax.grid(axis="both", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


for method_id, method_label in METHODS:
    fname = method_id.replace("-", "-")   # keep as-is for file lookup

    # ── Method K_ext per filtered question ────────────────────────────────────
    method_ke_cv = df_all[
        (df_all["model_id"]    == method_id) &
        (df_all["split_type"]  == "cv") &
        (df_all["domain"]      == "bio") &
        (df_all["clf"]         == "LR") &
        (df_all["probe_type"]  == "own") &
        (df_all["layer_config"]== "layer_0") &
        (df_all["fold"]        == 0) &
        df_all["question_idx"].isin(perfect_qidx)
    ][["question_idx", "k_external"]].set_index("question_idx")["k_external"]

    if method_ke_cv.empty:
        print(f"[skip] {method_id} — no CV data in parquet")
        continue

    # ── Method K_int classification (multi if available, else layer-26 proba) ─
    method_ki_single = df_all[
        (df_all["model_id"]    == method_id) &
        (df_all["split_type"]  == "single") &
        (df_all["layer_config"]== "multi") &
        (df_all["clf"]         == "LR") &
        (df_all["domain"]      == "bio") &
        df_all["question_idx"].isin(perfect_qidx)
    ][["question_idx", "k_internal"]].set_index("question_idx")["k_internal"]

    int_source = "multi-layer LR"
    if method_ki_single.empty:
        # Fallback: layer-26 int_proba file
        int_fname  = method_id.replace("_ck8", "").replace("-LAT", "-LAT")
        proba_path = EXT_DIR / f"{int_fname}_ck8_bio_int_proba.npy"
        if not proba_path.exists():
            print(f"[skip] {method_id} — no K_int data available")
            continue
        # Load and compute K_int for filtered questions
        import sys; sys.path.insert(0, str(HERE.parent))
        from datasets import load_dataset
        ds  = load_dataset("cais/wmdp", "wmdp-bio", split="test")
        ci  = np.array([int(ex["answer"]) for ex in ds])
        proba = np.load(proba_path)   # (1273, 4)
        ki_dict = {}
        for qi in perfect_qidx:
            c  = ci[qi]
            ws = [proba[qi, j] for j in range(4) if j != c]
            ki_dict[qi] = sum(float(proba[qi, c] > w) for w in ws) / 3.0
        method_ki_single = pd.Series(ki_dict)
        int_source = "layer-26 LR"

    # ── Build category map ─────────────────────────────────────────────────────
    merged = pd.DataFrame({
        "method_ke": method_ke_cv,
        "method_ki": method_ki_single,
    }).dropna()

    mk_ext = merged["method_ke"] > KNOWS_THRESHOLD
    mk_int = merged["method_ki"] > KNOWS_THRESHOLD

    cat_map = pd.Series("forgotten", index=merged.index)
    cat_map[ mk_int &  mk_ext] = "retained"
    cat_map[ mk_int & ~mk_ext] = "suppressed"
    cat_map[~mk_int &  mk_ext] = "lucky"

    counts = cat_map.value_counts().to_dict()
    total  = sum(counts.values())

    print(f"\n{method_id}  (K_int source: {int_source})  n_filtered={total}")
    for cat_id, cat_label, _ in CATEGORIES:
        n = counts.get(cat_id, 0)
        print(f"  {cat_id:<12}: {n:3d}  ({100*n/total:.1f}%)")

    # ── Per-layer CV data, method ──────────────────────────────────────────────
    method_cv = df_all[
        (df_all["model_id"]   == method_id) &
        (df_all["split_type"] == "cv") &
        (df_all["domain"]     == "bio") &
        (df_all["clf"]        == "LR") &
        (df_all["probe_type"] == "own") &
        df_all["layer_config"].str.match(r"^layer_\d+$") &
        df_all["question_idx"].isin(perfect_qidx)
    ].copy()

    if method_cv.empty:
        print(f"  [skip] no per-layer CV data")
        continue

    method_cv["layer_idx"] = method_cv["layer_config"].str.extract(r"layer_(\d+)").astype(int)
    method_cv["hk_gap"]    = method_cv["k_internal"] - method_cv["k_external"]

    method_stats = layer_hk_stats(method_cv, cat_map)
    base_stats   = layer_hk_stats(base_cv,   cat_map)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"hspace": 0.38}
    )

    draw_subplot(ax_top, method_stats, counts,
                 f"Post-unlearning model ({method_label} ck8) — filtered questions")
    draw_subplot(ax_bot, base_stats,   counts,
                 "Base model (same filtered questions, coloured by method's classification)")

    ax_bot.set_xlabel("Layer  (0 = embedding, 1–32 = transformer)", fontsize=9)

    fig.suptitle(
        f"Per-layer HK gap (K_internal − K_external) — {method_label} · Bio · LR · 5-fold CV\n"
        f"Pre-filter: base K_int=1 AND K_ext=1  ({n_filtered} of {n_total} questions, {100*n_filtered/n_total:.1f}%)\n"
        f"New subsets: method K_int vs K_ext threshold=0.5  (K_int source: {int_source})",
        fontsize=9, fontweight="bold"
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    stem = OUT_DIR / f"per_layer_hk_gap_filtered_{method_id}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  Saved -> {stem}.png")
    plt.close(fig)

print("\nDone.")
