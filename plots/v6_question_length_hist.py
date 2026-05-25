# -*- coding: utf-8 -*-
"""
Question length histogram by Q* subset -- best_layer, v6 KFold, ck8.

For each of 8 methods: 4 overlaid histograms of question word count
(retained / suppressed / forgotten / lucky), using Q* questions only.

Output: plots/v6_question_length_hist/<method>.pdf/.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO     = Path(__file__).parent.parent
OUT_DIR  = REPO / "inside_out_out"
SAVE_DIR = Path(__file__).parent / "v6_question_length_hist"
SAVE_DIR.mkdir(exist_ok=True)
KNOWS    = 0.5

METHODS = [
    ("GradDiff", "GradDiff"),
    ("PB_J",     "PB&J"),
    ("RMU",      "RMU"),
    ("RMU-LAT",  "RMU-LAT"),
    ("RepNoise", "RepNoise"),
    ("ELM",      "ELM"),
    ("RR",       "RR"),
    ("TAR",      "TAR"),
]

SUBSET_COLORS = {
    "retained":  "#2ca02c",
    "suppressed":"#d62728",
    "forgotten": "#888888",
    "lucky":     "#1f77b4",
}

# ── Load question text → word count per original_id ───────────────────────────
csv_path = REPO / "data" / "wmdp_tf_pairs.csv"
tf = pd.read_csv(csv_path)[["original_id", "question"]].drop_duplicates("original_id")
tf["word_count"] = tf["question"].str.split().str.len()
word_count = tf.set_index("original_id")["word_count"].to_dict()

# ── Load Q* from base ─────────────────────────────────────────────────────────
def load_cv_bl(model_id: str) -> pd.DataFrame:
    p = OUT_DIR / model_id / "k_scores.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    return df[
        (df["split_type"] == "cv") & (df["domain"] == "bio") &
        (df["clf"] == "LR") & (df["probe_type"] == "own") &
        (df["layer_config"] == "best_layer")
    ][["question_idx", "k_internal", "k_external"]].copy()

base_cv = load_cv_bl("base")
qstar   = set(base_cv.loc[
    (base_cv["k_internal"] == 1.0) & (base_cv["k_external"] == 1.0),
    "question_idx"
])
print(f"Q* size: {len(qstar)}")

BINS = np.arange(0, 120, 6)

# ── One figure per method ─────────────────────────────────────────────────────
for mname, label in METHODS:
    ck8 = load_cv_bl(f"{mname}_ck8")
    if ck8.empty:
        print(f"  {label}: SKIP (missing ck8)"); continue

    ck8_q = ck8[ck8["question_idx"].isin(qstar)].set_index("question_idx")
    subset_words = {s: [] for s in SUBSET_COLORS}
    for qi, row in ck8_q.iterrows():
        ki, ke = row["k_internal"], row["k_external"]
        if   ki > KNOWS and ke > KNOWS:   s = "retained"
        elif ki > KNOWS and ke <= KNOWS:  s = "suppressed"
        elif ki <= KNOWS and ke <= KNOWS: s = "forgotten"
        else:                             s = "lucky"
        wc = word_count.get(int(qi))
        if wc is not None:
            subset_words[s].append(wc)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    for s, color in SUBSET_COLORS.items():
        vals = subset_words[s]
        if not vals:
            continue
        ax.hist(vals, bins=BINS, color=color, alpha=0.45,
                label=f"{s.capitalize()} (n={len(vals)})",
                density=True, zorder=3)
        ax.axvline(np.median(vals), color=color, linewidth=1.5,
                   linestyle="--", alpha=0.85, zorder=4)

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Question word count", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.01, 0.99,
            "Dashed line = median. Density-normalised.",
            transform=ax.transAxes, fontsize=7, va="top", color="#444444")

    fig.tight_layout()
    stem = mname.replace("-", "_")
    fig.savefig(SAVE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(SAVE_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {label}: saved")

print(f"\nSaved to: {SAVE_DIR}/")
