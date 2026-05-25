# -*- coding: utf-8 -*-
"""
Data-driven term analysis: log-odds ratio of word frequencies by Q* subset.
One figure per method, 2x2 grid (one panel per subset vs rest within that method).

Output: plots/v6_logodds/<method>.pdf/.png
"""
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from math import log

REPO     = Path(__file__).parent.parent
OUT_DIR  = REPO / "inside_out_out"
SAVE_DIR = Path(__file__).parent / "v6_logodds"
SAVE_DIR.mkdir(exist_ok=True)
KNOWS    = 0.5
TOP_N    = 20

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
SUBSETS = ["retained", "suppressed", "forgotten", "lucky"]
SUBSET_COLORS = {
    "retained":  "#2ca02c",
    "suppressed":"#d62728",
    "forgotten": "#888888",
    "lucky":     "#1f77b4",
}

def tokenize(text):
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if len(w) >= 2]


def load_cv_bl(model_id):
    p = OUT_DIR / model_id / "k_scores.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    return df[
        (df["split_type"] == "cv") & (df["domain"] == "bio") &
        (df["clf"] == "LR") & (df["probe_type"] == "own") &
        (df["layer_config"] == "best_layer")
    ][["question_idx", "k_internal", "k_external"]].copy()


def label_subset(ki, ke):
    if   ki > KNOWS and ke > KNOWS:   return "retained"
    elif ki > KNOWS and ke <= KNOWS:  return "suppressed"
    elif ki <= KNOWS and ke <= KNOWS: return "forgotten"
    else:                             return "lucky"


def log_odds(c_a, n_a, c_b, n_b, alpha=0.5):
    p_a = (c_a + alpha) / (n_a + alpha * 2)
    p_b = (c_b + alpha) / (n_b + alpha * 2)
    return log(p_a / (1 - p_a)) - log(p_b / (1 - p_b))


def build_logodds_df(counts, obs_n, min_count=3):
    all_words = set().union(*[set(c) for c in counts.values()])
    rows = []
    for w in all_words:
        if sum(counts[s].get(w, 0) for s in SUBSETS) < min_count:
            continue
        row = {"word": w}
        for s in SUBSETS:
            row[f"n_{s}"] = counts[s].get(w, 0)
        for s in SUBSETS:
            rest = [x for x in SUBSETS if x != s]
            n_r  = sum(obs_n[x] for x in rest)
            row[f"lo_{s}"] = log_odds(
                row[f"n_{s}"], obs_n[s],
                sum(row[f"n_{x}"] for x in rest), n_r
            )
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def draw_panel(ax, df, subset):
    col   = f"lo_{subset}"
    color = SUBSET_COLORS[subset]
    top_p = df.nlargest(TOP_N, col)
    top_n = df.nsmallest(TOP_N, col)
    pdf   = pd.concat([top_p, top_n]).drop_duplicates("word").sort_values(col)
    bars  = [color if v > 0 else "#cccccc" for v in pdf[col]]
    ax.barh(pdf["word"], pdf[col], color=bars, edgecolor="none", height=0.72)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Q* from base ──────────────────────────────────────────────────────────────
base_cv = load_cv_bl("base")
qstar = set(base_cv.loc[
    (base_cv["k_internal"] == 1.0) & (base_cv["k_external"] == 1.0),
    "question_idx"
])
print(f"Q* size: {len(qstar)}")

tf    = pd.read_csv(REPO / "data" / "wmdp_tf_pairs.csv")[
    ["original_id", "question"]].drop_duplicates("original_id")
qtext = tf.set_index("original_id")["question"].to_dict()

# ── Per-method figures ────────────────────────────────────────────────────────
for mname, label in METHODS:
    ck8 = load_cv_bl(f"{mname}_ck8")
    if ck8.empty:
        print(f"  {label}: SKIP"); continue
    ck8_q = ck8[ck8["question_idx"].isin(qstar)].set_index("question_idx")

    words  = {s: [] for s in SUBSETS}
    obs_n  = {s: 0  for s in SUBSETS}
    for qi, row in ck8_q.iterrows():
        s    = label_subset(row["k_internal"], row["k_external"])
        text = qtext.get(int(qi), "")
        if not text:
            continue
        words[s].extend(tokenize(text))
        obs_n[s] += 1

    counts     = {s: Counter(words[s]) for s in SUBSETS}
    total_toks = {s: len(words[s]) for s in SUBSETS}
    df_m   = build_logodds_df(counts, total_toks, min_count=3)
    if df_m.empty:
        print(f"  {label}: no words"); continue

    fig, axes = plt.subplots(2, 2, figsize=(13, max(8, TOP_N * 0.40)))
    for ax, s in zip(axes.flatten(), SUBSETS):
        draw_panel(ax, df_m, s)
        ax.set_title(f"{s.capitalize()}  (n={obs_n[s]}) vs rest",
                     fontsize=9, fontweight="bold", color=SUBSET_COLORS[s])
    axes[1, 0].set_xlabel("Log-odds ratio", fontsize=8)
    axes[1, 1].set_xlabel("Log-odds ratio", fontsize=8)
    fig.suptitle(f"{label} — word log-odds by Q* subset  (Laplace smoothed, min count=3)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    stem = mname.replace("-", "_")
    fig.savefig(SAVE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(SAVE_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {label}: saved  {dict((s, obs_n[s]) for s in SUBSETS)}")

print(f"\nSaved to: {SAVE_DIR}/")
