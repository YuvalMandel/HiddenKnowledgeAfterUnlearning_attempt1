# -*- coding: utf-8 -*-
"""
Word presence percentage by Q* subset.

For each method: 2x2 grid, one panel per subset.
Each panel shows the top 20 words by % of questions in that subset
containing the word (binary: word present yes/no per question).

Output: plots/v6_word_pct/<method>.pdf/.png
"""
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

REPO     = Path(__file__).parent.parent
OUT_DIR  = REPO / "inside_out_out"
SAVE_DIR = Path(__file__).parent / "v6_word_pct"
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

STOPWORDS = {
    "a","an","the","and","or","but","if","in","on","at","to","of","for","with",
    "by","from","into","through","during","before","after","above","below",
    "between","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "can","not","no","nor","than","that","this","these","those","it","its",
    "which","who","what","how","when","where","why","all","both","each","few",
    "more","most","other","some","such","any","only","own","same","so","just",
    "because","as","until","while","he","she","they","we","you","i","my",
    "your","his","her","our","their","also","about","up","out","then","there",
    "here","very","into","onto","over","under","used","using","following",
    "known","based","due","via","per","two","one","three","four","five","six",
    "many","often","well","whether","without","within","among",
}


def tokenize(text):
    words = re.findall(r"[a-z]+", text.lower())
    return set(w for w in words if len(w) >= 3 and w not in STOPWORDS)


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


def draw_panel(ax, pct_df, subset, n_qs):
    color = SUBSET_COLORS[subset]
    col   = f"pct_{subset}"
    top   = pct_df.nlargest(TOP_N, col).sort_values(col)
    ax.barh(top["word"], top[col], color=color, edgecolor="none", height=0.72, alpha=0.85)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of questions", fontsize=7)
    ax.set_title(f"{subset.capitalize()}  (n={n_qs})",
                 fontsize=9, fontweight="bold", color=color)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # annotate bar end with value
    for bar in ax.patches:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{w:.0f}%", va="center", ha="left", fontsize=6, color="#444444")


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

    # Binary word presence: count questions (not occurrences) per word per subset
    word_q_count = {s: Counter() for s in SUBSETS}
    obs_n        = {s: 0 for s in SUBSETS}

    for qi, row in ck8_q.iterrows():
        s    = label_subset(row["k_internal"], row["k_external"])
        text = qtext.get(int(qi), "")
        if not text:
            continue
        obs_n[s] += 1
        for w in tokenize(text):   # tokenize returns a set — no duplicates
            word_q_count[s][w] += 1

    # Build percentage DataFrame
    all_words = set().union(*[set(c) for c in word_q_count.values()])
    rows = []
    for w in all_words:
        row = {"word": w}
        for s in SUBSETS:
            n = obs_n[s]
            row[f"pct_{s}"] = 100.0 * word_q_count[s][w] / n if n > 0 else 0.0
        rows.append(row)
    pct_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(13, max(8, TOP_N * 0.40)))
    for ax, s in zip(axes.flatten(), SUBSETS):
        draw_panel(ax, pct_df, s, obs_n[s])

    fig.suptitle(f"{label} — word presence (% of questions) by Q* subset",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    stem = mname.replace("-", "_")
    fig.savefig(SAVE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(SAVE_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {label}: saved  {dict((s, obs_n[s]) for s in SUBSETS)}")

print(f"\nSaved to: {SAVE_DIR}/")
