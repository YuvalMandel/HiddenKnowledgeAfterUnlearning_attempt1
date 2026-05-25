"""
Fig:pair-composition — pair-level outcome composition by question subset, v6 pipeline.

For each question in Q* at ck8, for each (correct, wrong) pair:
  - int_win: int_proba_correct > int_proba_wrong  (best available per-option probe scores)
  - ext_win: ext_correct > ext_wrong              (logit scores from npy files)
  - pair_outcome: hidden_pair / retained_pair / forgotten_pair / lucky_pair

Subset labels use v6 best_layer parquets. Per-option scores from inside_out_ext/*.npy.

Outputs: v6_pair_composition.pdf/.png, v6_pair_composition.csv, v6_hp_frac.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO     = Path(__file__).parent.parent
OUT_DIR  = REPO / "inside_out_out"
EXT_DIR  = REPO / "inside_out_ext"
SAVE_DIR = Path(__file__).parent
KNOWS    = 0.5
N_OPT    = 4  # 4 MCQ options

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

PALETTE = {
    "hidden_pair":   "#d62728",
    "retained_pair": "#2ca02c",
    "forgotten_pair":"#7f7f7f",
    "lucky_pair":    "#1f77b4",
}
PAIR_ORDER = ["retained_pair", "hidden_pair", "forgotten_pair", "lucky_pair"]
SUBSET_ORDER = ["retained", "suppressed", "forgotten", "lucky"]


def load_cv_bl(model_id: str) -> pd.DataFrame:
    p = OUT_DIR / model_id / "k_scores.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    return df[
        (df["split_type"] == "cv") & (df["domain"] == "bio") &
        (df["clf"] == "LR") & (df["probe_type"] == "own") &
        (df["layer_config"] == "best_layer")
    ][["model_id", "question_idx", "k_internal", "k_external"]].copy()


def pair_outcome(int_win, ext_win):
    if int_win and not ext_win:   return "hidden_pair"
    if int_win and ext_win:       return "retained_pair"
    if not int_win and not ext_win: return "forgotten_pair"
    return "lucky_pair"


# Load Q* and correct answer indices
base_cv = load_cv_bl("base")
qstar = set(base_cv.loc[
    (base_cv["k_internal"] == 1.0) & (base_cv["k_external"] == 1.0),
    "question_idx"
])
print(f"Q* size: {len(qstar)}")

tf = pd.read_csv(REPO / "data" / "wmdp_tf_pairs.csv")
correct_idx = (tf[["original_id", "correct_idx"]]
               .drop_duplicates("original_id")
               .sort_values("original_id")["correct_idx"].astype(int).to_numpy())

# Collect per-pair rows
all_rows = []
hp_rows  = []

for mname, label in METHODS:
    ext_path = EXT_DIR / f"{mname}_ck8_bio_ext.npy"
    int_path = EXT_DIR / f"{mname}_ck8_bio_int_proba.npy"
    if not ext_path.exists() or not int_path.exists():
        print(f"  Skipping {label}: missing npy files"); continue

    ck8_ext = np.load(ext_path)   # (1273, 4)
    ck8_int = np.load(int_path)   # (1273, 4)

    # Subset labels from v6 best_layer ck8
    ck8_df = load_cv_bl(f"{mname}_ck8")
    if ck8_df.empty: continue
    ck8_q = ck8_df[ck8_df["question_idx"].isin(qstar)].set_index("question_idx")

    labels_map = {}
    for qi, row in ck8_q.iterrows():
        ki, ke = row["k_internal"], row["k_external"]
        if   ki > KNOWS and ke > KNOWS:  labels_map[qi] = "retained"
        elif ki > KNOWS and ke <= KNOWS: labels_map[qi] = "suppressed"
        elif ki <= KNOWS and ke > KNOWS: labels_map[qi] = "lucky"
        else:                            labels_map[qi] = "forgotten"

    n_supp = sum(1 for v in labels_map.values() if v == "suppressed")
    print(f"  {label}: suppressed={n_supp}, total Q*={len(labels_map)}")

    for qi, subset in labels_map.items():
        c = int(correct_idx[qi])
        for w in range(N_OPT):
            if w == c: continue
            iw = bool(ck8_int[qi, c] > ck8_int[qi, w])
            ew = bool(ck8_ext[qi, c] > ck8_ext[qi, w])
            po = pair_outcome(iw, ew)
            all_rows.append({
                "method": label, "question_idx": qi, "subset": subset,
                "wrong_idx": w, "pair_outcome": po,
                "int_win": iw, "ext_win": ew,
            })

df = pd.DataFrame(all_rows)
df.to_csv(SAVE_DIR / "v6_pair_composition.csv", index=False)
print(f"\nSaved v6_pair_composition.csv  ({len(df)} rows)")

# Fraction of each pair type within each (method, subset)
comp = df.groupby(["method", "subset", "pair_outcome"]).size().reset_index(name="n")
comp["frac"] = comp["n"] / comp.groupby(["method", "subset"])["n"].transform("sum")

# HP fraction per method (suppressed questions)
hp = comp[(comp["subset"] == "suppressed") & (comp["pair_outcome"] == "hidden_pair")]
hp_mean = hp.groupby("method")["frac"].mean().reset_index()
hp_mean.columns = ["method", "hp_frac"]
hp_mean.to_csv(SAVE_DIR / "v6_hp_frac.csv", index=False)
print("\nHP frac per method (suppressed questions):")
print(hp_mean.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nMean HP frac across methods: {hp['frac'].mean():.3f}")

# Aggregate across methods: mean frac per (subset, pair_outcome)
comp_agg = comp.groupby(["subset", "pair_outcome"])["frac"].mean().reset_index()
comp_agg.to_csv(SAVE_DIR / "v6_pair_composition_agg.csv", index=False)

# ── Fig:pair-composition — stacked bar, 4 subsets, averaged across methods ──────
fig, ax = plt.subplots(figsize=(6, 4.5))

x = np.arange(len(SUBSET_ORDER))
bottoms = np.zeros(len(SUBSET_ORDER))

for po in PAIR_ORDER:
    vals = []
    for s in SUBSET_ORDER:
        r = comp_agg[(comp_agg["subset"] == s) & (comp_agg["pair_outcome"] == po)]
        vals.append(float(r["frac"].iloc[0]) if not r.empty else 0.0)
    ax.bar(x, vals, bottom=bottoms, color=PALETTE.get(po, "#333333"),
           label=po.replace("_", " "), width=0.55, edgecolor="white", linewidth=0.5)
    bottoms += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels([s.capitalize() for s in SUBSET_ORDER], fontsize=10)
ax.set_ylabel("Fraction of pair outcomes", fontsize=10)
ax.set_ylim(0, 1)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
          fontsize=9, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(SAVE_DIR / "v6_pair_composition.pdf", bbox_inches="tight")
fig.savefig(SAVE_DIR / "v6_pair_composition.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved v6_pair_composition.pdf/.png")

# ── Per-method figure (8-panel) ────────────────────────────────────────────────
mlist = [label for _, label in METHODS]
fig2, axes = plt.subplots(4, 2, figsize=(10, 12), sharey=True)
axes = axes.ravel()

for i, method_label in enumerate(mlist):
    ax = axes[i]
    m = comp[comp["method"] == method_label]
    if m.empty:
        ax.axis("off"); continue
    bottoms = np.zeros(len(SUBSET_ORDER))
    for po in PAIR_ORDER:
        vals = []
        for s in SUBSET_ORDER:
            r = m[(m["subset"] == s) & (m["pair_outcome"] == po)]
            vals.append(float(r["frac"].iloc[0]) if not r.empty else 0.0)
        ax.bar(np.arange(len(SUBSET_ORDER)), vals, bottom=bottoms,
               color=PALETTE.get(po, "#333"), label=po.replace("_", " "),
               width=0.55, edgecolor="white", linewidth=0.5)
        bottoms += np.array(vals)
    ax.set_xticks(np.arange(len(SUBSET_ORDER)))
    ax.set_xticklabels([s[:4] for s in SUBSET_ORDER], fontsize=8)
    ax.set_title(method_label, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="upper center", ncol=4,
            bbox_to_anchor=(0.5, 0.99), fontsize=9, frameon=False)
fig2.suptitle("Pair outcome composition by question subset", fontsize=11, y=1.01)
fig2.supylabel("Fraction of pair outcomes", fontsize=9)
fig2.tight_layout(rect=[0.03, 0, 1, 0.97])
fig2.savefig(SAVE_DIR / "v6_pair_composition_per_method.pdf", bbox_inches="tight")
fig2.savefig(SAVE_DIR / "v6_pair_composition_per_method.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved v6_pair_composition_per_method.pdf/.png")
