"""
Q* subset counts per method at ck8 — best_layer probe, v6 KFold.

Q* = base model best_layer K_int=1 AND K_ext=1.
Subsets: retained / suppressed / forgotten / lucky (expected ~0 in Q*).
Outputs: v6_subset_counts.csv, v6_subset_counts.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR  = Path(__file__).parent.parent / "inside_out_out"
SAVE_DIR = Path(__file__).parent
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
    "forgotten": "#7f7f7f",
    "lucky":     "#1f77b4",
}


def load_cv_bl(model_id: str) -> pd.DataFrame:
    p = OUT_DIR / model_id / "k_scores.parquet"
    if not p.exists():
        print(f"  Missing: {p}"); return pd.DataFrame()
    df = pd.read_parquet(p)
    return df[
        (df["split_type"] == "cv") & (df["domain"] == "bio") &
        (df["clf"] == "LR") & (df["probe_type"] == "own") &
        (df["layer_config"] == "best_layer")
    ].copy()


# ── Q* from base ──────────────────────────────────────────────────────────────
base_cv = load_cv_bl("base")
qstar   = set(base_cv.loc[
    (base_cv["k_internal"] == 1.0) & (base_cv["k_external"] == 1.0),
    "question_idx"
])
print(f"Q* size: {len(qstar)}")

# ── Subset counts ─────────────────────────────────────────────────────────────
rows = []
for mname, label in METHODS:
    cv = load_cv_bl(f"{mname}_ck8")
    if cv.empty: continue
    q  = cv[cv["question_idx"].isin(qstar)]
    ki = q["k_internal"].values
    ke = q["k_external"].values
    r = dict(
        method    = label,
        retained  = int(((ki > KNOWS) & (ke > KNOWS)).sum()),
        suppressed= int(((ki > KNOWS) & (ke <= KNOWS)).sum()),
        forgotten = int(((ki <= KNOWS) & (ke <= KNOWS)).sum()),
        lucky     = int(((ki <= KNOWS) & (ke > KNOWS)).sum()),
        total     = len(q),
    )
    rows.append(r)
    print(f"  {label:<12}  retained={r['retained']:4d}  suppressed={r['suppressed']:4d}"
          f"  forgotten={r['forgotten']:4d}  lucky={r['lucky']:3d}  (n={r['total']})")

df_counts = pd.DataFrame(rows)
df_counts.to_csv(SAVE_DIR / "v6_subset_counts.csv", index=False)
print(f"\nSaved: v6_subset_counts.csv")

# ── Stacked bar chart (retained / suppressed / forgotten; drop lucky if all 0) ─
labels   = df_counts["method"].tolist()
n        = len(labels)
x        = np.arange(n)
subsets  = ["retained", "suppressed", "forgotten"]
if df_counts["lucky"].sum() > 0:
    subsets.append("lucky")

fig, ax = plt.subplots(figsize=(9, 4.5))
bottoms = np.zeros(n)
for s in subsets:
    vals = df_counts[s].values
    ax.bar(x, vals, bottom=bottoms, color=SUBSET_COLORS[s],
           label=s.capitalize(), width=0.6, zorder=3)
    for xi, v, b in zip(x, vals, bottoms):
        if v > 5:
            ax.text(xi, b + v/2, str(v), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Number of Q* questions", fontsize=9)
ax.set_ylim(0, df_counts["total"].max() * 1.08)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=8.5, loc="upper right", ncol=len(subsets))
fig.tight_layout()
fig.savefig(SAVE_DIR / "v6_subset_counts.pdf", bbox_inches="tight")
fig.savefig(SAVE_DIR / "v6_subset_counts.png", dpi=150, bbox_inches="tight")
print(f"Saved: v6_subset_counts.pdf / .png")
plt.close(fig)
