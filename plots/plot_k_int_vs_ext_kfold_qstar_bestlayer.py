"""
Mean pairwise K score (K_int vs K_ext) — Q* subset, 5-fold CV — single panel.
Best-layer probe variant: layer_config == "best_layer".

Q* = questions where base model has k_internal=1 AND k_external=1 (best-layer probe).
Per fold, only Q* questions that fall in that fold's test set are evaluated.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "inside_out_out"

MODELS = [
    ("base",          "Base"),
    ("GradDiff_ck8",  "GradDiff"),
    ("PB_J_ck8",      "PB&J"),
    ("RMU_ck8",       "RMU"),
    ("RMU-LAT_ck8",   "RMU-LAT"),
    ("RepNoise_ck8",  "RepNoise"),
    ("ELM_ck8",       "ELM"),
    ("RR_ck8",        "RR"),
    ("TAR_ck8",       "TAR"),
]

INT_COLOR = "#2166ac"
EXT_COLOR = "#d6604d"


def load_cv(model_id: str) -> pd.DataFrame:
    p = OUT_DIR / model_id / "k_scores.parquet"
    if not p.exists():
        print(f"  Missing: {p}")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    return df[
        (df["split_type"] == "cv") &
        (df["domain"] == "bio") &
        (df["clf"] == "LR") &
        (df["probe_type"] == "own") &
        (df["layer_config"] == "best_layer")
    ].copy()


# ── Build Q* from base model (best-layer probe) ───────────────────────────────
base_cv = load_cv("base")
if base_cv.empty:
    raise RuntimeError("Base model CV parquet not found or empty.")

qstar = set(
    base_cv.loc[
        (base_cv["k_internal"] == 1.0) & (base_cv["k_external"] == 1.0),
        "question_idx"
    ]
)
print(f"Q* size: {len(qstar)} questions (base k_int=k_ext=1, best-layer probe)")

# ── Compute per-fold stats for each model on Q* questions only ────────────────
rows = []
for model_id, label in MODELS:
    cv = load_cv(model_id)
    if cv.empty:
        print(f"  Missing CV data for {model_id}, skipping.")
        continue

    cv_qstar = cv[cv["question_idx"].isin(qstar)]

    fold_stats = []
    for fold_i, grp in cv_qstar.groupby("fold"):
        n = len(grp)
        if n == 0:
            continue
        fold_stats.append({
            "fold":       fold_i,
            "n":          n,
            "fold_k_int": grp["k_internal"].mean(),
            "fold_k_ext": grp["k_external"].mean(),
        })

    if not fold_stats:
        print(f"  No Q* test rows for {model_id}, skipping.")
        continue

    fs = pd.DataFrame(fold_stats)
    n_folds = len(fs)

    def ms(col):
        return fs[col].mean(), fs[col].std(ddof=1) if n_folds > 1 else (fs[col].mean(), float("nan"))

    ki_m, ki_s = ms("fold_k_int")
    ke_m, ke_s = ms("fold_k_ext")

    rows.append({
        "label":      label,
        "n_folds":    n_folds,
        "k_int":      ki_m,  "k_int_std": ki_s,
        "k_ext":      ke_m,  "k_ext_std": ke_s,
    })
    print(f"  {label:<12}  K_int={ki_m*100:.1f}±{ki_s*100:.1f}  "
          f"K_ext={ke_m*100:.1f}±{ke_s*100:.1f}  (n≈{fs['n'].mean():.0f}/fold)")

data = pd.DataFrame(rows)
labels = data["label"].tolist()
n = len(labels)
x = np.arange(n)
w = 0.35

vi   = data["k_int"].values * 100
ve   = data["k_ext"].values * 100
vi_e = data["k_int_std"].values * 100
ve_e = data["k_ext_std"].values * 100

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(x - w/2, vi, width=w, color=INT_COLOR, alpha=0.88,
       label=r"$K_\mathrm{int}$ (probe, best layer)", zorder=3,
       yerr=vi_e, capsize=3, error_kw=dict(elinewidth=1.0, ecolor="black", alpha=0.7))
ax.bar(x + w/2, ve, width=w, color=EXT_COLOR, alpha=0.88,
       label=r"$K_\mathrm{ext}$ (logit margin)", zorder=3,
       yerr=ve_e, capsize=3, error_kw=dict(elinewidth=1.0, ecolor="black", alpha=0.7))
ax.axhline(50, color="black", linestyle="--", linewidth=0.9, alpha=0.45, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=9)
ax.set_ylabel(r"Mean pairwise $K$ score (%)", fontsize=9)
ax.set_ylim(0, 115)
ax.tick_params(labelsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=8.5, loc="upper right")
for xi, v, se in zip(x - w/2, vi, vi_e):
    ax.text(xi, v + se + 1.5, f"{v:.1f}", ha="center", va="bottom",
            fontsize=6.5, color=INT_COLOR)
for xi, v, se in zip(x + w/2, ve, ve_e):
    ax.text(xi, v + se + 1.5, f"{v:.1f}", ha="center", va="bottom",
            fontsize=6.5, color=EXT_COLOR)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "k_int_vs_ext_kfold_qstar_bestlayer")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}.pdf / .png")
plt.close(fig)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Model':<12} {'K_int':>10} {'K_ext':>10}")
print("-" * 34)
for _, r in data.iterrows():
    print(f"{r['label']:<12} "
          f"{r['k_int']*100:6.1f}±{r['k_int_std']*100:4.1f} "
          f"{r['k_ext']*100:6.1f}±{r['k_ext_std']*100:4.1f}")
