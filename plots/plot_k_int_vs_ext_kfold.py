"""
K-internal vs K-external with 5-fold CV variance bars.
Reads plots/all_k_scores.parquet (all models, cv split_type rows).
For each model uses the layer with the highest mean CV K_internal as probe layer.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARQUET = os.path.join(os.path.dirname(__file__), "all_k_scores.parquet")

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

PANELS = [
    ("k_int",   "k_ext",   "Mean K  (pairwise ranking score)",    "K (%)"),
    ("acc_int", "acc_ext", "Accuracy  (K > 0.5)",                 "Accuracy (%)"),
    ("auc_int", "auc_ext", "AUC  (option-level, probe vs logit)", "AUC (%)"),
]

df_all = pd.read_parquet(PARQUET)
cv = df_all[
    (df_all["split_type"] == "cv") &
    (df_all["domain"] == "bio") &
    (df_all["clf"] == "LR") &
    (df_all["probe_type"] == "own") &
    (df_all["layer_config"].str.startswith("layer_"))
].copy()

# Per-model best layer by CV mean K_internal
best_layer = (
    cv.groupby(["model_id", "layer_config"])["k_internal"]
    .mean()
    .reset_index()
    .sort_values("k_internal", ascending=False)
    .groupby("model_id")
    .first()["layer_config"]
)

rows = []
for model_id, label in MODELS:
    if model_id not in best_layer.index:
        print(f"  Missing CV data for {model_id}, skipping.")
        continue
    lc = best_layer[model_id]

    sub = cv[(cv["model_id"] == model_id) & (cv["layer_config"] == lc)]
    if sub.empty:
        continue

    fold_stats = (
        sub.groupby("fold")
        .agg(
            fold_k_int   = ("k_internal", "mean"),
            fold_k_ext   = ("k_external", "mean"),
            fold_acc_int = ("k_internal", lambda x: (x > 0.5).mean()),
            fold_acc_ext = ("k_external", lambda x: (x > 0.5).mean()),
            fold_auc_int = ("test_auc",   "first"),
            fold_auc_ext = ("ext_auc",    "first"),
        )
        .reset_index()
    )

    n_folds = len(fold_stats)

    def ms(col):
        return fold_stats[col].mean(), fold_stats[col].std(ddof=1)

    ki_m, ki_s = ms("fold_k_int")
    ke_m, ke_s = ms("fold_k_ext")
    ai_m, ai_s = ms("fold_acc_int")
    ae_m, ae_s = ms("fold_acc_ext")
    ui_m, ui_s = ms("fold_auc_int")
    ue_m, ue_s = ms("fold_auc_ext")

    rows.append({
        "label":    label,
        "best_lc":  lc,
        "n_folds":  n_folds,
        "k_int":    ki_m,  "k_int_std":    ki_s,
        "k_ext":    ke_m,  "k_ext_std":    ke_s,
        "acc_int":  ai_m,  "acc_int_std":  ai_s,
        "acc_ext":  ae_m,  "acc_ext_std":  ae_s,
        "auc_int":  ui_m,  "auc_int_std":  ui_s,
        "auc_ext":  ue_m,  "auc_ext_std":  ue_s,
    })
    print(f"  {label:<12}  best_layer={lc}  K_int={ki_m*100:.1f}±{ki_s*100:.1f}  K_ext={ke_m*100:.1f}±{ke_s*100:.1f}")

data = pd.DataFrame(rows)
labels = data["label"].tolist()
n = len(labels)
x = np.arange(n)
w = 0.35

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
fig.suptitle(
    "Internal (probe, best layer) vs External (logit) — Bio · LR · 5-Fold CV",
    fontsize=11, fontweight="bold", y=1.02
)

for ax, (col_i, col_e, title, ylabel) in zip(axes, PANELS):
    vi    = data[col_i].values * 100
    ve    = data[col_e].values * 100
    vi_e  = data[f"{col_i}_std"].values * 100
    ve_e  = data[f"{col_e}_std"].values * 100

    ax.bar(x - w/2, vi, width=w, color=INT_COLOR, alpha=0.88,
           label="Internal (probe)", zorder=3,
           yerr=vi_e, capsize=3, error_kw=dict(elinewidth=1.0, ecolor="black", alpha=0.7))
    ax.bar(x + w/2, ve, width=w, color=EXT_COLOR, alpha=0.88,
           label="External (logit)", zorder=3,
           yerr=ve_e, capsize=3, error_kw=dict(elinewidth=1.0, ecolor="black", alpha=0.7))

    ax.axhline(50, color="black", linestyle="--", linewidth=0.9, alpha=0.45, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8.5)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper right")

    for xi, v in zip(x - w/2, vi):
        ax.text(xi, v + vi_e[list(x - w/2).index(xi)] + 1.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=6, color=INT_COLOR)
    for xi, v, se in zip(x + w/2, ve, ve_e):
        ax.text(xi, v + se + 1.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=6, color=EXT_COLOR)

fig.tight_layout()
base_out = os.path.join(os.path.dirname(__file__), "k_int_vs_ext_kfold")
fig.savefig(base_out + ".pdf", bbox_inches="tight")
fig.savefig(base_out + ".png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {base_out}.pdf / .png")
plt.close(fig)

# Single-panel version (Mean K only) for paper Figure 1
col_i, col_e, title, ylabel = PANELS[0]
vi   = data[col_i].values * 100
ve   = data[col_e].values * 100
vi_e = data[f"{col_i}_std"].values * 100
ve_e = data[f"{col_e}_std"].values * 100

fig1, ax1 = plt.subplots(figsize=(7, 4.2))
ax1.bar(x - w/2, vi, width=w, color=INT_COLOR, alpha=0.88,
        label=r"$K_\mathrm{int}$ (probe, best layer)", zorder=3,
        yerr=vi_e, capsize=3, error_kw=dict(elinewidth=1.0, ecolor="black", alpha=0.7))
ax1.bar(x + w/2, ve, width=w, color=EXT_COLOR, alpha=0.88,
        label=r"$K_\mathrm{ext}$ (logit margin)", zorder=3,
        yerr=ve_e, capsize=3, error_kw=dict(elinewidth=1.0, ecolor="black", alpha=0.7))
ax1.axhline(50, color="black", linestyle="--", linewidth=0.9, alpha=0.45, zorder=2)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=38, ha="right", fontsize=9)
ax1.set_ylabel(r"Mean pairwise $K$ score (%)", fontsize=9)
ax1.set_ylim(0, 110)
ax1.tick_params(labelsize=8.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(fontsize=8.5, loc="upper right")
for xi, v, se in zip(x - w/2, vi, vi_e):
    ax1.text(xi, v + se + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INT_COLOR)
for xi, v, se in zip(x + w/2, ve, ve_e):
    ax1.text(xi, v + se + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=EXT_COLOR)
fig1.tight_layout()
single_out = os.path.join(os.path.dirname(__file__), "k_int_vs_ext_kfold_single")
fig1.savefig(single_out + ".pdf", bbox_inches="tight")
fig1.savefig(single_out + ".png", dpi=150, bbox_inches="tight")
print(f"Saved: {single_out}.pdf / .png")
plt.close(fig1)

# Summary table
print(f"\n{'Model':<12} {'Layer':<10} {'K_int':>10} {'K_ext':>10}  {'Acc_int':>10} {'Acc_ext':>10}  {'AUC_int':>10} {'AUC_ext':>10}")
print("-" * 82)
for _, r in data.iterrows():
    print(f"{r['label']:<12} {r['best_lc']:<10} "
          f"{r['k_int']*100:6.1f}±{r['k_int_std']*100:4.1f} "
          f"{r['k_ext']*100:6.1f}±{r['k_ext_std']*100:4.1f}  "
          f"{r['acc_int']*100:6.1f}±{r['acc_int_std']*100:4.1f} "
          f"{r['acc_ext']*100:6.1f}±{r['acc_ext_std']*100:4.1f}  "
          f"{r['auc_int']*100:6.1f}±{r['auc_int_std']*100:4.1f} "
          f"{r['auc_ext']*100:6.1f}±{r['auc_ext_std']*100:4.1f}")
