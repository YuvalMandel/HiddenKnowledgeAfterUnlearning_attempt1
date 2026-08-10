import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hk_utils import iclr_figsize, use_iclr_style

use_iclr_style()

OUT_DIR = Path("inside_out_out")
MODELS = [
    ("base",         "Base"),
    ("GradDiff_ck8", "GradDiff"),
    ("PB_J_ck8",     "PB&J"),
    ("RMU_ck8",      "RMU"),
    ("RMU-LAT_ck8",  "RMU-LAT"),
    ("RepNoise_ck8", "RepNoise"),
    ("ELM_ck8",      "ELM"),
    ("RR_ck8",       "RR"),
    ("TAR_ck8",      "TAR"),
]
INT_COLOR = "#2166ac"; EXT_COLOR = "#d6604d"

frames = []
for model_id, _ in MODELS:
    p = OUT_DIR / model_id / "k_scores.parquet"
    if p.exists():
        frames.append(pd.read_parquet(p))
df_all = pd.concat(frames, ignore_index=True)

cv = df_all[
    (df_all["split_type"]   == "cv") &
    (df_all["domain"]       == "bio") &
    (df_all["clf"]          == "LR") &
    (df_all["probe_type"]   == "own") &
    (df_all["layer_config"] == "best_layer")
].copy()

rows = []
for model_id, label in MODELS:
    sub = cv[cv["model_id"] == model_id]
    if sub.empty:
        print(f"  Missing: {model_id}")
        continue
    fs = sub.groupby("fold").agg(ki=("k_internal","mean"), ke=("k_external","mean")).reset_index()
    ki_m, ki_s = fs["ki"].mean(), fs["ki"].std(ddof=1)
    ke_m, ke_s = fs["ke"].mean(), fs["ke"].std(ddof=1)
    rows.append({"label":label,"k_int":ki_m,"k_int_std":ki_s,"k_ext":ke_m,"k_ext_std":ke_s})
    print(f"  {label:<12} K_int={ki_m*100:.1f}+-{ki_s*100:.1f}  K_ext={ke_m*100:.1f}+-{ke_s*100:.1f}")

data = pd.DataFrame(rows)
x = np.arange(len(data)); w = 0.35
vi   = data["k_int"].values * 100
ve   = data["k_ext"].values * 100
vi_e = data["k_int_std"].values * 100
ve_e = data["k_ext_std"].values * 100

fig, ax = plt.subplots(figsize=iclr_figsize(aspect=4.2 / 7))
ax.bar(x-w/2, vi, w, color=INT_COLOR, alpha=0.88,
       label=r"$K_\mathrm{int}$ (best-layer probe)", zorder=3,
       yerr=vi_e, capsize=3, error_kw=dict(elinewidth=1.0, ecolor="black", alpha=0.7))
ax.bar(x+w/2, ve, w, color=EXT_COLOR, alpha=0.88,
       label=r"$K_\mathrm{ext}$ (logit margin)", zorder=3,
       yerr=ve_e, capsize=3, error_kw=dict(elinewidth=1.0, ecolor="black", alpha=0.7))
ax.axhline(50, color="black", ls="--", lw=0.9, alpha=0.45, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(data["label"].tolist(), rotation=38, ha="right", fontsize=9)
ax.set_ylabel(r"Mean pairwise $K$ score (%)", fontsize=9)
ax.set_ylim(0, 110)
ax.tick_params(labelsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.55, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=8.5, loc="upper right")
for xi, v, se in zip(x-w/2, vi, vi_e):
    ax.text(xi, v+se+1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INT_COLOR)
for xi, v, se in zip(x+w/2, ve, ve_e):
    ax.text(xi, v+se+1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=EXT_COLOR)
fig.tight_layout()
out = "plots/k_int_vs_ext_kfold_single_bestlayer"
fig.savefig(out+".pdf", bbox_inches="tight")
fig.savefig(out+".png", dpi=150, bbox_inches="tight")
print(f"Saved: {out}.png")
plt.close(fig)
