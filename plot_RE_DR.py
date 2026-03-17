#!/usr/bin/env python3
"""
Plot Representational Erasure (RE) vs Directional Rotation (DR)
for each unlearning method, with attack score as color.

Notation (A = accuracy, uppercase = Post model/probe, lowercase = Base model/probe):
  APP_base : Base probe → Base model  (Table 2, Base row, ml_lr_acc)  = 0.681
  APP_post : Post probe → Post model  (Table 3, method row, ml_lr_acc)
  ApP      : Base probe → Post model  (Table 2, method row, ml_lr_acc)
  APp      : Post probe → Base model  (Table 5, method row, avg_lr_acc)

RE = (APP_base - APP_post) / (APP_base - 0.5)
   ≈ 0: method internal accuracy ≈ base (no erasure)
   ≈ 1: method internal accuracy ≈ chance (full erasure)

DR_forward  = (APP_post - ApP) / (APP_post - 0.5)
   ≈ 0: old (base) axis still tracks post model well (no rotation)
   ≈ 1: old axis captures nothing in post model (strong misalignment)

DR_backward = 1 - (APp - 0.5) / (APP_base - 0.5)
            = (APP_base - APp) / (APP_base - 0.5)
   ≈ 0: new (post) axis works well on base model (compatible geometry)
   ≈ 1: new axis is useless on base model (strong misalignment)

DR = (DR_forward + DR_backward) / 2

Attack score (color) = "WMDP, Best Tamp. Attack" from LLM-GAT summary table
   (best tampering attack WMDP accuracy; higher = more knowledge retained after unlearning)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--re_band",  default="mb",   help="Probe band for RE  (default: mb)")
ap.add_argument("--re_clf",   default="lr",   help="Classifier for RE  (default: lr)")
ap.add_argument("--dr_band",  default="mb",   help="Probe band for DR  (default: mb)")
ap.add_argument("--dr_clf",   default="lr",   help="Classifier for DR  (default: lr)")
ap.add_argument("--metric",   default="acc",  help="Metric column suffix (default: acc)")
ap.add_argument("--out",      default=None,   help="Output file (default: auto)")
args = ap.parse_args()

RE_COL  = f"{args.re_band}_{args.re_clf}_{args.metric}"
DR_COL  = f"{args.dr_band}_{args.dr_clf}_{args.metric}"
# APp (cross-probe t5) uses same band/clf as DR
APp_COL = DR_COL

DATA_DIR = Path("data")

# --------------------------------------------------------------------------- #
#  Load tables                                                                 #
# --------------------------------------------------------------------------- #
t2   = pd.read_csv(DATA_DIR / "summary_table2_base_probes.csv",   index_col=0)
t3   = pd.read_csv(DATA_DIR / "summary_table3_method_probes.csv", index_col=0)
t5   = pd.read_csv(DATA_DIR / "summary_table5_cross_probes.csv",  index_col=0)
tgat = pd.read_csv(DATA_DIR / "LLM-GAT_summary_table.csv",        index_col=0)

# LLM-GAT method names → our method names
GAT_NAME_MAP = {
    "Grad Diff":  "GradDiff",
    "RMU":        "RMU",
    "RMU + LAT":  "RMU-LAT",
    "RepNoise":   "RepNoise",
    "ELM":        "ELM",
    "RR":         "RR",
    "TAR":        "TAR",
    "PB&J":       "PB&J",
}
# Build lookup: our name -> GAT attack score
gat_attack = {
    our: float(tgat.loc[gat, "WMDP, Best Tamp. Attack"])
    for gat, our in GAT_NAME_MAP.items()
}

APP_base = float(t2.loc["Base", RE_COL])
print(f"APP_base (Base->Base, {RE_COL}): {APP_base:.4f}")

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]

# --------------------------------------------------------------------------- #
#  Compute metrics                                                              #
# --------------------------------------------------------------------------- #
rows = []
for m in METHODS:
    APP_post = float(t3.loc[m, RE_COL])    # Post probe -> Post model
    ApP      = float(t2.loc[m, DR_COL])    # Base probe -> Post model
    APp      = float(t5.loc[m, APp_COL])   # Post probe -> Base model
    attack   = gat_attack[m]                   # LLM-GAT best tampering attack score

    # Representational Erasure
    RE = (APP_base - APP_post) / (APP_base - 0.5)

    # DR-forward: old axis fails on post model
    denom_fwd = APP_post - 0.5
    if abs(denom_fwd) < 1e-6:
        DR_fwd = float("nan")
    else:
        DR_fwd = (APP_post - ApP) / denom_fwd

    # DR-backward: new axis fails on base model
    # 1 - (APp - 0.5) / (APP_base - 0.5)
    DR_bwd = (APP_base - APp) / (APP_base - 0.5)

    DR = (DR_fwd + DR_bwd) / 2.0

    print(f"{m:10s}  APP_post={APP_post:.4f}  ApP={ApP:.4f}  APp={APp:.4f}"
          f"  RE={RE:+.3f}  DR_fwd={DR_fwd:+.3f}  DR_bwd={DR_bwd:+.3f}"
          f"  DR={DR:+.3f}  attack={attack:.4f}")

    rows.append(dict(method=m, APP_post=APP_post, ApP=ApP, APp=APp,
                     RE=RE, DR_fwd=DR_fwd, DR_bwd=DR_bwd, DR=DR, attack=attack))

df = pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
#  Plot helper                                                                  #
# --------------------------------------------------------------------------- #
OFFSETS = {
    "GradDiff": ( 8,  5),
    "RMU":      ( 8,  5),
    "RMU-LAT":  ( 8, -12),
    "RepNoise": ( 8,  5),
    "ELM":      ( 8, -12),
    "RR":       ( 8,  5),
    "TAR":      ( 8,  5),
    "PB&J":     ( 8,  5),
}

XLABEL = (
    "DR  --  Directional Rotation proxy\n"
    r"$\frac{1}{2}\!\left["
    r"\frac{A_{PP}^{\rm post}{-}A_{pP}}{A_{PP}^{\rm post}{-}0.5}"
    r"+\frac{A_{PP}^{\rm base}{-}A_{Pp}}{A_{PP}^{\rm base}{-}0.5}"
    r"\right]$"
)
YLABEL = (
    "RE  --  Representational Erasure\n"
    r"$\frac{A_{PP}^{\rm base} - A_{PP}^{\rm post}}{A_{PP}^{\rm base} - 0.5}$"
)

def draw_panel(ax, df, attack_col, cbar_label, shared_norm):
    cmap = plt.get_cmap("RdYlGn_r")
    sc = ax.scatter(
        df["DR"], df["RE"],
        c=df[attack_col], cmap=cmap, norm=shared_norm,
        s=220, zorder=5, edgecolors="k", linewidths=0.7,
    )
    for _, row in df.iterrows():
        ox, oy = OFFSETS.get(row["method"], (8, 5))
        ax.annotate(row["method"], xy=(row["DR"], row["RE"]),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=9, fontweight="semibold")
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    ax.set_xlabel(XLABEL, fontsize=10)
    ax.set_ylabel(YLABEL, fontsize=10)
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1, color="steelblue",  lw=0.7, ls=":", alpha=0.4, label="RE=1")
    ax.axvline(1, color="darkorange", lw=0.7, ls=":", alpha=0.4, label="DR=1")
    ax.legend(fontsize=8, loc="lower right")
    xm = (df["DR"].max() - df["DR"].min()) * 0.15 + 0.05
    ym = (df["RE"].max() - df["RE"].min()) * 0.15 + 0.05
    ax.set_xlim(df["DR"].min() - xm, df["DR"].max() + xm)
    ax.set_ylim(df["RE"].min() - ym, df["RE"].max() + ym)
    return sc

# --------------------------------------------------------------------------- #
#  Add both attack columns to df                                                #
# --------------------------------------------------------------------------- #
df["attack_tamp"]  = df["method"].map({m: float(tgat.loc[g, "WMDP, Best Tamp. Attack"])
                                        for g, m in GAT_NAME_MAP.items()})
df["attack_input"] = df["method"].map({m: float(tgat.loc[g, "WMDP, Best Input Attack"])
                                        for g, m in GAT_NAME_MAP.items()})

# Shared colour scale across both panels for fair comparison
all_vals = pd.concat([df["attack_tamp"], df["attack_input"]])
shared_norm = Normalize(vmin=all_vals.min(), vmax=all_vals.max())

# --------------------------------------------------------------------------- #
#  Side-by-side figure                                                          #
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle(
    "Hidden Knowledge After Unlearning\n"
    "Representational Erasure vs Geometric Rotation of the Knowledge Axis",
    fontsize=13, y=1.01,
)

draw_panel(axes[0], df, "attack_tamp",
           "Best Tampering Attack  (WMDP, LLM-GAT)", shared_norm)
axes[0].set_title("Color = Best Tampering Attack", fontsize=11)

draw_panel(axes[1], df, "attack_input",
           "Best Input Attack  (WMDP, LLM-GAT)", shared_norm)
axes[1].set_title("Color = Best Input Attack", fontsize=11)

plt.tight_layout()
_auto_out = f"RE_DR_{args.re_band}_{args.re_clf}_RE_{args.dr_band}_{args.dr_clf}_DR.png"
out = Path(args.out) if args.out else DATA_DIR / _auto_out
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.show()
