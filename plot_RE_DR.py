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

K-fold mode (--kfold):
  RE is recomputed from the 5-fold aggregated probe accuracies in
  kfold_table3_probes.csv (mean ± CI95).  DR still uses single-fold Tables 2 & 5
  (cross-probe eval is not part of the kfold pipeline).  Vertical error bars
  show ±CI95 on the RE axis.
"""

import argparse
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path

DATA_DIR = Path("data")

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--re_band",  default="mb",   help="Probe band for RE  (default: mb)")
ap.add_argument("--re_clf",   default="lr",   help="Classifier for RE  (default: lr)")
ap.add_argument("--dr_band",  default="mb",   help="Probe band for DR  (default: mb)")
ap.add_argument("--dr_clf",   default="lr",   help="Classifier for DR  (default: lr)")
ap.add_argument("--metric",   default="acc",  help="Metric column suffix (default: acc)")
ap.add_argument("--out",      default=None,   help="Output file (default: auto)")
ap.add_argument("--kfold",    action="store_true", default=False,
                help=(
                    "Use 5-fold aggregated RE (mean ± CI95) from kfold_table3_probes.csv. "
                    "DR still comes from single-fold Tables 2 & 5. "
                    "Adds vertical error bars on RE axis."
                ))
args = ap.parse_args()

RE_COL  = f"{args.re_band}_{args.re_clf}_{args.metric}"
DR_COL  = f"{args.dr_band}_{args.dr_clf}_{args.metric}"
APp_COL = DR_COL

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
gat_attack = {
    our: float(tgat.loc[gat, "WMDP, Best Tamp. Attack"])
    for gat, our in GAT_NAME_MAP.items()
}

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]

# --------------------------------------------------------------------------- #
#  Optionally load kfold aggregated values for RE                              #
# --------------------------------------------------------------------------- #
kf_acc: dict = {}          # model → (mean_acc, ci95_acc)
kf_acc_col_mean = f"{args.re_band}_{args.re_clf}_{args.metric}_mean"
kf_acc_col_ci95 = f"{args.re_band}_{args.re_clf}_{args.metric}_ci95"

if args.kfold:
    kf_path = DATA_DIR / "kfold_table3_probes.csv"
    if not kf_path.exists():
        raise FileNotFoundError(
            f"{kf_path} not found. "
            "Run: python kfold_probe.py --stage kfold_tables"
        )
    # kfold model keys: "base" for Base, method names otherwise
    _KF_KEY = {"Base": "base", **{m: m for m in METHODS}}
    with open(kf_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mdl  = row["model"]
            mean = row.get(kf_acc_col_mean, "")
            ci95 = row.get(kf_acc_col_ci95, "")
            try:
                kf_acc[mdl] = (float(mean), float(ci95))
            except (ValueError, TypeError):
                pass

    # APP_base from kfold
    if "base" in kf_acc:
        APP_base_kf, APP_base_ci = kf_acc["base"]
        print(f"APP_base (kfold mean, {RE_COL}): {APP_base_kf:.4f} ± {APP_base_ci:.4f}")
    else:
        raise KeyError("'base' row not found in kfold_table3_probes.csv")
else:
    APP_base_kf = APP_base_ci = None

APP_base_sf = float(t2.loc["Base", RE_COL])
APP_base    = APP_base_kf if args.kfold else APP_base_sf
print(f"APP_base (single-fold, {RE_COL}): {APP_base_sf:.4f}")

# --------------------------------------------------------------------------- #
#  Compute metrics                                                              #
# --------------------------------------------------------------------------- #
rows = []
for m in METHODS:
    # DR always from single-fold tables
    APP_post_sf = float(t3.loc[m, RE_COL])
    ApP         = float(t2.loc[m, DR_COL])
    APp         = float(t5.loc[m, APp_COL])
    attack      = gat_attack[m]

    # RE
    if args.kfold and m in kf_acc:
        APP_post_kf, APP_post_ci = kf_acc[m]
        RE_mean = (APP_base_kf - APP_post_kf) / (APP_base_kf - 0.5)
        # Error propagation (first-order): δRE ≈ sqrt(δpost² + δbase²) / (APP_base - 0.5)
        denom   = APP_base_kf - 0.5
        RE_ci95 = float(np.sqrt(APP_post_ci**2 + APP_base_ci**2) / abs(denom)) if abs(denom) > 1e-6 else float("nan")
        RE      = RE_mean
    else:
        APP_post_kf = APP_post_sf
        RE = (APP_base - APP_post_sf) / (APP_base - 0.5)
        RE_ci95 = float("nan")

    # DR-forward: old axis fails on post model
    denom_fwd = APP_post_sf - 0.5
    DR_fwd = (APP_post_sf - ApP) / denom_fwd if abs(denom_fwd) > 1e-6 else float("nan")

    # DR-backward: new axis fails on base model
    DR_bwd = (APP_base_sf - APp) / (APP_base_sf - 0.5)

    DR = (DR_fwd + DR_bwd) / 2.0

    print(f"{m:10s}  APP_post={APP_post_kf:.4f}  ApP={ApP:.4f}  APp={APp:.4f}"
          f"  RE={RE:+.3f}  RE_ci95={'nan' if np.isnan(RE_ci95) else f'{RE_ci95:.3f}'}"
          f"  DR_fwd={DR_fwd:+.3f}  DR_bwd={DR_bwd:+.3f}"
          f"  DR={DR:+.3f}  attack={attack:.4f}")

    rows.append(dict(method=m, APP_post=APP_post_kf, ApP=ApP, APp=APp,
                     RE=RE, RE_ci95=RE_ci95,
                     DR_fwd=DR_fwd, DR_bwd=DR_bwd, DR=DR, attack=attack))

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
if args.kfold:
    YLABEL += "\n(5-fold mean ± 95 % CI)"


def draw_panel(ax, df, attack_col, cbar_label, shared_norm, show_re_ci=False):
    cmap = plt.get_cmap("RdYlGn_r")
    sc = ax.scatter(
        df["DR"], df["RE"],
        c=df[attack_col], cmap=cmap, norm=shared_norm,
        s=220, zorder=5, edgecolors="k", linewidths=0.7,
    )
    if show_re_ci:
        for _, row in df.iterrows():
            ci = row.get("RE_ci95", float("nan"))
            if not np.isnan(ci):
                ax.errorbar(row["DR"], row["RE"],
                            yerr=ci, fmt="none",
                            ecolor="black", elinewidth=1.2, capsize=4, zorder=6)
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
all_vals   = pd.concat([df["attack_tamp"], df["attack_input"]])
shared_norm = Normalize(vmin=all_vals.min(), vmax=all_vals.max())

# --------------------------------------------------------------------------- #
#  Side-by-side figure                                                          #
# --------------------------------------------------------------------------- #
kfold_note = "  (RE = 5-fold mean ± CI95;  DR = single-fold)" if args.kfold else ""
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle(
    "Hidden Knowledge After Unlearning\n"
    "Representational Erasure vs Geometric Rotation of the Knowledge Axis"
    + kfold_note,
    fontsize=13, y=1.01,
)

draw_panel(axes[0], df, "attack_tamp",
           "Best Tampering Attack  (WMDP, LLM-GAT)", shared_norm,
           show_re_ci=args.kfold)
axes[0].set_title("Color = Best Tampering Attack", fontsize=11)

draw_panel(axes[1], df, "attack_input",
           "Best Input Attack  (WMDP, LLM-GAT)", shared_norm,
           show_re_ci=args.kfold)
axes[1].set_title("Color = Best Input Attack", fontsize=11)

plt.tight_layout()
kfold_tag = "_kfold" if args.kfold else ""
_auto_out = f"RE_DR_{args.re_band}_{args.re_clf}_RE_{args.dr_band}_{args.dr_clf}_DR{kfold_tag}.png"
out = Path(args.out) if args.out else DATA_DIR / _auto_out
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
