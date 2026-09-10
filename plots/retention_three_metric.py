#!/usr/bin/env python3
"""The three-metric retention figure: WMDP vs K_ext vs K_int on one scale.

Table `tab:wmdp-k-comparison` says the same thing in seven columns. The single
comparable quantity across three metrics with three different chance floors and
three different base-model ceilings is the fraction of the base model's
above-chance signal that survives unlearning:

    R(M) = 100 * (M - M_chance) / (M_base - M_chance)

so every metric runs 0 (at chance) to 100 (base model) and the three become
directly comparable. Each method is one row carrying three marks; the row reads
left to right as "what the standard benchmark sees" -> "what the model's own
readout gives away" -> "what is actually still in there".

A grouped bar chart of the same numbers needs 24 bars and reads as texture; the
dot plot keeps one row per method and makes the WMDP->K_int span the visual
quantity, which is the claim.

K_ext and K_int come from the same parquet query as
`plots/k_int_vs_ext_kfold_single_bestlayer.pdf`, so the two figures and the
table cannot disagree. WMDP accuracies are LLM-GAT's published values.

Usage: python plots/retention_three_metric.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hk_utils import iclr_figsize, use_iclr_style  # noqa: E402

use_iclr_style()

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "inside_out_out"
IMGS = ROOT / "overleaf_claims" / "imgs"

MODELS = [("base", "Base"), ("GradDiff_ck8", "GradDiff"), ("PB_J_ck8", "PB&J"),
          ("RMU_ck8", "RMU"), ("RMU-LAT_ck8", "RMU-LAT"),
          ("RepNoise_ck8", "RepNoise"), ("ELM_ck8", "ELM"), ("RR_ck8", "RR"),
          ("TAR_ck8", "TAR")]
# data/LLM-GAT_summary_table.csv -- GAT's published 4-way MCQ accuracies
WMDP = {"GradDiff": 0.25, "RMU": 0.26, "RMU-LAT": 0.32, "RepNoise": 0.29,
        "ELM": 0.24, "RR": 0.26, "TAR": 0.28, "PB&J": 0.31, "Base": 0.70}

INT_C, EXT_C, WMDP_C = "#2166ac", "#d6604d", "#4d4d4d"


def load():
    frames = [pd.read_parquet(OUT_DIR / m / "k_scores.parquet")
              for m, _ in MODELS if (OUT_DIR / m / "k_scores.parquet").exists()]
    df = pd.concat(frames, ignore_index=True)
    cv = df[(df.split_type == "cv") & (df.domain == "bio") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    rows = {}
    for mid, label in MODELS:
        sub = cv[cv.model_id == mid]
        if sub.empty:
            continue
        f = sub.groupby("fold").agg(ki=("k_internal", "mean"),
                                    ke=("k_external", "mean"))
        rows[label] = dict(k_int=f.ki.mean(), k_ext=f.ke.mean(),
                           k_int_sd=f.ki.std(ddof=1), k_ext_sd=f.ke.std(ddof=1))
    return rows


def main():
    raw = load()
    base = raw["Base"]
    # R(M): 0 = chance, 100 = base model, per metric.
    def R(v, chance, top):
        return 100.0 * (v - chance) / (top - chance)

    rows = []
    for _, label in MODELS:
        if label == "Base" or label not in raw:
            continue
        rows.append(dict(
            label=label,
            wmdp=R(WMDP[label], 0.25, WMDP["Base"]),
            k_ext=R(raw[label]["k_ext"], 0.5, base["k_ext"]),
            k_int=R(raw[label]["k_int"], 0.5, base["k_int"]),
            # the denominator is held fixed, so these are the fold spread of
            # the method only -- not the ratio's full uncertainty
            k_ext_sd=100 * raw[label]["k_ext_sd"] / (base["k_ext"] - 0.5),
            k_int_sd=100 * raw[label]["k_int_sd"] / (base["k_int"] - 0.5)))
    d = (pd.DataFrame(rows).sort_values("k_int", ascending=False)
     .reset_index(drop=True))
    mean = d[["wmdp", "k_ext", "k_int"]].mean()
    print(d.round(1).to_string(index=False))
    print("\nmean:", mean.round(1).to_dict(),
          "  (table says 5.8 / 32.1 / 66.7)")

    fig, ax = plt.subplots(figsize=iclr_figsize(aspect=4.2 / 7,
                                                width_frac=0.80))
    y = range(len(d))
    # connector first, so the row reads as one span
    for i, r in d.iterrows():
        ax.plot([r.wmdp, r.k_int], [i, i], color="0.90", lw=4.0, zorder=1,
                solid_capstyle="round")
    ax.scatter(d.wmdp, y, s=26, marker="o", facecolors="white",
               edgecolors=WMDP_C, linewidths=1.1, zorder=3,
               label="WMDP-Bio accuracy")
    ax.errorbar(d.k_ext, y, xerr=d.k_ext_sd, fmt="none", ecolor=EXT_C,
                elinewidth=1.0, capsize=0, alpha=0.9, zorder=2)
    ax.errorbar(d.k_int, y, xerr=d.k_int_sd, fmt="none", ecolor=INT_C,
                elinewidth=1.0, capsize=0, alpha=0.9, zorder=2)
    ax.scatter(d.k_ext, y, s=26, marker="D", color=EXT_C, zorder=3,
               label=r"$K_\mathrm{ext}$ (logit margin)")
    ax.scatter(d.k_int, y, s=34, marker="o", color=INT_C, zorder=3,
               label=r"$K_\mathrm{int}$ (best-layer probe)")

    # mean row, set apart below the rule
    ym = len(d) + 0.6
    ax.axhline(len(d) - 0.5 + 0.05, color="0.85", lw=0.7)
    ax.plot([mean.wmdp, mean.k_int], [ym, ym], color="0.90", lw=4.0, zorder=1)
    ax.scatter([mean.wmdp], [ym], s=26, marker="o", facecolors="white",
               edgecolors=WMDP_C, linewidths=1.1, zorder=3)
    ax.scatter([mean.k_ext], [ym], s=26, marker="D", color=EXT_C, zorder=3)
    ax.scatter([mean.k_int], [ym], s=34, marker="o", color=INT_C, zorder=3)
    for v, c in ((mean.wmdp, WMDP_C), (mean.k_ext, EXT_C), (mean.k_int, INT_C)):
        ax.annotate(f"{v:.0f}", (v, ym), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color=c)

    ax.axvline(0, color="0.35", ls=":", lw=0.8, zorder=0)
    ax.axvline(100, color="0.35", ls="--", lw=0.8, zorder=0)
    ax.set_yticks(list(y) + [ym])
    ax.set_yticklabels(list(d.label) + ["mean"], fontsize=8)
    ax.get_yticklabels()[-1].set_style("italic")
    ax.set_ylim(ym + 1.9, -0.7)
    ax.set_xlim(-9, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("% of the base model's above-chance signal retained",
                  fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.text(0, ym + 1.35, "chance", fontsize=7, color="0.35", ha="center")
    ax.text(100, ym + 1.35, "base model", fontsize=7, color="0.35",
            ha="center")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", ls=":", lw=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False, handletextpad=0.35, columnspacing=1.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = ROOT / "plots" / f"retention_three_metric.{ext}"
        fig.savefig(p, bbox_inches="tight", **({"dpi": 200} if ext == "png"
                                               else {}))
        print("wrote", p)
    if IMGS.is_dir():
        fig.savefig(IMGS / "retention_three_metric.pdf", bbox_inches="tight")
        print("wrote", IMGS / "retention_three_metric.pdf")


if __name__ == "__main__":
    main()
