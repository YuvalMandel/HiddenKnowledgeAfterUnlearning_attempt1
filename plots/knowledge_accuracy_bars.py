#!/usr/bin/env python3
"""Raw scores: pairwise knowledge on the left axis, WMDP accuracy on the right.

The three metrics do not share a scale -- WMDP is four-way (chance 0.25), the two
K scores are pairwise (chance 0.5) -- so they get two axes and two chance lines,
each metric read against its own floor.

Three equal-width bars per method, side by side, styled after the pair-
composition figure: uniform bars, legend underneath.

Every value is printed above its own bar, rotated upright so that neighbouring
labels in a group cannot touch. (This is why the bars are side by side and not
layered: layered bars share one x, and on the base model all three series finish
within 0.03 of each other, so their labels collide whatever offsets they get.)

Note these three bars are NOT stackable: WMDP is a four-way accuracy on its own
axis and the K scores are pairwise, so they are three measurements, not parts of
one whole. The additive decomposition lives in the retention figure, where the
segments really do sum.

Companion to `retention_three_metric.py`, which normalises these same numbers.

Usage: python plots/knowledge_accuracy_bars.py
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

INT_C, EXT_C, WMDP_C = "#2166ac", "#d6604d", "#8c8c8c"
# separate ceilings keep the two chance lines at different heights, which is the
# point of the second axis: each metric is read against its own floor
# One shared scale: A_TOP == K_TOP so 0.8 sits at the same height on both
# axes. The two chance lines (0.5 pairwise, 0.25 four-way) still land at
# different heights, which is what the second axis is for.
K_TOP, A_TOP = 0.96, 0.96
# both axes start at 0.15, not 0: nothing lives below it (the lowest value is
# ELM's 0.24 WMDP accuracy) and the empty band only cost height
Y_BOT = 0.15


def load():
    frames = [pd.read_parquet(OUT_DIR / m / "k_scores.parquet")
              for m, _ in MODELS if (OUT_DIR / m / "k_scores.parquet").exists()]
    df = pd.concat(frames, ignore_index=True)
    cv = df[(df.split_type == "cv") & (df.domain == "bio") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    rows = []
    for mid, label in MODELS:
        sub = cv[cv.model_id == mid]
        if sub.empty:
            continue
        f = sub.groupby("fold").agg(ki=("k_internal", "mean"),
                                    ke=("k_external", "mean"))
        rows.append(dict(label=label, k_int=f.ki.mean(), k_ext=f.ke.mean(),
                         k_int_sd=f.ki.std(ddof=1), k_ext_sd=f.ke.std(ddof=1),
                         wmdp=WMDP[label]))
    return pd.DataFrame(rows)


def main():
    d = load()
    base = d[d.label == "Base"]
    rest = d[d.label != "Base"].sort_values("k_int", ascending=False)
    d = pd.concat([base, rest], ignore_index=True)
    print(d.round(3).to_string(index=False))

    x = list(range(len(d)))
    fig, ax = plt.subplots(figsize=iclr_figsize(aspect=0.321,
                                               width_frac=1.0))
    ax2 = ax.twinx()

    W = 0.26
    xw = [i - W for i in x]
    xe = list(x)
    xi_ = [i + W for i in x]
    ax2.bar(xw, d.wmdp, W, color=WMDP_C, zorder=3,
            label="WMDP-Bio accuracy")
    ax.bar(xe, d.k_ext, W, color=EXT_C, zorder=3,
           label=r"$K_\mathrm{ext}$")
    ax.bar(xi_, d.k_int, W, color=INT_C, zorder=3,
           label=r"$K_\mathrm{int}$")
    ax.errorbar(xe, d.k_ext, yerr=d.k_ext_sd, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=1.5, alpha=0.6, zorder=5)
    ax.errorbar(xi_, d.k_int, yerr=d.k_int_sd, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=1.5, alpha=0.6, zorder=5)
    for xx, v, sd, c in ((xe, d.k_ext, d.k_ext_sd, EXT_C),
                         (xi_, d.k_int, d.k_int_sd, INT_C)):
        for xi2, vv, ss in zip(xx, v, sd):
            ax.text(xi2, vv + ss + 0.015, f"{vv:.2f}", ha="center",
                    va="bottom", fontsize=5.6, color=c, rotation=90, zorder=6)
    for xi2, vv in zip(xw, d.wmdp):
        ax2.text(xi2, vv + 0.012, f"{vv:.2f}", ha="center", va="bottom",
                 fontsize=5.6, color="#5a5a5a", rotation=90, zorder=6)

    ax.axhline(0.5, color=EXT_C, ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax2.axhline(0.25, color="#5a5a5a", ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax.text(-1.32, 0.507, "chance 0.50", fontsize=6, color=EXT_C,
            va="bottom", ha="left")
    ax2.text(-1.32, 0.256, "chance 0.25", fontsize=6, color="#5a5a5a",
             va="bottom", ha="left")

    ax.set_ylim(Y_BOT, K_TOP)
    ax2.set_ylim(Y_BOT, A_TOP)
    ax.set_ylabel("Knowledge: pairwise $K$", fontsize=8)
    ax2.set_ylabel("Accuracy: WMDP-Bio (4-way)", fontsize=8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticks(x)
    ax.set_xticklabels(d.label, rotation=0, ha="center", fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax2.tick_params(labelsize=7.5)
    ax.set_xlim(-1.37, len(d) - 0.35)   # left gutter holds the
    #                                     two chance labels clear
    #                                     of the base-model bars
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # One row. The labels lost their parentheticals, so three entries fit
    # across; at ncol=3 column-major fill is the same as row order.
    ho = [h2[0], h1[0], h1[1]]
    lo = [l2[0], l1[0], l1[1]]
    ax.legend(ho, lo, fontsize=7, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
              handletextpad=0.4, columnspacing=1.2, borderaxespad=0.15)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = ROOT / "plots" / f"knowledge_accuracy_bars.{ext}"
        fig.savefig(p, bbox_inches="tight",
                    **({"dpi": 200} if ext == "png" else {}))
        print("wrote", p)
    if IMGS.is_dir():
        fig.savefig(IMGS / "knowledge_accuracy_bars.pdf", bbox_inches="tight")
        print("wrote", IMGS / "knowledge_accuracy_bars.pdf")


if __name__ == "__main__":
    main()
