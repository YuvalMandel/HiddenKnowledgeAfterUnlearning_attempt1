#!/usr/bin/env python3
"""Raw scores: pairwise knowledge on the left axis, WMDP accuracy on the right.

The three metrics do not share a scale -- WMDP is four-way (chance 0.25), the two
K scores are pairwise (chance 0.5) -- so they get two axes and two chance lines,
each metric read against its own floor.

Three bars per method, layered rather than side by side: K_int widest at the
back, K_ext inside it, WMDP narrowest in front.

The values go in a strip beneath rather than on the bars. On the base model all
three series finish within 0.03 of each other, so bar-top labels collide there
whatever offsets they are given; the strip is collision-proof, keeps the bars
clean, and matches `retention_three_metric.py` so the two figures read as a
pair.

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
K_TOP, A_TOP = 0.92, 0.80


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
    fig, (ax, tb) = plt.subplots(
        2, 1, figsize=iclr_figsize(aspect=5.2 / 7, width_frac=0.80),
        gridspec_kw=dict(height_ratios=[2.75, 1.25], hspace=0.08))
    ax2 = ax.twinx()

    # back to front: widest bar is the tallest series, so nothing is hidden
    ax.bar(x, d.k_int, 0.78, color=INT_C, alpha=0.9, zorder=2,
           label=r"$K_\mathrm{int}$ (best-layer probe)")
    ax.bar(x, d.k_ext, 0.46, color=EXT_C, alpha=0.95, zorder=3,
           label=r"$K_\mathrm{ext}$ (logit margin)")
    ax2.bar(x, d.wmdp, 0.17, color=WMDP_C, alpha=1.0, zorder=4,
            label="WMDP-Bio accuracy (right axis)")
    ax.errorbar(x, d.k_int, yerr=d.k_int_sd, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=1.5, alpha=0.55, zorder=5)
    ax.errorbar(x, d.k_ext, yerr=d.k_ext_sd, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=1.5, alpha=0.55, zorder=5)

    ax.axhline(0.5, color=EXT_C, ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax2.axhline(0.25, color="#5a5a5a", ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax.text(-1.50, 0.507, "chance 0.50", fontsize=6, color=EXT_C,
            va="bottom", ha="left")
    ax2.text(-1.50, 0.256, "chance 0.25", fontsize=6, color="#5a5a5a",
             va="bottom", ha="left")

    ax.set_ylim(0, K_TOP)
    ax2.set_ylim(0, A_TOP)
    ax.set_ylabel("Knowledge: pairwise $K$", fontsize=8)
    ax2.set_ylabel("Accuracy: WMDP-Bio (4-way)", fontsize=8)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(labelsize=7.5)
    ax2.tick_params(labelsize=7.5)
    ax.set_xlim(-1.55, len(d) - 0.35)   # left gutter holds the
    #                                     two chance labels clear
    #                                     of the base-model bars
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handletextpad=0.4, columnspacing=1.2)
    # ---- numeric strip, same idiom as the retention figure
    LEFT = -0.72
    strip = [("WMDP-Bio", d.wmdp, "%.2f", "#5a5a5a"),
             (r"$K_\mathrm{ext}$", d.k_ext, "%.3f", EXT_C),
             (r"$K_\mathrm{int}$", d.k_int, "%.3f", INT_C)]
    tb.set_xlim(ax.get_xlim())
    tb.set_ylim(len(strip) + 1.9, -0.8)
    tb.axis("off")
    tb.plot([LEFT - 0.75, len(d) - 0.45], [-0.55, -0.55], color="0.8", lw=0.6)
    for r, (name, vals, fmt, colr) in enumerate(strip):
        tb.text(LEFT, r, name, fontsize=6.5, ha="right", va="center",
                color=colr)
        for xi, v in zip(x, vals):
            tb.text(xi, r, fmt % v, fontsize=6.4, ha="center", va="center",
                    color=colr)
    for xi, c in zip(x, d.label):
        tb.text(xi, len(strip) - 0.65, c, fontsize=6.8, ha="center", va="top",
                rotation=90, style="italic" if c == "Base" else "normal")
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
