#!/usr/bin/env python3
"""Figure 2: WMDP-Bio accuracy against K_ext and K_int at the final checkpoint.

WMDP accuracy is four-way (chance 0.25) and the two K scores are pairwise
(chance 0.5), so they get two axes and two chance lines. Whiskers are the
standard deviation across the five folds. Prints the plotted values.

Usage: python plots/knowledge_accuracy_bars.py
"""
import matplotlib.pyplot as plt
import pandas as pd

from common import (EXT_C, INT_C, METHODS, WMDP_C, WMDP_TEXT_C, figsize,
                    k_summary, label, published_accuracy, save, use_style)

use_style()

MODELS = ["base"] + [f"{m}_ck8" for m in METHODS]
K_TOP = A_TOP = 0.96    # one shared scale: a value sits at the same height on both axes
Y_BOT = 0.15            # nothing plotted lies below 0.24


def load() -> pd.DataFrame:
    acc = published_accuracy("llm-gat")
    rows = [dict(label="Base" if mid == "base" else label(mid[:-4]),
                 wmdp=acc[mid], **k_summary(mid)) for mid in MODELS]
    d = pd.DataFrame(rows)
    # base first, then methods by decreasing K_int
    return pd.concat([d.iloc[:1], d.iloc[1:].sort_values("k_int", ascending=False)],
                     ignore_index=True)


def main():
    d = load()
    print(d.round(3).to_string(index=False))

    x = list(range(len(d)))
    fig, ax = plt.subplots(figsize=figsize(aspect=0.321))
    ax2 = ax.twinx()

    W = 0.26
    xw, xe, xi_ = [i - W for i in x], list(x), [i + W for i in x]
    ax2.bar(xw, d.wmdp, W, color=WMDP_C, zorder=3, label="WMDP-Bio accuracy")
    ax.bar(xe, d.k_ext, W, color=EXT_C, zorder=3, label=r"$K_\mathrm{ext}$")
    ax.bar(xi_, d.k_int, W, color=INT_C, zorder=3, label=r"$K_\mathrm{int}$")
    for xx, v, sd in ((xe, d.k_ext, d.k_ext_sd), (xi_, d.k_int, d.k_int_sd)):
        ax.errorbar(xx, v, yerr=sd, fmt="none", ecolor="black",
                    elinewidth=0.8, capsize=1.5, alpha=0.6, zorder=5)
    for xx, v, sd, c in ((xe, d.k_ext, d.k_ext_sd, EXT_C),
                         (xi_, d.k_int, d.k_int_sd, INT_C)):
        for xi2, vv, ss in zip(xx, v, sd):
            ax.text(xi2, vv + ss + 0.015, f"{vv:.2f}", ha="center",
                    va="bottom", fontsize=5.6, color=c, rotation=90, zorder=6)
    for xi2, vv in zip(xw, d.wmdp):
        ax2.text(xi2, vv + 0.012, f"{vv:.2f}", ha="center", va="bottom",
                 fontsize=5.6, color=WMDP_TEXT_C, rotation=90, zorder=6)

    ax.axhline(0.5, color=EXT_C, ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax2.axhline(0.25, color=WMDP_TEXT_C, ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax.text(-1.32, 0.507, "chance 0.50", fontsize=6, color=EXT_C,
            va="bottom", ha="left")
    ax2.text(-1.32, 0.256, "chance 0.25", fontsize=6, color=WMDP_TEXT_C,
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
    ax.set_xlim(-1.37, len(d) - 0.35)    # left gutter holds the chance labels
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend([h2[0], h1[0], h1[1]], [l2[0], l1[0], l1[1]], fontsize=7,
              loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=3,
              frameon=False, handletextpad=0.4, columnspacing=1.2,
              borderaxespad=0.15)
    fig.tight_layout()
    save(fig, "knowledge_accuracy_bars")


if __name__ == "__main__":
    main()
