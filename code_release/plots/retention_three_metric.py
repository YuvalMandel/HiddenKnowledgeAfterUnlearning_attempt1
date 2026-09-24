#!/usr/bin/env python3
"""Figure 3 and Table 3: share of the base model's above-chance signal retained.

Each metric M is put on a common scale, R(M) = 100 (M - chance) / (base - chance),
so 0 is chance and 100 is the base model, for WMDP-Bio accuracy (chance 0.25)
and for K_ext and K_int (chance 0.5).

Prints Table 3 (raw values with R in parentheses, the mean over the eight
methods and the sample standard deviation across them), then draws Figure 3.
In the figure R is floored at 0, since a score below chance carries no
evidence; the table reports it unfloored (ELM's WMDP accuracy, 0.24, is -2.2).

Usage: python plots/retention_three_metric.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from common import (EXT_C, INT_C, K_CHANCE, METHODS, WMDP_C, WMDP_CHANCE,
                    WMDP_TEXT_C, figsize, k_summary, label, published_accuracy,
                    retention, save, use_style)

use_style()


def table() -> pd.DataFrame:
    """Table 3: one row per method, raw scores and their retention."""
    acc = published_accuracy("llm-gat")
    base = k_summary("base")
    rows = []
    for m in METHODS:
        mid = f"{m}_ck8"
        k = k_summary(mid)
        rows.append(dict(
            label=label(m), wmdp=acc[mid], k_ext=k["k_ext"], k_int=k["k_int"],
            r_wmdp=retention(acc[mid], WMDP_CHANCE, acc["base"]),
            r_ext=retention(k["k_ext"], K_CHANCE, base["k_ext"]),
            r_int=retention(k["k_int"], K_CHANCE, base["k_int"])))
    return pd.DataFrame(rows)


def print_table(d: pd.DataFrame) -> None:
    acc, base = published_accuracy("llm-gat"), k_summary("base")
    print("Table 3  (raw, % above-chance signal retained)")
    print(f"{'Base':<16}{acc['base']:.3f} (100%)  {base['k_ext']:.3f} (100%)  "
          f"{base['k_int']:.3f} (100%)")

    def line(name, r):
        print(f"{name:<16}{r.wmdp:.3f} ({r.r_wmdp:.1f}%)  "
              f"{r.k_ext:.3f} ({r.r_ext:.1f}%)  {r.k_int:.3f} ({r.r_int:.1f}%)")
    for r in d.itertuples():
        line(r.label, r)
    line("Unlearned mean", d.mean(numeric_only=True))
    sd = d.std(numeric_only=True, ddof=1)
    print(f"{'Across-method SD':<16}{sd.wmdp:.3f} ({sd.r_wmdp:.1f} pp)  "
          f"{sd.k_ext:.3f} ({sd.r_ext:.1f} pp)  {sd.k_int:.3f} ({sd.r_int:.1f} pp)")


def main():
    d = table()
    print_table(d)

    d = d.sort_values("r_int", ascending=False).reset_index(drop=True)
    floor = lambda v: max(0.0, v)             # noqa: E731
    cols = list(d.label) + ["mean"]
    vw = [floor(v) for v in d.r_wmdp] + [floor(d.r_wmdp.clip(lower=0).mean())]
    ve = [floor(v) for v in d.r_ext] + [floor(d.r_ext.clip(lower=0).mean())]
    vi = [floor(v) for v in d.r_int] + [floor(d.r_int.clip(lower=0).mean())]
    x = list(range(len(cols)))
    x[-1] += 0.6                               # set the mean column apart
    W = 0.46

    fig, ax = plt.subplots(figsize=figsize(aspect=0.60 * 3.6 / 7))
    # Every level runs from zero; the shortest is drawn in front, so none is hidden.
    for xi, w, e, i2 in zip(x, vw, ve, vi):
        trio = sorted(((w, WMDP_C), (e, EXT_C), (i2, INT_C)), key=lambda t: -t[0])
        for z, (v, colour) in enumerate(trio, start=2):
            ax.bar(xi, v, W, color=colour, zorder=z)

    # K_int total above the bar; WMDP and K_ext levels as ticks on the left,
    # nudged apart when two levels are too close to print both.
    for xi, a_, b_, c_ in zip(x, vw, ve, vi):
        ax.text(xi, c_ + 2.5, f"{c_:.0f}", ha="center", va="bottom",
                fontsize=6.6, color=INT_C, zorder=6)
        ya, yb = a_, b_
        if abs(b_ - a_) < 7:
            mid = (a_ + b_) / 2
            ya, yb = mid - 3.6, mid + 3.6
        elif a_ < 5:
            ya = 4.5 + 2 * a_                  # lift off the axis
        for y, ylab, colr in ((a_, ya, WMDP_TEXT_C), (b_, yb, EXT_C)):
            ax.plot([xi - W / 2 - 0.06, xi - W / 2], [y, y], color=colr,
                    lw=0.9, zorder=6, clip_on=False)
            ax.text(xi - W / 2 - 0.09, ylab, f"{y:.0f}", ha="right",
                    va="center", fontsize=6.3, color=colr, zorder=6)

    ax.axhline(0, color="0.35", ls=":", lw=0.8, zorder=1)
    ax.axhline(100, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.text(-0.95, 101, "base model", fontsize=6.3, color="0.35",
            va="bottom", ha="left")
    ax.set_ylim(0, 118)                        # headroom above 100 holds the legend
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of base model's above-chance\nsignal retained", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=0, ha="center", fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(-1.0, x[-1] + 0.75)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    keys = [Patch(color=WMDP_C, label="WMDP-Bio accuracy"),
            Patch(color=EXT_C, label=r"$K_\mathrm{ext}$ (logit margin)"),
            Patch(color=INT_C, label=r"$K_\mathrm{int}$ (best-layer probe)")]
    ax.legend(handles=keys, fontsize=7, loc="upper right",
              bbox_to_anchor=(1.0, 1.02), ncol=3, frameon=False,
              handletextpad=0.4, columnspacing=1.4, borderaxespad=0.0)
    fig.tight_layout()
    save(fig, "retention_three_metric")


if __name__ == "__main__":
    main()
