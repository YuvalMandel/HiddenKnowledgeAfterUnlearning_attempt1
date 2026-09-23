#!/usr/bin/env python3
"""Figure 6: WMDP's own RMU checkpoints on three bases, raw and normalised.

Replaces the two separate figures that were pasted side by side. Each of those
was authored about 7 inches wide and then placed at half of a 5.5in textwidth,
so every point size in them was shrunk by ~2.5x and the bars, sized for a wide
canvas with only three groups, came out far too thick. This draws both panels
in ONE figure at the real print width, so the type renders as authored and the
bars are sized for the space they actually get.

Left: raw pairwise K on the left axis, four-way WMDP accuracy on the right,
each against its own chance floor, with a dashed marker at each group's OWN
base value -- raw K is not comparable across bases, so the drop from marker to
bar is what the panel says.

Right: the same numbers as retention, R(M) = 100*(M - chance)/(base - chance),
which IS comparable across bases. Stacked, so segment heights are the gaps.

Usage: python plots/rmu_families_two_panel.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hk_utils import iclr_figsize, use_iclr_style   # noqa: E402
from rmu_three_families import WMDP_RMU, row, INT_C, EXT_C, WMDP_C  # noqa: E402
from optml_figs import stack_order                  # noqa: E402

use_iclr_style()

ROOT = Path(__file__).resolve().parent.parent
IMGS = ROOT / "overleaf_claims" / "imgs"
NL = chr(10)


def panel_raw(ax, d):
    n = len(d)
    ax2 = ax.twinx()
    x = list(range(n))
    W = 0.24
    xw = [i - W for i in x]
    xe = list(x)
    xi_ = [i + W for i in x]

    ax2.bar(xw, d.raw_wmdp, W, color=WMDP_C, zorder=3)
    ax.bar(xe, d.raw_ke, W, color=EXT_C, zorder=3)
    ax.bar(xi_, d.raw_ki, W, color=INT_C, zorder=3)
    ax.errorbar(xe, d.raw_ke, yerr=d.ke_sd, fmt="none", ecolor="black",
                elinewidth=0.7, capsize=1.4, alpha=0.65, zorder=5)
    ax.errorbar(xi_, d.raw_ki, yerr=d.ki_sd, fmt="none", ecolor="black",
                elinewidth=0.7, capsize=1.4, alpha=0.65, zorder=5)

    # dashed base marker, one per metric per group
    for xx, col, vals, tgt in ((xw, "#5a5a5a", d.base_wmdp, ax2),
                               (xe, EXT_C, d.base_ke, ax),
                               (xi_, INT_C, d.base_ki, ax)):
        for xi2, vv in zip(xx, vals):
            tgt.plot([xi2 - W / 2, xi2 + W / 2], [vv, vv], color=col, lw=1.1,
                     ls=(0, (2.0, 1.3)), alpha=0.9, zorder=7)

    for xx, v, sd, c in ((xe, d.raw_ke, d.ke_sd, EXT_C),
                         (xi_, d.raw_ki, d.ki_sd, INT_C)):
        for xi2, vv, ss in zip(xx, v, sd):
            ax.text(xi2, vv + ss + 0.014, f"{vv:.2f}", ha="center",
                    va="bottom", fontsize=5.4, color=c, rotation=90, zorder=6)
    for xi2, vv in zip(xw, d.raw_wmdp):
        ax2.text(xi2, vv + 0.012, f"{vv:.2f}", ha="center", va="bottom",
                 fontsize=5.4, color="#5a5a5a", rotation=90, zorder=6)

    ax.axhline(0.5, color=EXT_C, ls=":", lw=0.9, alpha=0.85, zorder=1)
    ax2.axhline(0.25, color="#5a5a5a", ls=":", lw=0.9, alpha=0.85, zorder=1)
    ax.text(-1.24, 0.508, "chance 0.50", fontsize=5.8, color=EXT_C,
            va="bottom", ha="left")
    ax2.text(-1.24, 0.258, "chance 0.25", fontsize=5.8, color="#5a5a5a",
             va="bottom", ha="left")

    # shared scale, so a value reads at the same height on either axis
    ax.set_ylim(0.15, 0.95)
    ax2.set_ylim(0.15, 0.95)
    ax.set_ylabel("Knowledge: pairwise $K$", fontsize=7.5)
    ax2.set_ylabel("Accuracy: WMDP-Bio (4-way)", fontsize=7.5)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticks(x)
    ax.set_xticklabels([l + NL + "RMU" for l in d.label], fontsize=6.4)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    ax.set_xlim(-1.28, n - 0.35)   # left gutter holds the two chance labels
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(a) raw scores, each against its own floor",
                 fontsize=7.8, pad=5)


def panel_retention(ax, d):
    n = len(d)
    x = list(range(n))
    W = 0.30
    vw, ve, vi = d.wmdp.tolist(), d.k_ext.tolist(), d.k_int.tolist()

    # same construction as the OPTML retention figure: both levels run from
    # zero with the one closer to zero in front, which for 0 <= w <= e is the
    # ordinary stack and for Yi (K_ext retention -1, WMDP +11) keeps the
    # negative metric alone below the line
    for xi, w, e, i2 in zip(x, vw, ve, vi):
        zg, ze = stack_order(w, e)
        ax.bar(xi, i2 - e, W, bottom=e, color=INT_C, zorder=2)
        ax.bar(xi, w, W, color=WMDP_C, zorder=zg)
        ax.bar(xi, e, W, color=EXT_C, zorder=ze)

    def band(xi, lo, hi, colr):
        """Gap number inside the band when it is tall enough, else to the
        right. Only ever called where the band IS a visible rectangle."""
        if abs(hi - lo) >= 9:
            ax.text(xi, (lo + hi) / 2, f"{hi - lo:+.0f}", ha="center",
                    va="center", fontsize=6.2, color="white", zorder=6)
        else:
            ax.text(xi + W / 2 + 0.04, (lo + hi) / 2, f"{hi - lo:+.0f}",
                    ha="left", va="center", fontsize=6.2, color=colr,
                    zorder=6)

    for xi, a, b, c in zip(x, vw, ve, vi):
        # blue always runs K_ext -> K_int, so its height is always its band,
        # negative K_ext or not. Red is the one drawn from zero when a level
        # goes below the line, and then its rectangle is NOT the gap.
        if a >= 0 and b >= 0:
            band(xi, a, b, EXT_C)
        else:
            ax.text(xi, min(a, b, 0.0) - 1.6, f"{b - a:+.0f}", ha="center",
                    va="top", fontsize=6.2, color=EXT_C, zorder=6)
        band(xi, b, c, INT_C)
        # ticks stay at the true heights; the numbers beside them move
        ya, yb = a, b
        if abs(b - a) < 8:
            mid = (a + b) / 2          # Mixtral: 18 and 22 would collide
            ya, yb = mid - 4.0, mid + 4.0
        if abs(yb) < 3.0:
            yb = -4.5                  # Yi: -1 would sit on the zero line
        for y, ylab, colr in ((a, ya, "#5a5a5a"), (b, yb, EXT_C)):
            ax.plot([xi - W / 2 - 0.06, xi - W / 2], [y, y], color=colr,
                    lw=0.9, zorder=6, clip_on=False)
            ax.text(xi - W / 2 - 0.09, ylab, f"{y:.0f}", ha="right",
                    va="center", fontsize=6.3, color=colr, zorder=6)
        ax.text(xi, c + 2.4, f"{c:.0f}", ha="center", va="bottom",
                fontsize=7.0, color=INT_C, zorder=6)

    ax.axhline(0, color="0.35", ls=":", lw=0.8, zorder=1)
    ax.axhline(100, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.text(-0.72, 99, "own base model", fontsize=6.3, color="0.35",
            va="top", ha="left")
    # nothing can exceed its own base, so 100 is the ceiling
    ax.set_ylim(-16, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of its own base model's\nabove-chance signal retained",
                  fontsize=7.5)
    ax.set_xticks(x)
    # panel (a) carries the full names; at a third of the width these
    # would overlap, so (b) uses the family alone
    short = [l.split("-")[0] for l in d.label]
    ax.set_xticklabels([l + NL + "RMU" for l in short], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.80, n - 0.35)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(b) normalised by each base's own range",
                 fontsize=7.8, pad=5)


def main():
    d = pd.DataFrame([row(*a) for a in WMDP_RMU])
    print(d[["label", "raw_wmdp", "raw_ke", "raw_ki", "wmdp", "k_ext",
             "k_int"]].round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=iclr_figsize(aspect=0.345),
                             gridspec_kw={"width_ratios": [2, 1]})
    panel_raw(axes[0], d)
    panel_retention(axes[1], d)

    keys = [Patch(color=WMDP_C, label="WMDP-Bio accuracy"),
            Patch(color=EXT_C, label=r"$K_\mathrm{ext}$ (logit margin)"),
            Patch(color=INT_C, label=r"$K_\mathrm{int}$ (best-layer probe)"),
            Line2D([], [], color="0.35", lw=1.1, ls=(0, (2.0, 1.3)),
                   label="its own base model (panel a)")]
    fig.legend(handles=keys, fontsize=6.8, loc="lower center",
               bbox_to_anchor=(0.5, -0.035), ncol=4, frameon=False,
               handletextpad=0.4, columnspacing=1.5)
    fig.subplots_adjust(wspace=0.46)
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    for ext in ("pdf", "png"):
        p = ROOT / "plots" / f"rmu_families_two_panel.{ext}"
        fig.savefig(p, bbox_inches="tight",
                    **({"dpi": 300} if ext == "png" else {}))
        print("wrote", p)
    if IMGS.is_dir():
        fig.savefig(IMGS / "rmu_families_two_panel.pdf", bbox_inches="tight")
        print("wrote", IMGS / "rmu_families_two_panel.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
