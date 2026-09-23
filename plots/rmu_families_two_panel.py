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
import os
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
from rmu_three_families import WMDP_RMU, row, R, OUT_DIR, INT_C, EXT_C, WMDP_C  # noqa: E402

DOMAIN = os.environ.get("FIG_DOMAIN", "bio")

# WMDP's own four-way accuracies, Table 2 of arXiv:2403.03218
CYBER_ACC = {  # label: (rmu, base)
    "Zephyr-7B":    (0.282, 0.440),
    "Mixtral-8x7B": (0.308, 0.520),
    "Yi-34B":       (0.290, 0.497),
}


def k_of_cyber(mid):
    d = pd.read_parquet(OUT_DIR / mid / "k_scores_cyber.parquet")
    cv = d[(d.split_type == "cv") & (d.clf == "LR")
           & (d.probe_type == "own") & (d.layer_config == "best_layer")]
    f = cv.groupby("fold").agg(ki=("k_internal", "mean"),
                               ke=("k_external", "mean"))
    return dict(k_int=f.ki.mean(), k_ext=f.ke.mean(),
                k_int_sd=f.ki.std(ddof=1), k_ext_sd=f.ke.std(ddof=1))


def row_cyber(label, rmu, _acc, bid, _bacc, src):
    acc, bacc = CYBER_ACC[label]
    r, b = k_of_cyber(rmu), k_of_cyber(bid)
    return dict(label=label, src=src,
                wmdp=R(acc, 0.25, bacc),
                k_ext=R(r["k_ext"], 0.5, b["k_ext"]),
                k_int=R(r["k_int"], 0.5, b["k_int"]),
                raw_ke=r["k_ext"], raw_ki=r["k_int"],
                ke_sd=r["k_ext_sd"], ki_sd=r["k_int_sd"],
                base_ke=b["k_ext"], base_ki=b["k_int"],
                raw_wmdp=acc, base_wmdp=bacc)
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

    # bio (fig 5) keeps the numbers centred; cyber (fig 6) pushes them to
    # opposite sides of their bars, where the labels sit closer together
    off = 0.0 if DOMAIN == "bio" else 0.62 * W
    for xx, v, sd, c, dx in ((xe, d.raw_ke, d.ke_sd, EXT_C, -off),
                             (xi_, d.raw_ki, d.ki_sd, INT_C, +off)):
        for xi2, vv, ss in zip(xx, v, sd):
            ax.text(xi2 + dx, vv + ss + 0.014, f"{vv:.2f}", ha="center",
                    va="bottom", fontsize=6.2, color=c, rotation=90, zorder=6)
    for xi2, vv in zip(xw, d.raw_wmdp):
        ax2.text(xi2, vv + 0.012, f"{vv:.2f}", ha="center", va="bottom",
                 fontsize=6.2, color="#5a5a5a", rotation=90, zorder=6)

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
    ax2.set_ylabel(f"Accuracy: WMDP-{DOMAIN.capitalize()} (4-way)", fontsize=7.5)
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
    W = 0.45
    vw, ve, vi = d.wmdp.tolist(), d.k_ext.tolist(), d.k_int.tolist()

    # every level runs from zero and the shortest is drawn in front, so
    # none is hidden behind a taller one. Retention is floored at 0, so
    # there are no negative levels to special-case.
    for xi, w, e, i2 in zip(x, vw, ve, vi):
        trio = sorted(((w, WMDP_C), (e, EXT_C), (i2, INT_C)),
                      key=lambda t: -t[0])
        for z, (v, colour) in enumerate(trio, start=2):
            ax.bar(xi, v, W, color=colour, zorder=z)

    # each bar carries its own value: with nothing stacked there are no
    # segment heights to report. Ticks sit at the true height on the left,
    # numbers beside them, nudged apart only when two levels are close.
    for xi, a_, b_, c_ in zip(x, vw, ve, vi):
        ax.text(xi, c_ + 2.0, f'{c_:.0f}', ha='center', va='bottom',
                fontsize=6.2, color=INT_C, zorder=6)
        ya, yb = a_, b_
        if abs(b_ - a_) < 8:
            mid = (a_ + b_) / 2
            ya, yb = mid - 4.0, mid + 4.0
        # a clamped level reads 0; lift its number clear of the zero line
        if a_ < 0.5:
            ya = 3.0
        if b_ < 0.5:
            yb = 3.0
        for y, ylab, colr in ((a_, ya, WMDP_C), (b_, yb, EXT_C)):
            ax.plot([xi - W / 2 - 0.06, xi - W / 2], [y, y], color=colr,
                    lw=0.9, zorder=6, clip_on=False)
            ax.text(xi - W / 2 - 0.09, ylab, f'{y:.0f}', ha='right',
                    va='center', fontsize=6.2, color=colr, zorder=6)

    ax.axhline(0, color="0.35", ls=":", lw=0.8, zorder=1)
    ax.axhline(100, color="0.35", ls="--", lw=0.8, zorder=1)
    # right-aligned: on cyber the leftmost bar reaches 90 and collided
    # with a left-aligned caption
    ax.text(n - 0.45, 99, "own base model", fontsize=6.3, color="0.35",
            va="top", ha="right")
    # nothing can exceed its own base, so 100 is the ceiling
    ax.set_ylim(0, 100)
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
    maker = row_cyber if DOMAIN == "cyber" else row
    d = pd.DataFrame([maker(*a) for a in WMDP_RMU])
    print(d[["label", "raw_wmdp", "raw_ke", "raw_ki", "wmdp", "k_ext",
             "k_int"]].round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=iclr_figsize(aspect=0.362),
                             gridspec_kw={"width_ratios": [2, 1]})
    panel_raw(axes[0], d)
    panel_retention(axes[1], d)

    keys = [Patch(color=WMDP_C, label=f"WMDP-{DOMAIN.capitalize()} accuracy"),
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
        stem = ("rmu_families_two_panel" if DOMAIN == "bio"
                else "wmdp_cyber_knowledge_lens_side_by_side")
        p = ROOT / "plots" / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight",
                    **({"dpi": 300} if ext == "png" else {}))
        print("wrote", p)
    if IMGS.is_dir():
        fig.savefig(IMGS / f"{stem}.pdf", bbox_inches="tight")
        print("wrote", IMGS / f"{stem}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
