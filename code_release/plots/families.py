#!/usr/bin/env python3
"""Sections 4.5 and 4.6: WMDP's own RMU checkpoints on three architectures.

For Zephyr-7B, Mixtral-8x7B and Yi-34B, each RMU model against its own base:

  --domain bio    Figure 5 (Section 4.5)
  --domain cyber  Figure 6 (Section 4.6)

Left panel: raw WMDP accuracy (right axis, chance 0.25) and pairwise K_ext and
K_int (left axis, chance 0.5), with a dashed marker at each base model's value.
Right panel: each metric as the % of its own base model's above-chance signal
retained, floored at 0.

Also prints the numbers quoted in those sections: raw scores, retention, the
gap K_int - K_ext, and the share of questions in each state for both the base
and the RMU model.

Usage: python plots/families.py --domain bio
       python plots/families.py --domain cyber
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (EXT_C, INT_C, K_CHANCE, WMDP_C, WMDP_CHANCE, WMDP_TEXT_C,
                    figsize, k_summary, published_accuracy, retention, save,
                    state_shares, use_style)

use_style()

FAMILIES = [("Zephyr-7B", "zephyr"), ("Mixtral-8x7B", "mixtral"), ("Yi-34B", "yi")]
NL = "\n"


def build(domain: str) -> pd.DataFrame:
    acc = published_accuracy("wmdp", domain)
    rows = []
    for name, fam in FAMILIES:
        base_id, rmu_id = f"{fam}_base", f"{fam}_rmu"
        b, r = k_summary(base_id, domain), k_summary(rmu_id, domain)
        rows.append(dict(
            label=name, base_id=base_id, rmu_id=rmu_id,
            raw_wmdp=acc[rmu_id], base_wmdp=acc[base_id],
            raw_ke=r["k_ext"], raw_ki=r["k_int"],
            ke_sd=r["k_ext_sd"], ki_sd=r["k_int_sd"],
            base_ke=b["k_ext"], base_ki=b["k_int"],
            wmdp=max(0.0, retention(acc[rmu_id], WMDP_CHANCE, acc[base_id])),
            k_ext=max(0.0, retention(r["k_ext"], K_CHANCE, b["k_ext"])),
            k_int=max(0.0, retention(r["k_int"], K_CHANCE, b["k_int"]))))
    return pd.DataFrame(rows)


def print_numbers(d: pd.DataFrame, domain: str) -> None:
    print(d[["label", "raw_wmdp", "raw_ke", "raw_ki", "wmdp", "k_ext", "k_int"]]
          .round(3).to_string(index=False))
    gap = 100 * (d.raw_ki - d.raw_ke)
    print("\ngap K_int - K_ext: " + ", ".join(
        f"{l} {g:.1f} pp" for l, g in zip(d.label, gap)))
    print("\n% of questions in each state (base -> RMU)")
    for r in d.itertuples():
        b, u = state_shares(r.base_id, domain), state_shares(r.rmu_id, domain)
        print(f"  {r.label:<13}" + "  ".join(
            f"{s} {b[s]:.1f}->{u[s]:.1f}" for s in b))


def panel_raw(ax, d, domain):
    n = len(d)
    ax2 = ax.twinx()
    x = list(range(n))
    W = 0.24
    xw, xe, xi_ = [i - W for i in x], list(x), [i + W for i in x]

    ax2.bar(xw, d.raw_wmdp, W, color=WMDP_C, zorder=3)
    ax.bar(xe, d.raw_ke, W, color=EXT_C, zorder=3)
    ax.bar(xi_, d.raw_ki, W, color=INT_C, zorder=3)
    for xx, v, sd in ((xe, d.raw_ke, d.ke_sd), (xi_, d.raw_ki, d.ki_sd)):
        ax.errorbar(xx, v, yerr=sd, fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=1.4, alpha=0.65, zorder=5)

    # dashed marker at each base model's own value
    for xx, col, vals, tgt in ((xw, WMDP_TEXT_C, d.base_wmdp, ax2),
                               (xe, EXT_C, d.base_ke, ax),
                               (xi_, INT_C, d.base_ki, ax)):
        for xi2, vv in zip(xx, vals):
            tgt.plot([xi2 - W / 2, xi2 + W / 2], [vv, vv], color=col, lw=1.1,
                     ls=(0, (2.0, 1.3)), alpha=0.9, zorder=7)

    # on cyber the two K labels sit on opposite sides of their bars
    off = 0.0 if domain == "bio" else 0.62 * W
    for xx, v, sd, c, dx in ((xe, d.raw_ke, d.ke_sd, EXT_C, -off),
                             (xi_, d.raw_ki, d.ki_sd, INT_C, +off)):
        for xi2, vv, ss in zip(xx, v, sd):
            ax.text(xi2 + dx, vv + ss + 0.014, f"{vv:.2f}", ha="center",
                    va="bottom", fontsize=6.2, color=c, rotation=90, zorder=6)
    for xi2, vv in zip(xw, d.raw_wmdp):
        ax2.text(xi2, vv + 0.012, f"{vv:.2f}", ha="center", va="bottom",
                 fontsize=6.2, color=WMDP_TEXT_C, rotation=90, zorder=6)

    ax.axhline(0.5, color=EXT_C, ls=":", lw=0.9, alpha=0.85, zorder=1)
    ax2.axhline(0.25, color=WMDP_TEXT_C, ls=":", lw=0.9, alpha=0.85, zorder=1)
    ax.text(-1.24, 0.508, "chance 0.50", fontsize=5.8, color=EXT_C,
            va="bottom", ha="left")
    ax2.text(-1.24, 0.258, "chance 0.25", fontsize=5.8, color=WMDP_TEXT_C,
             va="bottom", ha="left")

    ax.set_ylim(0.15, 0.95)
    ax2.set_ylim(0.15, 0.95)
    ax.set_ylabel("Knowledge: pairwise $K$", fontsize=7.5)
    ax2.set_ylabel(f"Accuracy: WMDP-{domain.capitalize()} (4-way)", fontsize=7.5)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticks(x)
    ax.set_xticklabels([l + NL + "RMU" for l in d.label], fontsize=6.4)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    ax.set_xlim(-1.28, n - 0.35)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(a) raw scores, each against its own floor", fontsize=7.8, pad=5)


def panel_retention(ax, d):
    n = len(d)
    x = list(range(n))
    W = 0.45
    vw, ve, vi = d.wmdp.tolist(), d.k_ext.tolist(), d.k_int.tolist()

    # every level runs from zero; the shortest is drawn in front
    for xi, w, e, i2 in zip(x, vw, ve, vi):
        trio = sorted(((w, WMDP_C), (e, EXT_C), (i2, INT_C)), key=lambda t: -t[0])
        for z, (v, colour) in enumerate(trio, start=2):
            ax.bar(xi, v, W, color=colour, zorder=z)

    for xi, a_, b_, c_ in zip(x, vw, ve, vi):
        ax.text(xi, c_ + 2.0, f"{c_:.0f}", ha="center", va="bottom",
                fontsize=6.2, color=INT_C, zorder=6)
        ya, yb = a_, b_
        if abs(b_ - a_) < 8:            # push close labels apart, keeping order
            mid = (a_ + b_) / 2
            ya, yb = (mid + 4.0, mid - 4.0) if a_ >= b_ else (mid - 4.0, mid + 4.0)
        if a_ < 0.5:
            ya = 3.0
        if b_ < 0.5:
            yb = 3.0
        for y, ylab, colr in ((a_, ya, WMDP_C), (b_, yb, EXT_C)):
            ax.plot([xi - W / 2 - 0.06, xi - W / 2], [y, y], color=colr,
                    lw=0.9, zorder=6, clip_on=False)
            ax.text(xi - W / 2 - 0.09, ylab, f"{y:.0f}", ha="right",
                    va="center", fontsize=6.2, color=colr, zorder=6)

    ax.axhline(0, color="0.35", ls=":", lw=0.8, zorder=1)
    ax.axhline(100, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.text(n - 0.45, 99, "own base model", fontsize=6.3, color="0.35",
            va="top", ha="right")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of its own base model's\nabove-chance signal retained",
                  fontsize=7.5)
    ax.set_xticks(x)
    short = [l.split("-")[0] for l in d.label]
    ax.set_xticklabels([l + NL + "RMU" for l in short], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.80, n - 0.35)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(b) normalised by each base's own range", fontsize=7.8, pad=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["bio", "cyber"], default="bio")
    domain = ap.parse_args().domain

    d = build(domain)
    print_numbers(d, domain)

    fig, axes = plt.subplots(1, 2, figsize=figsize(aspect=0.362),
                             gridspec_kw={"width_ratios": [2, 1]})
    panel_raw(axes[0], d, domain)
    panel_retention(axes[1], d)
    keys = [Patch(color=WMDP_C, label=f"WMDP-{domain.capitalize()} accuracy"),
            Patch(color=EXT_C, label=r"$K_\mathrm{ext}$ (logit margin)"),
            Patch(color=INT_C, label=r"$K_\mathrm{int}$ (best-layer probe)"),
            Line2D([], [], color="0.35", lw=1.1, ls=(0, (2.0, 1.3)),
                   label="its own base model (panel a)")]
    fig.legend(handles=keys, fontsize=6.8, loc="lower center",
               bbox_to_anchor=(0.5, -0.035), ncol=4, frameon=False,
               handletextpad=0.4, columnspacing=1.5)
    fig.subplots_adjust(wspace=0.46)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, "rmu_families_two_panel" if domain == "bio"
         else "wmdp_cyber_knowledge_lens_side_by_side")
    plt.close(fig)


if __name__ == "__main__":
    main()
