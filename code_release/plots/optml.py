#!/usr/bin/env python3
"""Appendix E: the nine OPTML unlearning models on Zephyr-7B-beta.

Draws Figure 8 (raw scores, as Figure 2) and Figure 9 (retained share of the
base model's above-chance signal, as Figure 3), and prints the numbers quoted
in Appendix E: K_int and K_ext, their retention, the gap, and the suppressed
and forgotten shares against the base model's own.

WMDP-Bio accuracies are OPTML's published values; GradDiff and GradDiff+SAM
have none, so they get no WMDP bar.

Usage: python plots/optml.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from common import (EXT_C, INT_C, K_CHANCE, WMDP_C, WMDP_CHANCE, WMDP_TEXT_C,
                    figsize, k_summary, published_accuracy, retention, save,
                    state_shares, use_style)

use_style()

BASE = "zephyr_base"
MODELS = [("zephyr_simnpo", "SimNPO"), ("zephyr_npo", "NPO"),
          ("zephyr_npo_sam", "NPO+SAM"), ("zephyr_npo_gp", "NPO+GP"),
          ("zephyr_npo_cr", "NPO+CR"), ("zephyr_npo_rs", "NPO+RS"),
          ("zephyr_npo_wa", "NPO+WA"), ("zephyr_graddiff", "GD"),
          ("zephyr_graddiff_sam", "GD+SAM")]
K_TOP = A_TOP = 0.96
Y_BOT = 0.15


def build():
    acc = published_accuracy("optml")
    b = k_summary(BASE)
    rows = [dict(label="Base", model_id=BASE, wmdp=acc.get(BASE), **b)]
    for mid, lab in MODELS:
        rows.append(dict(label=lab, model_id=mid, wmdp=acc.get(mid), **k_summary(mid)))
    d = pd.DataFrame(rows)
    d["r_wmdp"] = [retention(a, WMDP_CHANCE, acc[BASE]) if pd.notna(a) else float("nan")
                   for a in d.wmdp]
    d["r_ext"] = retention(d.k_ext, K_CHANCE, b["k_ext"])
    d["r_int"] = retention(d.k_int, K_CHANCE, b["k_int"])
    return d


def print_numbers(d):
    print(d[["label", "wmdp", "k_ext", "k_int", "r_wmdp", "r_ext", "r_int"]]
          .round(3).to_string(index=False))
    m = d.iloc[1:]
    shares = {mid: state_shares(mid) for mid in d.model_id}
    sup = [shares[mid]["suppressed"] for mid in m.model_id]
    forg = [shares[mid]["forgotten"] for mid in m.model_id]
    gap = 100 * (m.k_int - m.k_ext)
    print(f"\nK_int {m.k_int.min():.3f}-{m.k_int.max():.3f} (base {d.k_int[0]:.3f}); "
          f"K_ext {m.k_ext.min():.3f}-{m.k_ext.max():.3f} (base {d.k_ext[0]:.3f})")
    print(f"K_int retains {m.r_int.min():.0f}-{m.r_int.max():.0f}%; "
          f"K_ext {m.r_ext.min():.0f}-{m.r_ext.max():.0f}%")
    print(f"gap {gap.min():.1f}-{gap.max():.1f} pp")
    print(f"suppressed {min(sup):.1f}-{max(sup):.1f}%; forgotten "
          f"{min(forg):.1f}-{max(forg):.1f}% (base {shares[BASE]['forgotten']:.1f}%)")


def figure_8(d):
    """Raw scores: two axes, two chance floors (as Figure 2)."""
    n = len(d)
    x = list(range(n))
    fig, ax = plt.subplots(figsize=figsize(aspect=0.321))
    ax2 = ax.twinx()
    W = 0.26
    xw, xe, xi_ = [i - W for i in x], list(x), [i + W for i in x]
    has = d.wmdp.notna().to_numpy()
    gx = [v for v, h in zip(xw, has) if h]
    gv = d.wmdp[has].tolist()
    ax2.bar(gx, gv, W, color=WMDP_C, zorder=3, label="WMDP-Bio accuracy")
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
    for xi2, vv in zip(gx, gv):
        ax2.text(xi2, vv + 0.012, f"{vv:.2f}", ha="center", va="bottom",
                 fontsize=5.6, color=WMDP_TEXT_C, rotation=90, zorder=6)
    # outline the base model's K bars: it is the reference for the other nine
    for off, v in ((0, d.k_ext[0]), (W, d.k_int[0])):
        ax.bar([off], [v], W, facecolor="none", edgecolor="black", lw=0.9, zorder=4)

    ax.axhline(0.5, color=EXT_C, ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax2.axhline(WMDP_CHANCE, color=WMDP_TEXT_C, ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax.text(-1.62, 0.507, "chance 0.50", fontsize=6, color=EXT_C,
            va="bottom", ha="left")
    ax2.text(-1.62, 0.256, "chance 0.25", fontsize=6, color=WMDP_TEXT_C,
             va="bottom", ha="left")
    ax.set_ylim(Y_BOT, K_TOP)
    ax2.set_ylim(Y_BOT, A_TOP)
    ax.set_ylabel("Knowledge: pairwise $K$", fontsize=8)
    ax2.set_ylabel("Accuracy: WMDP-Bio (4-way)", fontsize=8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("+", "\n+") for l in d.label], rotation=0,
                       ha="center", fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax2.tick_params(labelsize=7.5)
    ax.set_xlim(-1.67, n - 0.35)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h2 + h1, l2 + l1, fontsize=7, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
              handletextpad=0.4, columnspacing=1.2, borderaxespad=0.15)
    fig.tight_layout()
    save(fig, "optml_raw")
    plt.close(fig)


def figure_9(d):
    """Retained share, floored at 0 (as Figure 3). No bar for missing WMDP."""
    d = d.iloc[1:].reset_index(drop=True)
    cols = list(d.label)
    has = d.wmdp.notna().tolist()
    vw = [max(0.0, v) if h else 0.0 for v, h in zip(d.r_wmdp, has)]
    ve = [max(0.0, v) for v in d.r_ext]
    vi = [max(0.0, v) for v in d.r_int]
    x = list(range(len(cols)))
    W = 0.46

    fig, ax = plt.subplots(figsize=figsize(aspect=0.60 * 3.6 / 7))
    for xi, w, e, i2 in zip(x, vw, ve, vi):
        trio = sorted(((w, WMDP_C), (e, EXT_C), (i2, INT_C)), key=lambda t: -t[0])
        for z, (v, colour) in enumerate(trio, start=2):
            ax.bar(xi, v, W, color=colour, zorder=z)

    def level(xi, y, y_label, txt, colr):
        """Tick at the true height, number to its left."""
        ax.plot([xi - W / 2 - 0.06, xi - W / 2], [y, y], color=colr, lw=0.9,
                zorder=6, clip_on=False)
        ax.text(xi - W / 2 - 0.09, y_label, txt, ha="right", va="center",
                fontsize=6.3, color=colr, zorder=6)

    for xi, a_, b_, c_, h in zip(x, vw, ve, vi, has):
        ya, yb = a_, b_
        if abs(b_ - a_) < 7:
            mid = (a_ + b_) / 2
            ya, yb = (mid + 3.6, mid - 3.6) if a_ >= b_ else (mid - 3.6, mid + 3.6)
        if a_ < 0.5:
            ya = 3.0
        if b_ < 0.5:
            yb = -7.0          # below the x axis, clear of the WMDP label
        if h:
            level(xi, a_, ya, f"{a_:.0f}", WMDP_TEXT_C)
        level(xi, b_, yb, f"{b_:.0f}", EXT_C)
        if b_ < 0.5:
            ax.texts[-1].set_clip_on(False)
        level(xi, c_, c_, f"{c_:.0f}", INT_C)
    ax.axhline(0, color="0.35", ls=":", lw=0.8, zorder=1)
    ax.axhline(100, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of base model's above-chance\nsignal retained", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=0, ha="center", fontsize=7.5)
    ax.tick_params(axis="x", pad=9)   # room for the 0 labels under the axis
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(-1.0, x[-1] + 0.75)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    keys = [(WMDP_C, "WMDP-Bio accuracy"),
            (EXT_C, r"$K_\mathrm{ext}$ (logit margin)"),
            (INT_C, r"$K_\mathrm{int}$ (best-layer probe)")]
    ax.legend(handles=[Patch(color=c, label=t) for c, t in keys],
              fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.02),
              ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.4,
              borderaxespad=0.0)
    fig.tight_layout()
    save(fig, "optml_retention")
    plt.close(fig)


def main():
    d = build()
    print_numbers(d)
    figure_8(d)
    figure_9(d)


if __name__ == "__main__":
    main()
