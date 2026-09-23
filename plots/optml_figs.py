#!/usr/bin/env python3
"""Figures 3 and 4 restricted to the OPTML models we keep.

Nine checkpoints on HuggingFaceH4/zephyr-7b-beta, from the two peer-reviewed
OPTML collections:
  Smooth-Unlearned Model  (ICML 2025)   GradDiff, GradDiff+SAM, NPO,
                                        NPO+{SAM,GP,CR,RS,WA}
  SimNPO-Unlearned Models (NeurIPS 2025) SimNPO

Geometry, colours, chance lines, label sizes and legend placement are taken
verbatim from `knowledge_accuracy_bars.py` (Fig 3) and
`retention_three_metric.py` (Fig 4), so these sit beside the paper's own
figures rather than looking like a different family.

WMDP-Bio accuracy is OPTML's published number, never ours:
  Base, SimNPO                SimNPO model card, as 1 - Acc_Bio
  NPO, NPO+{SAM,RS,CR,GP,WA}  ICML 2025 Table 2, as Acc = 1 - UE
  GD, GD+SAM                  NOT PUBLISHED -- no grey bar is drawn for them,
                              and in Fig 4 their stack simply starts at zero.
Both sources agree on the base (0.648), so the NPO/SimNPO difference is a real
checkpoint difference and not a harness artifact.

NEGATIVE RETENTION. Most of these drive K_ext below the pairwise chance floor,
so their K_ext retention is negative while WMDP retention stays positive, and a
plain stack would drag the red segment across zero from a positive grey base.
Where either level is negative the two bars are drawn from ZERO instead of
stacked, and the level further from zero goes behind the other -- see
`stack_order`. Only a metric that is itself negative then occupies the negative
region, and when both are, the deeper one shows past the shallower.

No mean column here, unlike Fig 4: two of the nine have no published accuracy,
so a grey mean would be over a different model set than the red and blue means.

Writes:
  plots/optml_raw.{png,pdf}        Fig 3 style, raw K against the 0.5 floor
  plots/optml_retention.{png,pdf}  Fig 4 style, % of base above-chance signal

Usage: python plots/optml_figs.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hk_utils import iclr_figsize, use_iclr_style  # noqa: E402

use_iclr_style()

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "inside_out_out"
IMGS = ROOT / "overleaf_claims" / "imgs"
BASE = "zephyr_base"

# reading order: SimNPO, NPO, the NPO ablation, then GradDiff -- the two with no
# published accuracy end up together at the right.
MODELS = [
    ("zephyr_simnpo",       "SimNPO"),
    ("zephyr_npo",          "NPO"),
    ("zephyr_npo_sam",      "NPO+SAM"),
    ("zephyr_npo_gp",       "NPO+GP"),
    ("zephyr_npo_cr",       "NPO+CR"),
    ("zephyr_npo_rs",       "NPO+RS"),
    ("zephyr_npo_wa",       "NPO+WA"),
    ("zephyr_graddiff",     "GD"),
    ("zephyr_graddiff_sam", "GD+SAM"),
]
ACC = {"Base": 0.648, "SimNPO": 0.416, "NPO": 0.26, "NPO+SAM": 0.26,
       "NPO+GP": 0.27, "NPO+CR": 0.25, "NPO+RS": 0.26, "NPO+WA": 0.26}
ACC_CHANCE, ACC_BASE = 0.25, 0.648

INT_C, EXT_C = "#2166ac", "#d6604d"
WMDP_C3, WMDP_C4 = "#8c8c8c", "#4d4d4d"   # Fig 3 grey, Fig 4 grey
# A_TOP == K_TOP: shared scale, so a value reads at the same height on
# either axis. See knowledge_accuracy_bars.py.
K_TOP, A_TOP, Y_BOT = 0.92, 0.92, 0.15


def k_of(mid):
    df = pd.read_parquet(OUT_DIR / mid / "k_scores.parquet")
    cv = df[(df.split_type == "cv") & (df.domain == "bio") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    f = cv.groupby("fold").agg(ki=("k_internal", "mean"),
                               ke=("k_external", "mean"))
    return dict(k_int=f.ki.mean(), k_ext=f.ke.mean(),
                k_int_sd=f.ki.std(ddof=1), k_ext_sd=f.ke.std(ddof=1))


def build():
    b = k_of(BASE)
    rows = []
    for mid, lab in MODELS:
        r = k_of(mid)
        r.update(label=lab,
                 r_ext=100 * (r["k_ext"] - 0.5) / (b["k_ext"] - 0.5),
                 r_int=100 * (r["k_int"] - 0.5) / (b["k_int"] - 0.5))
        rows.append(r)
    d = pd.DataFrame(rows)
    base_row = dict(b, label="Base", r_ext=100.0, r_int=100.0)
    d_raw = pd.concat([pd.DataFrame([base_row]), d], ignore_index=True)
    return b, d, d_raw


# ---- Figure 3: raw scores, two axes, two chance floors --------------------
def fig_raw(d):
    n = len(d)
    x = list(range(n))
    fig, ax = plt.subplots(figsize=iclr_figsize(aspect=0.62, width_frac=0.80))
    ax2 = ax.twinx()

    W = 0.26
    xw = [i - W for i in x]
    xe = list(x)
    xi_ = [i + W for i in x]
    gx = [xi2 for xi2, lab in zip(xw, d.label) if lab in ACC]
    gv = [ACC[lab] for lab in d.label if lab in ACC]
    ax2.bar(gx, gv, W, color=WMDP_C3, zorder=3,
            label="WMDP-Bio accuracy (right axis)")
    ax.bar(xe, d.k_ext, W, color=EXT_C, zorder=3,
           label=r"$K_\mathrm{ext}$ (logit margin)")
    ax.bar(xi_, d.k_int, W, color=INT_C, zorder=3,
           label=r"$K_\mathrm{int}$ (best-layer probe)")
    ax.errorbar(xe, d.k_ext, yerr=d.k_ext_sd, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=1.5, alpha=0.6, zorder=5)
    ax.errorbar(xi_, d.k_int, yerr=d.k_int_sd, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=1.5, alpha=0.6, zorder=5)
    for xx, v, sd, c in ((xe, d.k_ext, d.k_ext_sd, EXT_C),
                         (xi_, d.k_int, d.k_int_sd, INT_C)):
        for xi2, vv, ss in zip(xx, v, sd):
            ax.text(xi2, vv + ss + 0.015, f"{vv:.3f}", ha="center",
                    va="bottom", fontsize=5.6, color=c, rotation=90, zorder=6)
    for xi2, vv in zip(gx, gv):
        ax2.text(xi2, vv + 0.012, f"{vv:.2f}", ha="center", va="bottom",
                 fontsize=5.6, color="#5a5a5a", rotation=90, zorder=6)
    # the base is the within-family reference
    for off, v in ((0, d.k_ext[0]), (W, d.k_int[0])):
        ax.bar([off], [v], W, facecolor="none", edgecolor="black", lw=0.9,
               zorder=4)

    ax.axhline(0.5, color=EXT_C, ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax2.axhline(ACC_CHANCE, color="#5a5a5a", ls=":", lw=1.0, alpha=0.85,
                zorder=1)
    ax.text(-1.85, 0.507, "chance 0.50", fontsize=6, color=EXT_C,
            va="bottom", ha="left")
    ax2.text(-1.85, 0.256, "chance 0.25", fontsize=6, color="#5a5a5a",
             va="bottom", ha="left")

    ax.set_ylim(Y_BOT, K_TOP)
    ax2.set_ylim(Y_BOT, A_TOP)
    ax.set_ylabel("Knowledge: pairwise $K$", fontsize=8)
    ax2.set_ylabel("Accuracy: WMDP-Bio (4-way)", fontsize=8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticks(x)
    ax.set_xticklabels(d.label, rotation=34, ha="right", fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax2.tick_params(labelsize=7.5)
    ax.set_xlim(-1.90, n - 0.35)   # left gutter holds the two chance labels
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h2 + h1, l2 + l1, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
              handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = ROOT / "plots" / f"optml_raw.{ext}"
        fig.savefig(p, bbox_inches="tight",
                    **({"dpi": 200} if ext == "png" else {}))
        print("wrote", p)
    if IMGS.is_dir():
        fig.savefig(IMGS / (p.stem + ".pdf"), bbox_inches="tight")
        print("wrote", IMGS / (p.stem + ".pdf"))
    plt.close(fig)


def stack_order(w, e):
    """Draw order for one retention column: (z_grey, z_red).

    Both bars run from ZERO -- nothing is stacked on anything. The level CLOSER
    TO ZERO is drawn in front, so the other one shows exactly where it reaches
    past it, and a metric that is negative occupies the negative region alone.

    Whenever 0 <= w <= e this reproduces the stack of the original Fig 4: the
    grey covers the red over 0..w, leaving the red visible over w..e, which is
    the gap. Everywhere else it is the rule that stacking cannot express.
    """
    front_is_grey = abs(w) <= abs(e)
    return (5, 3) if front_is_grey else (3, 5)


def _selfcheck():
    assert stack_order(3, 32) == (5, 3)     # grey smaller -> grey in front
    assert stack_order(42, 32) == (3, 5)    # red smaller  -> red in front
    assert stack_order(3, -8) == (5, 3)     # grey up, red down
    assert stack_order(-2, -3) == (5, 3)    # red deeper   -> red behind
    assert stack_order(-5, -4) == (3, 5)    # grey deeper  -> grey behind


# ---- Figure 4: retention, stacked on a common chance-to-base scale --------
def fig_retention(d):
    cols = list(d.label)
    vw = [100 * (ACC[l] - ACC_CHANCE) / (ACC_BASE - ACC_CHANCE)
          if l in ACC else 0.0 for l in cols]
    ve, vi = d.r_ext.tolist(), d.r_int.tolist()
    x = list(range(len(cols)))
    W = 0.46

    fig, ax = plt.subplots(figsize=iclr_figsize(aspect=5.0 / 7,
                                                width_frac=0.80))
    # Retention is floored at 0, as in the main-text figures: below chance
    # is no evidence, not negative evidence. Every level then runs from zero
    # with the SHORTEST drawn in front, so none is hidden.
    vw = [max(0.0, v) for v in vw]
    ve = [max(0.0, v) for v in ve]
    vi = [max(0.0, v) for v in vi]
    for xi, w, e, i2 in zip(x, vw, ve, vi):
        trio = sorted(((w, WMDP_C4), (e, EXT_C), (i2, INT_C)),
                      key=lambda t: -t[0])
        for z, (v, colour) in enumerate(trio, start=2):
            ax.bar(xi, v, W, color=colour, zorder=z)

    def level(xi, y, y_label, txt, colr):
        """Tick at the true height, number to its left."""
        ax.plot([xi - W / 2 - 0.06, xi - W / 2], [y, y], color=colr, lw=0.9,
                zorder=6, clip_on=False)
        ax.text(xi - W / 2 - 0.09, y_label, txt, ha="right", va="center",
                fontsize=6.3, color=colr, zorder=6)

    for xi, a_, b_, c_, lab in zip(x, vw, ve, vi, cols):
        ya, yb = a_, b_
        if abs(b_ - a_) < 7:
            mid = (a_ + b_) / 2
            ya, yb = ((mid + 3.6, mid - 3.6) if a_ >= b_
                      else (mid - 3.6, mid + 3.6))
        if a_ < 0.5:
            ya = 3.0
        if b_ < 0.5:
            yb = 3.0
        if lab in ACC:
            level(xi, a_, ya, f'{a_:.0f}', '#5a5a5a')
        level(xi, b_, yb, f'{b_:.0f}', EXT_C)
        level(xi, c_, c_, f'{c_:.0f}', INT_C)
    ax.axhline(0, color="0.35", ls=":", lw=0.8, zorder=1)
    ax.axhline(100, color="0.35", ls="--", lw=0.8, zorder=1)
    # No "base model" caption on this panel: its totals run 86-96 and would
    # collide with it at the top left. The y label already says the bars are a
    # percentage of the base, and the axis now tops out exactly at the line.
    # nothing can exceed its own base, so 100 is the ceiling
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of base model's above-chance\nsignal retained",
                  fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=34, ha="right", fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(-1.0, x[-1] + 0.75)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    # the bars are drawn one at a time, so the legend needs its own handles
    keys = [(WMDP_C4, "WMDP-Bio accuracy"),
            (EXT_C, r"$K_\mathrm{ext}$ (logit margin)"),
            (INT_C, r"$K_\mathrm{int}$ (best-layer probe)")]
    ax.legend(handles=[Patch(color=c, label=t) for c, t in keys],
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = ROOT / "plots" / f"optml_retention.{ext}"
        fig.savefig(p, bbox_inches="tight",
                    **({"dpi": 200} if ext == "png" else {}))
        print("wrote", p)
    if IMGS.is_dir():
        fig.savefig(IMGS / (p.stem + ".pdf"), bbox_inches="tight")
        print("wrote", IMGS / (p.stem + ".pdf"))
    plt.close(fig)


def main():
    _selfcheck()
    b, d, d_raw = build()
    print("base: K_ext %.3f  K_int %.3f\n" % (b["k_ext"], b["k_int"]))
    print(d[["label", "k_ext", "k_int", "r_ext", "r_int"]]
          .round(1).to_string(index=False))
    fig_raw(d_raw)
    fig_retention(d)


if __name__ == "__main__":
    main()
