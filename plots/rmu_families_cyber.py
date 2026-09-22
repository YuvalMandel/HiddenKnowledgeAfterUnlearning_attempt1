#!/usr/bin/env python3
"""WMDP-Cyber counterparts to Figures 3 and 4, for WMDP's own RMU checkpoints.

Same pipeline, same three families, same probe configuration as the bio
figures; only the domain changes.

WMDP-Cyber accuracies are WMDP's own, from Table 1 of arXiv:2403.03218 -- the
same table our bio numbers come from, and its bio column matches the values
already recorded here (63.7/31.2, 74.8/34.0, 75.3/30.7), which is what makes
the cyber column of the same table trustworthy.

REBUILT 2026-09-23 from the re-extracted runs. The previous version carried
two caveats that were both artefacts of the 512-token prompt truncation
(KNOWN_ISSUES.md, 2026-09-22): that Yi's cyber base was sub-chance and had
to be dropped from the retention figure, and that Mixtral inverted. Neither
survives re-extraction. All three bases are well above chance on both axes,
all three are in both figures, and all three replicate the bio pattern.

What does happen is that Yi's RMU K_ext lands at 0.469, BELOW the 0.5
pairwise floor, so its retention is negative and the axis reaches -31.

Writes:
  plots/rmu_cyber_raw.{pdf,png}        Fig 3 layout, two axes, two chance floors
  plots/rmu_cyber_retention.{pdf,png}  Fig 4 layout, stacked retention

Usage: python plots/rmu_families_cyber.py
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

use_iclr_style()

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "inside_out_out"
NL = chr(10)

# (label, base_id, rmu_id, base WMDP-Cyber acc, RMU WMDP-Cyber acc)
# accuracies: WMDP paper Table 1, arXiv:2403.03218
FAMILIES = [
    ("Zephyr-7B",    "zephyr_base",  "zephyr_rmu",  0.440, 0.282),
    ("Mixtral-8x7B", "mixtral_base", "mixtral_rmu", 0.520, 0.308),
    ("Yi-34B",       "yi_base",      "yi_rmu",      0.497, 0.290),
]

INT_C, EXT_C, WMDP_C = "#2166ac", "#d6604d", "#8c8c8c"
WMDP_C4 = "#4d4d4d"
K_CHANCE, A_CHANCE = 0.5, 0.25


def k_of(mid):
    df = pd.read_parquet(OUT_DIR / mid / "k_scores_cyber.parquet")
    cv = df[(df.split_type == "cv") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    f = cv.groupby("fold").agg(ki=("k_internal", "mean"),
                               ke=("k_external", "mean"))
    return dict(k_int=f.ki.mean(), k_ext=f.ke.mean(),
                ki_sd=f.ki.std(ddof=1), ke_sd=f.ke.std(ddof=1))


def build():
    rows = []
    for label, bid, uid, bacc, uacc in FAMILIES:
        b, u = k_of(bid), k_of(uid)
        rows.append(dict(label=label, wmdp=uacc, base_wmdp=bacc,
                         base_k_ext=b["k_ext"], base_k_int=b["k_int"], **u))
    return pd.DataFrame(rows)


def R(v, chance, base):
    return 100.0 * (v - chance) / (base - chance)


# ---- Figure 3 layout ------------------------------------------------------
def fig_raw(d):
    n = len(d)
    x = list(range(n))
    W = 0.26
    xw = [i - W for i in x]
    xe = list(x)
    xi_ = [i + W for i in x]

    fig, ax = plt.subplots(figsize=iclr_figsize(aspect=0.62, width_frac=0.80))
    ax2 = ax.twinx()

    ax2.bar(xw, d.wmdp, W, color=WMDP_C, zorder=3)
    ax.bar(xe, d.k_ext, W, color=EXT_C, zorder=3)
    ax.bar(xi_, d.k_int, W, color=INT_C, zorder=3)
    ax.errorbar(xe, d.k_ext, yerr=d.ke_sd, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=1.5, alpha=0.6, zorder=5)
    ax.errorbar(xi_, d.k_int, yerr=d.ki_sd, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=1.5, alpha=0.6, zorder=5)

    # dashed marker at each family's OWN base, one per metric
    for xx, col, vals, tgt in ((xw, "#5a5a5a", d.base_wmdp, ax2),
                               (xe, EXT_C, d.base_k_ext, ax),
                               (xi_, INT_C, d.base_k_int, ax)):
        for xi2, vv in zip(xx, vals):
            tgt.plot([xi2 - W / 2, xi2 + W / 2], [vv, vv], color=col, lw=1.2,
                     ls=(0, (2.2, 1.4)), alpha=0.9, zorder=7)

    for xx, v, sd, c in ((xe, d.k_ext, d.ke_sd, EXT_C),
                         (xi_, d.k_int, d.ki_sd, INT_C)):
        for xi2, vv, ss in zip(xx, v, sd):
            ax.text(xi2, vv + ss + 0.010, f"{vv:.3f}", ha="center", va="bottom",
                    fontsize=5.6, color=c, rotation=90, zorder=6)
    for xi2, vv in zip(xw, d.wmdp):
        ax2.text(xi2, vv + 0.008, f"{vv:.2f}", ha="center", va="bottom",
                 fontsize=5.6, color="#5a5a5a", rotation=90, zorder=6)

    ax.axhline(K_CHANCE, color=EXT_C, ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax2.axhline(A_CHANCE, color="#5a5a5a", ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax.text(-1.55, K_CHANCE + 0.006, "chance 0.50", fontsize=6, color=EXT_C,
            va="bottom", ha="left")
    ax2.text(-1.55, A_CHANCE + 0.005, "chance 0.25", fontsize=6, color="#5a5a5a",
             va="bottom", ha="left")

    ax.set_ylim(0.30, 0.92)
    ax2.set_ylim(0.18, 0.68)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax2.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_ylabel("Knowledge: pairwise $K$", fontsize=8)
    ax2.set_ylabel("Accuracy: WMDP-Cyber (4-way)", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([l + NL + "RMU" for l in d.label], fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax2.tick_params(labelsize=7.5)
    ax.set_xlim(-1.60, n - 0.35)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    h = [Patch(color=WMDP_C, label="WMDP-Cyber accuracy (right axis)"),
         Patch(color=EXT_C, label=r"$K_\mathrm{ext}$ (logit margin)"),
         Patch(color=INT_C, label=r"$K_\mathrm{int}$ (best-layer probe)"),
         Line2D([], [], color="0.35", lw=1.2, ls=(0, (2.2, 1.4)),
                label="its own base model")]
    ax.legend(handles=h, fontsize=6.6, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
              handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = ROOT / "plots" / f"rmu_cyber_raw.{ext}"
        fig.savefig(p, bbox_inches="tight",
                    **({"dpi": 200} if ext == "png" else {}))
        print("wrote", p)
    plt.close(fig)


# ---- Figure 4 layout ------------------------------------------------------
def fig_retention(d):
    ok = d[d.base_k_ext > K_CHANCE].reset_index(drop=True)
    dropped = [l for l in d.label if l not in set(ok.label)]
    if dropped:
        print("excluded from retention (base at or below chance):", dropped)

    vw = [R(a, A_CHANCE, b) for a, b in zip(ok.wmdp, ok.base_wmdp)]
    ve = [R(v, K_CHANCE, b) for v, b in zip(ok.k_ext, ok.base_k_ext)]
    vi = [R(v, K_CHANCE, b) for v, b in zip(ok.k_int, ok.base_k_int)]
    x = list(range(len(ok)))
    W = 0.42

    fig, ax = plt.subplots(figsize=iclr_figsize(aspect=5.0 / 7, width_frac=0.80))
    # grey in front of red, as in the bio figure: where the red band is short
    # it would otherwise paint over the grey
    ax.bar(x, [c - b for b, c in zip(ve, vi)], W, bottom=ve, color=INT_C,
           zorder=2)
    ax.bar(x, vw, W, color=WMDP_C4, zorder=5)
    ax.bar(x, [b - a for a, b in zip(vw, ve)], W, bottom=vw, color=EXT_C,
           zorder=3)

    for xi, a, b, c in zip(x, vw, ve, vi):
        for lo, hi, colr in ((a, b, EXT_C), (b, c, INT_C)):
            txt = f"{hi - lo:+.0f}"
            if abs(hi - lo) >= 9:
                ax.text(xi, (lo + hi) / 2, txt, ha="center", va="center",
                        fontsize=6.1, color="white", zorder=6)
            else:
                ax.text(xi + W / 2 + 0.03, (lo + hi) / 2, txt, ha="left",
                        va="center", fontsize=6.1, color=colr, zorder=6)
        # Mixtral's two edge values are 2 points apart, so the ticks stay on
        # the real values and only the numerals are pulled apart
        ly = [a, b]
        if abs(a - b) < 6:
            mid = (a + b) / 2
            ly = [mid + 3.2, mid - 3.2] if a > b else [mid - 3.2, mid + 3.2]
        for y, yl, colr in zip((a, b), ly, ("#5a5a5a", EXT_C)):
            ax.plot([xi - W / 2 - 0.06, xi - W / 2], [y, y], color=colr, lw=0.9,
                    zorder=6, clip_on=False)
            ax.text(xi - W / 2 - 0.09, yl, f"{y:.0f}", ha="right", va="center",
                    fontsize=6.3, color=colr, zorder=6)
        ax.text(xi, max(b, c) + 2.5, f"{c:.0f}", ha="center", va="bottom",
                fontsize=6.6, color=INT_C, zorder=6)

    ax.axhline(0, color="0.35", ls=":", lw=0.8, zorder=1)
    ax.axhline(100, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.text(-0.72, 101, "own base model", fontsize=6.3, color="0.35",
            va="bottom", ha="left")
    ax.set_ylim(-42, 116)
    ax.set_yticks([-25, 0, 25, 50, 75, 100])
    ax.set_ylabel("% of its own base model's\nabove-chance signal retained",
                  fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l + NL + "RMU" for l in ok.label], fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(-0.80, len(ok) - 0.35)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    keys = [(WMDP_C4, "WMDP-Cyber accuracy"),
            (EXT_C, r"$+\;K_\mathrm{ext}$ (logit margin)"),
            (INT_C, r"$+\;K_\mathrm{int}$ (best-layer probe)")]
    ax.legend(handles=[Patch(color=c, label=t) for c, t in keys],
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = ROOT / "plots" / f"rmu_cyber_retention.{ext}"
        fig.savefig(p, bbox_inches="tight",
                    **({"dpi": 200} if ext == "png" else {}))
        print("wrote", p)
    plt.close(fig)


def main():
    d = build()
    print(d[["label", "base_wmdp", "wmdp", "base_k_ext", "k_ext",
             "base_k_int", "k_int"]].round(3).to_string(index=False))
    print("\nretention (WMDP / K_ext / K_int):")
    for _, r in d.iterrows():
        print("  %-13s %5.1f %6.1f %6.1f"
              % (r.label, R(r.wmdp, A_CHANCE, r.base_wmdp),
                 R(r.k_ext, K_CHANCE, r.base_k_ext),
                 R(r.k_int, K_CHANCE, r.base_k_int)))
    print()
    fig_raw(d)
    fig_retention(d)


if __name__ == "__main__":
    main()
