#!/usr/bin/env python3
"""Mean K_int and K_ext at every checkpoint, averaged over all eight methods.

The companion to v6_subset_flow_mean.py (Figure 3): same axis, same checkpoints,
same mean-plus-min-max-band idiom -- but tracking the two knowledge axes rather
than the four subset populations.

This is the figure the paper has been missing. The abstract and the intro both
say the hidden-knowledge gap "widens on average as unlearning proceeds", and
5.1 is entirely ck8 while 5.2 counts subsets, so nothing in the main text has
ever plotted the gap over time (KNOWN_ISSUES #21). The shaded wedge between the
two curves is that gap.

Both axes come from the same query that reproduces tab:subsets: split_type=cv,
domain=bio, clf=LR, probe_type=own, layer_config=best_layer, over all 1,273
questions, with ck0 = the base model.

Usage: python plots/v6_k_trajectory_mean.py

Outputs: plots/v6_k_trajectory_mean.pdf/.png/.csv
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hk_utils import iclr_figsize, use_iclr_style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "inside_out_out"
METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
N_CK = 8
CHANCE = 0.5

STYLE = {
    "$K_\\text{int}$": ("#1f77b4", "-",  "o"),
    "$K_\\text{ext}$": ("#d62728", "--", "s"),
}


def load(model_id):
    df = pd.read_parquet(OUT / model_id / "k_scores.parquet")
    cv = df[(df.split_type == "cv") & (df.domain == "bio") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    q = cv.groupby("question_idx").agg(ki=("k_internal", "mean"),
                                       ke=("k_external", "mean"))
    return q.ki.mean(), q.ke.mean()


def main():
    use_iclr_style()
    xs = list(range(N_CK + 1))

    # ck0 is one model, so its "spread" is a point, not a range
    b_int, b_ext = load("base")
    ints = [[b_int] * len(METHODS)]
    exts = [[b_ext] * len(METHODS)]
    for ck in range(1, N_CK + 1):
        pair = [load(f"{m}_ck{ck}") for m in METHODS]
        ints.append([p[0] for p in pair])
        exts.append([p[1] for p in pair])
    A = {"$K_\\text{int}$": np.array(ints), "$K_\\text{ext}$": np.array(exts)}

    rows = []
    for name, a in A.items():
        for i, c in enumerate(xs):
            rows.append(dict(checkpoint="base" if c == 0 else f"ck{c}", axis=name,
                             mean=a[i].mean(), min=a[i].min(), max=a[i].max()))
    stem = str(Path(__file__).with_suffix(""))
    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)

    WIDTH_FRAC = float(os.environ.get("FIG_WIDTH_FRAC", "0.80"))
    fig, ax = plt.subplots(figsize=iclr_figsize(aspect=0.60,
                                                width_frac=WIDTH_FRAC))

    mi, me = A["$K_\\text{int}$"].mean(1), A["$K_\\text{ext}$"].mean(1)
    # the wedge IS the hidden-knowledge gap of eq:hk -- draw it under the lines
    ax.fill_between(xs, me, mi, color="#888888", alpha=0.30, linewidth=0,
                    zorder=2.5, label="hidden-knowledge gap")

    for name, (colour, ls, marker) in STYLE.items():
        a = A[name]
        ax.fill_between(xs, a.min(1), a.max(1), color=colour, alpha=0.16,
                        linewidth=0, zorder=2)
        ax.plot(xs, a.mean(1), ls, color=colour, marker=marker, markersize=3.6,
                linewidth=1.7, label=name, zorder=3)

    ax.axhline(CHANCE, color="black", linestyle=":", linewidth=0.8, zorder=2)
    ax.annotate("chance", xy=(N_CK, CHANCE), xytext=(-2, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=7)

    ax.set_xticks(xs)
    ax.set_xticklabels(["base\n(ck0)"] + [f"ck{c}" for c in range(1, N_CK + 1)])
    ax.set_xlabel("unlearning checkpoint")
    ax.set_ylabel("mean $K$ over 1,273 questions")
    ax.set_ylim(0.45, 0.88)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Both knowledge axes across unlearning, mean of 8 methods",
                 fontsize=9)

    h, l = ax.get_legend_handles_labels()
    leg = ax.legend(h, l, frameon=False, ncol=3, loc="upper center",
                    bbox_to_anchor=(0.5, -0.16), fontsize=8, columnspacing=1.2,
                    handlelength=1.8)
    # drop the legend below the xlabel -- a fixed anchor cannot know where the
    # label lands, since the tick labels are two lines tall (same fix as Fig. 3)
    fig.canvas.draw()
    y = ax.transAxes.inverted().transform(
        (0, ax.xaxis.get_label().get_window_extent().y0))[1]
    leg.set_bbox_to_anchor((0.5, y - 0.05), transform=ax.transAxes)

    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {Path(stem).name}.pdf  (all questions, n=1273, 8 methods)\n")

    print(f"{'ck':>5}{'K_int':>8}{'K_ext':>8}{'gap(pp)':>9}   "
          f"{'K_int range':>13}{'K_ext range':>14}")
    for i, c in enumerate(xs):
        tag = "base" if c == 0 else f"ck{c}"
        ki, ke = A["$K_\\text{int}$"][i], A["$K_\\text{ext}$"][i]
        print(f"{tag:>5}{ki.mean():>8.3f}{ke.mean():>8.3f}"
              f"{100*(ki.mean()-ke.mean()):>9.2f}   "
              f"{ki.min():.3f}-{ki.max():.3f}  {ke.min():.3f}-{ke.max():.3f}")
    g = 100 * (mi - me)
    print(f"\ngap: {g[0]:.2f} pp at base -> {g[-1]:.2f} pp at ck8 "
          f"({g[-1]/g[0]:.1f}x); monotone: {bool(np.all(np.diff(g) >= 0))}")


if __name__ == "__main__":
    main()
