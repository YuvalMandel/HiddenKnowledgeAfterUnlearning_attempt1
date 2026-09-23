#!/usr/bin/env python3
"""K_int and K_ext across checkpoints, one panel per unlearning method.

The per-method version of v6_k_trajectory_mean.py, in the panel layout of
v6_subset_flow_panels.py. The mean figure shows the two axes separating; these
panels show that the separation is produced by visibly different dynamics --
some methods drop K_ext once and stop, others keep going, and at least one
partially recovers.

The shaded wedge in each panel is that method's hidden-knowledge gap (eq:hk),
so the widening can be compared across methods by area rather than by reading
two lines against each other.

Same query as everywhere else: split_type=cv, domain=bio, clf=LR,
probe_type=own, layer_config=best_layer, all 1,273 questions, ck0 = base.

Usage: python plots/v6_k_trajectory_panels.py

Outputs: plots/v6_k_trajectory_panels.pdf/.png/.csv
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
LABEL = {"PB_J": "PB&J"}
N_CK = 8
CHANCE = 0.5

# identical to v6_k_trajectory_mean.py so the two read as one figure
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


def render_single(xs, series, method="RMU", width_frac=0.32):
    """One method at the size it had as a cell of the 2x4 grid.

    The grid is 5.5in wide over four columns, so a panel is ~1.375in.
    At width_frac 0.32 the axes come out the same size once the y label
    and ticks are allowed for -- the point of the single panel is to buy
    back the space the other seven took, not to enlarge RMU."""
    ki, ke = series[method]
    w, h = iclr_figsize(aspect=0.63, width_frac=width_frac)
    fig, ax = plt.subplots(figsize=(w, h))
    ax.fill_between(xs, ke, ki, color="#888888", alpha=0.30, linewidth=0,
                    zorder=2.5, label="gap")
    for name, (colour, ls, marker) in STYLE.items():
        v = ki if "int" in name else ke
        ax.plot(xs, v, ls, color=colour, marker=marker, markersize=2.2,
                linewidth=1.2, label=name, zorder=3)
    ax.axhline(CHANCE, color="black", linestyle=":", linewidth=0.8, zorder=2)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0.47, 0.85)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xticks(xs)
    ax.set_xticklabels(["0"] + [str(c) for c in range(1, N_CK + 1)],
                       fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel("mean $K$", fontsize=7.5)
    ax.set_xlabel("unlearning checkpoint (0 = base)", fontsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # no legend: at 40% of the original height it lands on the curves,
    # and the colour key costs nothing in the caption
    fig.tight_layout(pad=0.3)

    stem = str(Path(__file__).resolve().parent / "v6_k_trajectory_rmu")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    imgs = Path(__file__).resolve().parent.parent / "overleaf_claims" / "imgs"
    if imgs.is_dir():
        fig.savefig(imgs / "v6_k_trajectory_rmu.pdf", bbox_inches="tight")
        print(f"Saved {imgs / 'v6_k_trajectory_rmu.pdf'}")
    plt.close(fig)
    print(f"Saved v6_k_trajectory_rmu.pdf  ({method} only, for the main text)")


def main():
    use_iclr_style()
    xs = list(range(N_CK + 1))

    b_int, b_ext = load("base")
    series, rows = {}, []
    for m in METHODS:
        pairs = [(b_int, b_ext)] + [load(f"{m}_ck{ck}")
                                    for ck in range(1, N_CK + 1)]
        series[m] = (np.array([p[0] for p in pairs]),
                     np.array([p[1] for p in pairs]))
        for i, (ki, ke) in enumerate(pairs):
            rows.append(dict(method=LABEL.get(m, m),
                             checkpoint="base" if i == 0 else f"ck{i}",
                             k_int=ki, k_ext=ke, gap_pp=100 * (ki - ke)))

    stem = str(Path(__file__).with_suffix(""))
    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)

    W = float(os.environ.get("FIG_WIDTH_FRAC", "1.0"))
    w, h = iclr_figsize(aspect=0.52, width_frac=W)
    fig, axes = plt.subplots(2, 4, figsize=(w, h * 1.02), sharex=True, sharey=True)

    for ax, m in zip(axes.ravel(), METHODS):
        ki, ke = series[m]
        ax.fill_between(xs, ke, ki, color="#888888", alpha=0.30, linewidth=0,
                        zorder=2.5, label="hidden-knowledge gap")
        for name, (colour, ls, marker) in STYLE.items():
            v = ki if "int" in name else ke
            ax.plot(xs, v, ls, color=colour, marker=marker, markersize=2.0,
                    linewidth=1.1, label=name, zorder=3)
        ax.axhline(CHANCE, color="black", linestyle=":", linewidth=0.8, zorder=2)
        ax.set_title(LABEL.get(m, m), fontsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0.47, 0.85)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8])
        ax.set_xticks(xs)
        ax.set_xticklabels(["0"] + [str(c) for c in range(1, N_CK + 1)],
                           fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)

    for ax in axes[:, 0]:
        ax.set_ylabel("mean $K$", fontsize=8)

    h_, l_ = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(h_, l_, frameon=False, ncol=3, loc="lower center",
                     bbox_to_anchor=(0.5, 0.0), fontsize=7.5,
                     columnspacing=1.4, handlelength=2.0)
    fig.tight_layout(rect=(0, 0.115, 1, 1), h_pad=0.7, w_pad=0.6)

    # Place the x label and legend from the MEASURED bottom of the lowest row
    # rather than from guessed figure coordinates: supxlabel(y=...) cannot know
    # where tight_layout put the axes, which is what opened the white band.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y_axes = min(inv.transform((0, ax.get_tightbbox(r).y0))[1]
                 for ax in axes[-1, :])
    lab = fig.text(0.5, y_axes - 0.018, "unlearning checkpoint (0 = base)",
                   ha="center", va="top", fontsize=8)
    fig.canvas.draw()
    y_lab = inv.transform((0, lab.get_window_extent().y0))[1]
    leg.set_bbox_to_anchor((0.5, y_lab - 0.075), transform=fig.transFigure)

    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    imgs = ROOT / "overleaf_claims" / "imgs"
    if imgs.is_dir():
        fig.savefig(imgs / "v6_k_trajectory_panels.pdf",
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {Path(stem).name}.pdf  (8 panels, all 1,273 questions)")

    render_single(xs, series)

    print(f"{'method':<10}{'gap ck0':>9}{'gap ck8':>9}{'peak':>7}{'at':>5}"
          f"{'  monotone':>11}")
    print("-" * 52)
    for m in METHODS:
        ki, ke = series[m]
        g = 100 * (ki - ke)
        print(f"{LABEL.get(m, m):<10}{g[0]:>9.2f}{g[-1]:>9.2f}{g.max():>7.1f}"
              f"{'ck' + str(int(g.argmax())):>5}"
              f"{str(bool(np.all(np.diff(g) >= 0))):>11}")


if __name__ == "__main__":
    main()
