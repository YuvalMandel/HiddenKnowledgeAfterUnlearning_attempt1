#!/usr/bin/env python3
"""Section 4.3: K_int and K_ext across the unlearning checkpoints.

Draws Figure 4 (RMU alone) and Figure 10 (Appendix H, one panel per method),
and prints the table behind the numbers in Section 4.3: at every checkpoint,
the mean of K_int and K_ext over the eight methods, their gap, and the range
across methods. Checkpoint 0 is the shared base model. All 1,273 questions.

Usage: python plots/k_trajectories.py
"""
import matplotlib.pyplot as plt
import numpy as np

from common import (METHODS, N_CHECKPOINTS, K_CHANCE, figsize, label, load_k,
                    save, use_style)

use_style()

XS = list(range(N_CHECKPOINTS + 1))
STYLE = {"$K_\\text{int}$": ("#1f77b4", "-", "o"),
         "$K_\\text{ext}$": ("#d62728", "--", "s")}


def mean_k(model_id: str) -> tuple[float, float]:
    d = load_k(model_id)
    return d.k_internal.mean(), d.k_external.mean()


def trajectories() -> dict:
    """{method: (K_int per checkpoint, K_ext per checkpoint)}, checkpoint 0 = base."""
    base = mean_k("base")
    out = {}
    for m in METHODS:
        pairs = [base] + [mean_k(f"{m}_ck{ck}") for ck in range(1, N_CHECKPOINTS + 1)]
        out[m] = (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
    return out


def draw(ax, ki, ke, markersize, linewidth):
    """The gap as a shaded wedge under the two lines, plus the chance line."""
    ax.fill_between(XS, ke, ki, color="#888888", alpha=0.30, linewidth=0,
                    zorder=2.5, label="gap")
    for name, (colour, ls, marker) in STYLE.items():
        ax.plot(XS, ki if "int" in name else ke, ls, color=colour, marker=marker,
                markersize=markersize, linewidth=linewidth, label=name, zorder=3)
    ax.axhline(K_CHANCE, color="black", linestyle=":", linewidth=0.8, zorder=2)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xticks(XS)
    ax.set_xticklabels([str(c) for c in XS], fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)


def figure_4(series, method="RMU"):
    """One method, sized like one cell of Figure 10 so it fits beside the text."""
    fig, ax = plt.subplots(figsize=figsize(aspect=0.458, width_frac=0.44))
    draw(ax, *series[method], markersize=2.2, linewidth=1.2)
    ax.set_ylim(0.47, 0.93)
    ax.set_ylabel("mean $K$", fontsize=7.5)
    ax.set_xlabel("unlearning checkpoint (0 = base)", fontsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.0, K_CHANCE + 0.008, "chance", fontsize=5.8, color="0.35",
            va="bottom", ha="left")
    ax.legend(frameon=False, fontsize=6.0, loc="upper right", ncol=3,
              handlelength=1.3, columnspacing=1.0, borderaxespad=0.15,
              handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    save(fig, "k_trajectory_rmu")
    plt.close(fig)


def figure_10(series):
    """All eight methods, 2 x 4 panels."""
    w, h = figsize(aspect=0.52)
    fig, axes = plt.subplots(2, 4, figsize=(w, h * 1.02), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), METHODS):
        draw(ax, *series[m], markersize=2.0, linewidth=1.1)
        ax.set_title(label(m), fontsize=8)
        ax.set_ylim(0.47, 0.85)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean $K$", fontsize=8)

    h_, l_ = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(h_, l_, frameon=False, ncol=3, loc="lower center",
                     bbox_to_anchor=(0.5, 0.0), fontsize=7.5,
                     columnspacing=1.4, handlelength=2.0)
    fig.tight_layout(rect=(0, 0.115, 1, 1), h_pad=0.7, w_pad=0.6)
    # Place the x label and the legend from the measured bottom of the last row.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y_axes = min(inv.transform((0, ax.get_tightbbox(r).y0))[1] for ax in axes[-1, :])
    lab = fig.text(0.5, y_axes - 0.018, "unlearning checkpoint (0 = base)",
                   ha="center", va="top", fontsize=8)
    fig.canvas.draw()
    y_lab = inv.transform((0, lab.get_window_extent().y0))[1]
    leg.set_bbox_to_anchor((0.5, y_lab - 0.075), transform=fig.transFigure)
    save(fig, "k_trajectory_panels")
    plt.close(fig)


def print_table(series):
    ki = np.array([series[m][0] for m in METHODS])      # (method, checkpoint)
    ke = np.array([series[m][1] for m in METHODS])
    print(f"{'checkpoint':<11}{'K_int':>7}{'K_ext':>7}{'gap pp':>8}"
          f"{'K_int range':>16}{'K_ext range':>15}")
    for c in XS:
        a, b = ki[:, c], ke[:, c]
        print(f"{'base' if c == 0 else c:<11}{a.mean():>7.3f}{b.mean():>7.3f}"
              f"{100 * (a.mean() - b.mean()):>8.1f}"
              f"{a.min():>10.3f}-{a.max():.3f}{b.min():>9.3f}-{b.max():.3f}")


def main():
    series = trajectories()
    print_table(series)
    figure_4(series)
    figure_10(series)


if __name__ == "__main__":
    main()
