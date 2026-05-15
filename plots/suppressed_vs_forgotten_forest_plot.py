#!/usr/bin/env python3
"""
Task D: Forest plot of suppressed-minus-forgotten K_int trajectory difference
per method, with bootstrap 95% CIs and BH-corrected significance.

Outputs (in plots/ckpt_layer_trajectory_metrics/):
  suppressed_vs_forgotten_forest_plot.pdf
  suppressed_vs_forgotten_forest_plot.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PUB_TABLE = REPO / "plots" / "ckpt_layer_trajectory_metrics" / "suppressed_vs_forgotten_publication_table.csv"
OUT_DIR = REPO / "plots" / "ckpt_layer_trajectory_metrics"

SIG_COLOR = "#d62728"   # red — significant after FDR
NS_COLOR  = "#aaaaaa"   # grey — not significant
MARKER_SIZE = 9
CI_LW = 2.2


def sig_label(q):
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def main():
    df = pd.read_csv(PUB_TABLE)

    df = df.sort_values("diff", ascending=True).reset_index(drop=True)

    n = len(df)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for i, row in df.iterrows():
        sig = row["q_value"] < 0.05
        color = SIG_COLOR if sig else NS_COLOR
        label = sig_label(row["q_value"])

        ax.plot(
            [row["ci95_lo"], row["ci95_hi"]], [y[i], y[i]],
            color=color, lw=CI_LW, solid_capstyle="round", zorder=2,
        )
        ax.scatter(
            row["diff"], y[i],
            color=color, s=MARKER_SIZE ** 2, zorder=3, marker="D",
        )
        ax.text(
            row["ci95_hi"] + 0.004, y[i], label,
            va="center", ha="left", fontsize=10,
            color=color, fontweight="bold" if sig else "normal",
        )
        ax.text(
            -0.005, y[i], f"d = {row['cohens_d']:.2f}",
            va="center", ha="right", fontsize=8, color="#555555",
        )

    ax.axvline(0, color="black", lw=0.9, ls="--", zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["method"], fontsize=11)
    ax.set_xlabel("Suppressed − Forgotten  (K_int trajectory, full-layer probe)", fontsize=11)
    ax.set_title(
        "Hidden knowledge: suppressed questions retain\nmore internal decodability than forgotten questions",
        fontsize=11, pad=10,
    )

    x_max = df["ci95_hi"].max()
    ax.set_xlim(-0.09, x_max + 0.07)

    sig_patch = mpatches.Patch(color=SIG_COLOR, label="q < 0.05 (BH-corrected)")
    ns_patch  = mpatches.Patch(color=NS_COLOR,  label="not significant")
    ax.legend(handles=[sig_patch, ns_patch], fontsize=9, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"suppressed_vs_forgotten_forest_plot.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
