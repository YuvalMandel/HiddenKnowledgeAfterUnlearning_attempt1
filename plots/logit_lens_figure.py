#!/usr/bin/env python3
"""Paper Figure 7 (Appendix C) from code_release's logit_lens.csv, layers 1-32.

The code release prints numbers only; this draws the paper's figure from the
same results. Layer 0 (the embedding output) is left out: the four claims of a
question share their last token, so their states are identical and every K is 0.
Drawn in matplotlib's default style, as the paper's figure always was.

Usage: python plots/logit_lens_figure.py <path/to/logit_lens.csv> <out.pdf>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import pandas as pd               # noqa: E402

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
MODELS = ["base"] + [f"{m}_ck8" for m in METHODS]
K_CHANCE = 0.5


def main(csv, out):
    df = pd.read_csv(csv)
    df = df[df.layer >= 1]
    titles = ["(a) the model's own axis $v$ at layer $\\ell$\n"
              "(logit lens, no fitted parameters)",
              "(b) a probe fitted at layer $\\ell$\n($K_{int}$, cross-validated)"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45), sharey=True)
    for ax, col, ttl in zip(axes, ["k_lens", "k_probe"], titles):
        for mid in MODELS:
            d = df[df.model_id == mid]
            style = (dict(lw=1.5, color="k", ls="--", zorder=5) if mid == "base"
                     else dict(lw=0.9))
            ax.plot(d.layer, d[col], marker="o", ms=1.8,
                    label="base" if mid == "base" else mid[:-4].replace("PB_J", "PB&J"),
                    **style)
        ax.axhline(K_CHANCE, color="k", lw=0.6, ls=":")
        ax.set_xlabel("layer $\\ell$", fontsize=7)
        ax.set_title(ttl, fontsize=7)
        ax.grid(color="0.93", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("$K$ (win fraction over the 3 distractors)", fontsize=7)
    axes[1].text(33, 0.506, "chance", fontsize=6, va="bottom", ha="right")
    axes[1].legend(fontsize=5.8, ncol=5, loc="lower right", frameon=False,
                   handlelength=1.0, handletextpad=0.3, labelspacing=0.25,
                   columnspacing=0.6, borderaxespad=0.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
