#!/usr/bin/env python3
"""The lambda sweep, all the way out to 1e6.

Two arms, because they answer different questions:

  LEFT, retention enforced (the reported setting). Rises to a peak at
  lambda = 64-512 and then FALLS BACK TO ZERO: past some lambda the probe term
  dominates enough to break more than 5% of the questions the model already got
  right, no layer qualifies, and the selector returns the unedited readout. At
  lambda = 1e6 that happens in 5/5 folds for every method. The flat tail is the
  constraint refusing, not the attack failing.

  RIGHT, constraint dropped. Monotone into a plateau at lambda >= 1e4, where the
  score is the probe alone and the value is K_int at the selected layer. That
  plateau is a restatement of K_int, not an independent measurement.

The vertical line at 64 is where the first grid stopped. Everything to its right
was invisible in the 2026-09-09 morning run, which is why those gains were floors.

Usage: python plots/bestlayer_lambda_plot.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import pandas as pd                 # noqa: E402

AV = Path(__file__).resolve().parent / "activation_vectors"
ORDER = ["base", "GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR",
         "PB_J"]

cur = pd.read_csv(AV / "bestlayer_lambda_curve.csv")
skip = pd.read_csv(AV / "bestlayer_skip.csv")
skip = skip[skip.labels == "real"].set_index("method")
grid = sorted(cur.lam.unique())
xs = {v: i for i, v in enumerate(grid)}
ticks = [("0" if v == 0 else f"{v:g}".replace("e+0", "e")) for v in grid]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
for ax, col, ttl in zip(
        axes, ["k", "k_free"],
        ["A. retention enforced (reported setting)",
         "B. retention constraint dropped"]):
    for m in ORDER:
        d = cur[cur.method == m].sort_values("lam")
        if d.empty:
            continue
        ke = float(skip.loc[m, "k_ext"])
        style = (dict(lw=2.4, color="k", ls="--", zorder=5) if m == "base"
                 else dict(lw=1.4))
        ax.plot([xs[v] for v in d.lam], d[col] - ke, marker="o", ms=3,
                label=m, **style)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.axvline(xs[64.0], color="#c0392b", lw=1.1, ls="--")
    ax.set_xticks(range(len(grid)))
    ax.set_xticklabels(ticks, rotation=60, fontsize=7)
    ax.set_xlabel("lambda")
    ax.set_title(ttl, fontsize=10)
    ax.grid(color="0.93")
    ax.set_axisbelow(True)
axes[0].set_ylabel("test K minus K_ext")
axes[0].annotate("first grid\nstopped here", (xs[64.0], 0.115),
                 textcoords="offset points", xytext=(-52, 0), fontsize=7.5,
                 color="#c0392b", ha="center")
axes[0].annotate("constraint blocks\nevery layer", (xs[1e6], 0.004),
                 textcoords="offset points", xytext=(-6, 26), fontsize=7.5,
                 color="0.35", ha="right",
                 arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9))
axes[1].annotate("probe alone:\nthis plateau IS K_int",
                 xy=(xs[16384.0], 0.1085), xytext=(0.44, 0.88),
                 textcoords="axes fraction", fontsize=7.5, color="0.35",
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9))
axes[1].legend(fontsize=7.5, ncol=2, loc="lower right")
fig.suptitle("How much of the readout to replace: the whole lambda sweep",
             fontsize=11)
fig.tight_layout()
fig.savefig(AV / "bestlayer_lambda_curve.png", dpi=180)
print(f"wrote {AV/'bestlayer_lambda_curve.png'}")
