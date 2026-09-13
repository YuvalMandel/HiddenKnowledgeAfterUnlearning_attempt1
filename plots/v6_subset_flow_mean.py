#!/usr/bin/env python3
"""Truth-table populations across unlearning, averaged over all eight methods.

Companion to v6_subset_flow.py, which shows one method. Here each of the four
cells gets one line = the mean count across the eight methods, with a
translucent band spanning the min-max range across those methods, so the
cross-method spread is visible without eight overlapping traces (the style of
kfold_line_methods_auc_lr_all).

Default runs with NO question filtering: all 1,273 questions, so the base model
(ck0) is not all-retained -- questions it already failed start in the other
cells, and the curves are absolute populations rather than movement away from a
known-good baseline. Pass --qstar for the 701-question pre-filtered pool.

Usage:
    python plots/v6_subset_flow_mean.py [--qstar]

Outputs: plots/v6_subset_flow_mean{,_qstar}.pdf/.png/.csv
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
from hk_utils import iclr_figsize, use_iclr_style

use_iclr_style()

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "inside_out_out"
SAVE_DIR = Path(__file__).resolve().parent
QSTAR_ONLY = "--qstar" in sys.argv
METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
N_CK = 8
KNOWS = 0.5
SEL = dict(split_type="cv", domain="bio", clf="LR",
           probe_type="own", layer_config="best_layer")

# same colours/styles as v6_subset_flow.py, greyscale-safe via linestyle
STYLE = {
    "Retained":   ("#56B4E9", "--", "s"),
    "Suppressed": ("#D55E00", "-",  "^"),
    "Forgotten":  ("#CC79A7", "-.", "v"),
    "Lucky":      ("#009E73", ":",  "D"),
}


def load(model_id):
    d = pd.read_parquet(OUT_DIR / model_id / "k_scores.parquet")
    for k, v in SEL.items():
        d = d[d[k] == v]
    return d.set_index("question_idx")[["k_internal", "k_external"]]


base = load("base")
qstar = base[(base.k_internal == 1.0) & (base.k_external == 1.0)].index
pool = qstar if QSTAR_ONLY else base.index
n_pool = len(pool)

xs = list(range(0, N_CK + 1))
# counts[subset][method] = list over checkpoints
counts = {name: {m: [] for m in METHODS} for name in STYLE}
for m in METHODS:
    for c in xs:
        d = base if c == 0 else load(f"{m}_ck{c}")
        q = d.loc[d.index.intersection(pool)]
        hi_i, hi_e = q.k_internal > KNOWS, q.k_external > KNOWS
        n = {
            "Retained":   int((hi_i & hi_e).sum()),
            "Suppressed": int((hi_i & ~hi_e).sum()),
            "Forgotten":  int((~hi_i & ~hi_e).sum()),
            "Lucky":      int((~hi_i & hi_e).sum()),
        }
        assert sum(n.values()) == len(q), f"{m} ck{c}: {sum(n.values())}"
        for k, v in n.items():
            counts[k][m].append(v)

stem = SAVE_DIR / ("v6_subset_flow_mean_qstar" if QSTAR_ONLY
                   else "v6_subset_flow_mean")

rows = []
# Authored at the width it will actually be printed at, so point sizes are true.
# Scaling a 5.5in figure down with width=0.8\linewidth instead would shrink the
# fonts by the same factor (~7.5pt -> ~6pt); regenerating keeps them at 8-9pt.
# ICLR's own template uses \includegraphics[width=0.8\linewidth] as its example,
# so the narrower width is the venue's own idiom, not a deviation.
WIDTH_FRAC = float(os.environ.get("FIG_WIDTH_FRAC", "0.80"))
fig, ax = plt.subplots(figsize=iclr_figsize(aspect=0.60, width_frac=WIDTH_FRAC))
for name, (colour, ls, marker) in STYLE.items():
    a = np.array([counts[name][m] for m in METHODS], dtype=float)  # 8 x 9
    mean, lo, hi = a.mean(0), a.min(0), a.max(0)
    ax.fill_between(xs, lo, hi, color=colour, alpha=0.16, linewidth=0)
    ax.plot(xs, mean, ls, color=colour, marker=marker, markersize=3.6,
            linewidth=1.7, label=name, zorder=3)
    for i, c in enumerate(xs):
        rows.append(dict(checkpoint="base" if c == 0 else f"ck{c}", subset=name,
                         mean=mean[i], min=lo[i], max=hi[i]))

pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)

# band legend entry, drawn as a neutral patch so it reads as "range", not a cell
band = plt.Rectangle((0, 0), 1, 1, color="#888888", alpha=0.28, linewidth=0)
h, l = ax.get_legend_handles_labels()
# below the axes, as in kfold_line_methods_auc_lr_all: the four curves plus
# their bands leave no interior space that does not collide with something
leg = ax.legend(h + [band], l + ["range over 8 methods"], frameon=False,
                ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                fontsize=8, columnspacing=1.2, handlelength=1.8)

ax.set_xticks(xs)
ax.set_xticklabels(["base\n(ck0)"] + [f"ck{c}" for c in range(1, N_CK + 1)])
ax.set_xlabel("unlearning checkpoint")
scope = r"|\mathcal{Q}^*|" if QSTAR_ONLY else "N"
ax.set_ylabel(f"questions (of ${scope}={n_pool}$)")
# cap just above the largest observed count rather than at |pool|: the base
# retained value is the maximum, and reserving empty headroom to n_pool would
# squash every other curve
_top = max(max(counts[s][m]) for s in STYLE for m in METHODS)
ax.set_ylim(-15, _top * 1.16)
ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)
sec = ax.secondary_yaxis("right",
                         functions=(lambda v: 100 * v / n_pool,
                                    lambda p: p * n_pool / 100))
sec.set_ylabel("% of all questions" if not QSTAR_ONLY
               else "% of $\\mathcal{Q}^*$")
ax.set_title("Subset membership re-assigned at each checkpoint, "
             "mean of 8 methods", fontsize=9)

# Drop the legend below the xlabel. A fixed anchor cannot know where the label
# lands -- the tick labels are two lines tall -- so measure the drawn label and
# put the legend under its bottom edge.
fig.canvas.draw()
y = ax.transAxes.inverted().transform(
    (0, ax.xaxis.get_label().get_window_extent().y0))[1]
leg.set_bbox_to_anchor((0.5, y - 0.05), transform=ax.transAxes)

fig.savefig(f"{stem}.pdf", bbox_inches="tight")
fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
plt.close(fig)

scope_txt = "Q*" if QSTAR_ONLY else "all questions"
print(f"Saved {stem.name}.pdf  ({scope_txt}, n={n_pool}, 8 methods)\n")
hdr = "base " + " ".join(f"ck{c}" for c in range(1, N_CK + 1))
print(f"{'subset':<12}{'stat':<6}{hdr}")
for name in STYLE:
    a = np.array([counts[name][m] for m in METHODS], dtype=float)
    for lbl, v in (("mean", a.mean(0)), ("min", a.min(0)), ("max", a.max(0))):
        print(f"{name:<12}{lbl:<6}" + " ".join(f"{x:5.0f}" for x in v))
