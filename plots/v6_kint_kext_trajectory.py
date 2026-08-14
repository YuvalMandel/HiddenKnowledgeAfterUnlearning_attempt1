#!/usr/bin/env python3
"""K_int and K_ext over checkpoints, shown separately for Q* and each subset.

Supersedes v6_gap_trajectory.py, which plotted only the difference
K_int - K_ext. The difference hides which of the two axes moved: the gap in
suppressed questions opens almost entirely because K_ext collapses while K_int
stays flat, and that asymmetry is the paper's actual claim. Here both axes are
drawn and the gap is retained as the shaded area between them.

Every panel starts at (1.0, 1.0) because Q* is pre-filtered to base-model
K_int = K_ext = 1, so all divergence visible is produced by unlearning.

One method is shown for legibility. RepNoise is the default because it has the
best-populated subsets of the eight (306/245/68/82 retained/suppressed/
forgotten/lucky); GradDiff would put only 21 questions in the forgotten panel.
Pass another method name as argv[1].

Outputs: plots/v6_kint_kext_trajectory.pdf/.png
"""
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
METHOD = sys.argv[1] if len(sys.argv) > 1 else "RepNoise"
N_CK = 8
SEL = dict(split_type="cv", domain="bio", clf="LR",
           probe_type="own", layer_config="best_layer")

# Okabe-Ito, and separated by line style so the pair survives greyscale.
INT_C, EXT_C = "#0072B2", "#D55E00"
GAP_C = "#999999"


def load(model_id):
    d = pd.read_parquet(OUT_DIR / model_id / "k_scores.parquet")
    for k, v in SEL.items():
        d = d[d[k] == v]
    return d.set_index("question_idx")[["k_internal", "k_external"]]


base = load("base")
qstar = base[(base.k_internal == 1.0) & (base.k_external == 1.0)].index

ck8 = load(f"{METHOD}_ck8")
q8 = ck8.loc[ck8.index.intersection(qstar)]
hi_i, hi_e = q8.k_internal > 0.5, q8.k_external > 0.5
groups = {
    r"$\mathcal{Q}^*$ (all)": q8.index,
    "Retained":   q8[hi_i & hi_e].index,
    "Suppressed": q8[hi_i & ~hi_e].index,
    "Forgotten":  q8[~hi_i & ~hi_e].index,
    "Lucky":      q8[~hi_i & hi_e].index,
}

# checkpoint 0 is the base model, where K_int = K_ext = 1 for all of Q*
per_ck = {0: base}
for c in range(1, N_CK + 1):
    per_ck[c] = load(f"{METHOD}_ck{c}")

xs = list(range(0, N_CK + 1))
fig, axes = plt.subplots(2, 3, figsize=iclr_figsize(aspect=0.62),
                         sharex=True, sharey=True)
flat = axes.ravel()

for ax, (name, idx) in zip(flat, groups.items()):
    mi, si, me, se = [], [], [], []
    for c in xs:
        d = per_ck[c]
        sub = d.loc[d.index.intersection(idx)]
        n = max(len(sub), 1)
        mi.append(sub.k_internal.mean())
        si.append(sub.k_internal.std(ddof=1) / np.sqrt(n))
        me.append(sub.k_external.mean())
        se.append(sub.k_external.std(ddof=1) / np.sqrt(n))
    mi, si = np.array(mi), np.array(si)
    me, se = np.array(me), np.array(se)

    # the quantity the superseded figure plotted, kept as area
    ax.fill_between(xs, me, mi, color=GAP_C, alpha=0.30, linewidth=0,
                    label=r"gap $K_\mathrm{int}-K_\mathrm{ext}$")
    ax.fill_between(xs, mi - si, mi + si, color=INT_C, alpha=0.18, linewidth=0)
    ax.fill_between(xs, me - se, me + se, color=EXT_C, alpha=0.18, linewidth=0)
    ax.plot(xs, mi, "-", color=INT_C, marker="o", markersize=2.6,
            linewidth=1.4, label=r"$K_\mathrm{int}$")
    ax.plot(xs, me, "--", color=EXT_C, marker="s", markersize=2.6,
            linewidth=1.4, label=r"$K_\mathrm{ext}$")

    # subsets are assigned by thresholding at 0.5 at ck8
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle=":", alpha=0.55)
    ax.axvline(N_CK, color="black", linewidth=0.7, linestyle=":", alpha=0.45)
    ax.set_title(f"{name} ($n{{=}}{len(idx)}$)", fontsize=8, pad=3)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.45)
    ax.set_axisbelow(True)

# last cell carries the legend instead of a sixth panel
lg = flat[-1]
lg.axis("off")
handles, labels = flat[0].get_legend_handles_labels()
order = [1, 2, 0]
lg.legend([handles[i] for i in order], [labels[i] for i in order],
          frameon=False, loc="center", fontsize=7.5,
          title=f"{METHOD.replace('_', '&')}", title_fontsize=8)
lg.text(0.5, 0.16, "dotted line at 0.5:\nsubset threshold, applied at ck8",
        ha="center", va="center", fontsize=7.0, alpha=0.75,
        transform=lg.transAxes)

ticklabels = ["base"] + [f"ck{c}" if c % 2 == 0 else ""
                         for c in range(1, N_CK + 1)]
# top-right panel sits above the legend cell, so it needs its own x labels;
# sharex hides them by default on every non-bottom row
for ax in list(axes[1]) + [axes[0][2]]:
    ax.set_xticks(xs)
    ax.set_xticklabels(ticklabels)
    ax.tick_params(labelbottom=True)
for ax in axes[:, 0]:
    ax.set_ylabel("knowledge score $K$")
fig.supxlabel("unlearning checkpoint", fontsize=8.5, y=0.02)
ax.set_ylim(-0.03, 1.05)

fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.9)
stem = SAVE_DIR / "v6_kint_kext_trajectory"
fig.savefig(f"{stem}.pdf", bbox_inches="tight")
fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved {stem.name}.pdf  (method={METHOD})\n")
hdr = "base " + " ".join(f"ck{c}" for c in range(1, N_CK + 1))
print(f"{'group':<16}{'axis':<7}{hdr}")
for name, idx in groups.items():
    for axis, col in (("K_int", "k_internal"), ("K_ext", "k_external")):
        vals = [per_ck[c].loc[per_ck[c].index.intersection(idx), col].mean()
                for c in xs]
        label = name.replace(r"$\mathcal{Q}^*$ (all)", "Q* (all)")
        print(f"{label:<16}{axis:<7}" + " ".join(f"{v:.3f}" for v in vals))
