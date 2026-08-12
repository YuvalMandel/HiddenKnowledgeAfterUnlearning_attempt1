#!/usr/bin/env python3
"""Hidden-knowledge gap (K_int - K_ext) over checkpoints, per question subset.

Shows the divergence directly rather than as a static endpoint effect size:
every subset starts at zero by construction (Q* is pre-filtered to base-model
K_int = K_ext = 1) and pulls apart as unlearning proceeds.

One method is shown for legibility. RepNoise is the default because it has the
best-populated subsets of the eight (306/245/68/82 retained/suppressed/
forgotten/lucky) and the highest suppression rate; GradDiff would put only 21
questions in the forgotten curve.

Outputs: plots/v6_gap_trajectory.pdf/.png
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

# Okabe-Ito: colour-blind safe, and distinguishable in greyscale by line style.
STYLE = {
    "$\\mathcal{Q}^*$ (all)": ("#000000", "-",  "o"),
    "Retained":               ("#56B4E9", "--", "s"),
    "Suppressed":             ("#D55E00", "-",  "^"),
    "Forgotten":              ("#CC79A7", "-.", "v"),
    "Lucky":                  ("#009E73", ":",  "D"),
}


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
    "$\\mathcal{Q}^*$ (all)": q8.index,
    "Retained":  q8[hi_i & hi_e].index,
    "Suppressed": q8[hi_i & ~hi_e].index,
    "Forgotten": q8[~hi_i & ~hi_e].index,
    "Lucky":     q8[~hi_i & hi_e].index,
}

# checkpoint 0 is the base model, where the gap is 0 for all of Q* by definition
per_ck = {0: base}
for c in range(1, N_CK + 1):
    per_ck[c] = load(f"{METHOD}_ck{c}")

fig, ax = plt.subplots(figsize=iclr_figsize(aspect=0.58))
xs = list(range(0, N_CK + 1))

for name, idx in groups.items():
    colour, ls, marker = STYLE[name]
    mean, sem = [], []
    for c in xs:
        d = per_ck[c]
        sub = d.loc[d.index.intersection(idx)]
        gap = sub.k_internal - sub.k_external
        mean.append(gap.mean())
        sem.append(gap.std(ddof=1) / np.sqrt(len(gap)))
    mean, sem = np.array(mean), np.array(sem)
    ax.plot(xs, mean, ls, color=colour, marker=marker, markersize=3.2,
            linewidth=1.4, label=f"{name} ($n{{=}}{len(idx)}$)")
    ax.fill_between(xs, mean - sem, mean + sem, color=colour, alpha=0.15,
                    linewidth=0)

ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
# subsets are assigned at ck8, so the ordering there is partly definitional
ax.axvline(N_CK, color="black", linestyle=":", linewidth=0.8, alpha=0.45)
ax.annotate("subsets assigned here", xy=(N_CK, ax.get_ylim()[0]),
            xytext=(-4, 4), textcoords="offset points",
            ha="right", va="bottom", fontsize=6.5, alpha=0.7, rotation=90)

ax.set_xticks(xs)
ax.set_xticklabels(["base"] + [f"ck{c}" for c in range(1, N_CK + 1)])
ax.set_xlabel("unlearning checkpoint")
ax.set_ylabel(r"hidden-knowledge gap  $K_\mathrm{int}-K_\mathrm{ext}$")
ax.set_title(f"{METHOD.replace('_', '\\&')}: gap opens progressively during unlearning")
ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7)

stem = SAVE_DIR / "v6_gap_trajectory"
fig.savefig(f"{stem}.pdf", bbox_inches="tight")
fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {stem.name}.pdf  (method={METHOD})")
for name, idx in groups.items():
    print(f"  {name:<22} n={len(idx)}")
