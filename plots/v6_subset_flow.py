#!/usr/bin/env python3
"""Subset populations over checkpoints, reclassified at every checkpoint.

Everywhere else in the paper the four subsets are assigned once, at ck8, and
questions keep that label for the whole trajectory. Here each question is
re-classified independently at each checkpoint from that checkpoint's own
K_int and K_ext, so the curves show how the population redistributes during
unlearning rather than how a fixed cohort behaves.

Two consequences worth knowing when reading the figure:
  * ck0 (base) is all-retained by construction -- Q* is pre-filtered to
    K_int = K_ext = 1, so every question is above threshold on both axes.
  * The four counts sum to |Q*| at every checkpoint (asserted below); a
    question moving between subsets is a real reclassification, not churn
    from missing data.

Outputs: plots/v6_subset_flow.pdf/.png, plots/v6_subset_flow.csv
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
KNOWS = 0.5
SEL = dict(split_type="cv", domain="bio", clf="LR",
           probe_type="own", layer_config="best_layer")

# same colours/styles as the other subset figures, greyscale-safe via linestyle
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
n_qstar = len(qstar)

xs = list(range(0, N_CK + 1))
counts = {name: [] for name in STYLE}
for c in xs:
    d = base if c == 0 else load(f"{METHOD}_ck{c}")
    q = d.loc[d.index.intersection(qstar)]
    hi_i, hi_e = q.k_internal > KNOWS, q.k_external > KNOWS
    n = {
        "Retained":   int((hi_i & hi_e).sum()),
        "Suppressed": int((hi_i & ~hi_e).sum()),
        "Forgotten":  int((~hi_i & ~hi_e).sum()),
        "Lucky":      int((~hi_i & hi_e).sum()),
    }
    assert sum(n.values()) == len(q), f"ck{c}: {sum(n.values())} != {len(q)}"
    for k, v in n.items():
        counts[k].append(v)

df = pd.DataFrame(counts, index=[f"ck{c}" if c else "base" for c in xs])
df.index.name = "checkpoint"
df.to_csv(SAVE_DIR / "v6_subset_flow.csv")

fig, ax = plt.subplots(figsize=iclr_figsize(aspect=0.60))
for name, (colour, ls, marker) in STYLE.items():
    ax.plot(xs, counts[name], ls, color=colour, marker=marker, markersize=3.6,
            linewidth=1.6, label=name)

ax.set_xticks(xs)
ax.set_xticklabels(["base\n(ck0)"] + [f"ck{c}" for c in range(1, N_CK + 1)])
ax.set_xlabel("unlearning checkpoint")
ax.set_ylabel(f"questions (of $|\\mathcal{{Q}}^*|={n_qstar}$)")
ax.set_ylim(-15, n_qstar * 1.04)
ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)

# right axis in % of Q*, since the absolute count is method-independent only
# because Q* is fixed
sec = ax.secondary_yaxis(
    "right", functions=(lambda v: 100 * v / n_qstar,
                        lambda p: p * n_qstar / 100))
sec.set_ylabel(r"\% of $\mathcal{Q}^*$" if plt.rcParams.get("text.usetex")
               else "% of $\\mathcal{Q}^*$")

label = METHOD.replace("_", "&")
ax.set_title(f"{label}: subset membership re-assigned at each checkpoint",
             fontsize=9)
ax.legend(frameon=False, ncol=2, loc="center left", fontsize=8)

stem = SAVE_DIR / "v6_subset_flow"
fig.savefig(f"{stem}.pdf", bbox_inches="tight")
fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved {stem.name}.pdf  (method={METHOD}, |Q*|={n_qstar})\n")
print(df.to_string())
print(f"\nrow sums (must all equal {n_qstar}): "
      f"{sorted(set(df.sum(axis=1).tolist()))}")
