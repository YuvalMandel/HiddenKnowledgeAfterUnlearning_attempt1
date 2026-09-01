"""Item 7a, levels view: steering vs a matched-norm RANDOM control, on BOTH the
suppressed and the forgotten set.

The companion script `recovery_supp_vs_forg.py` plots only the two *deltas*
(steered minus random). This one plots the absolute $K_ext$ levels behind them,
so the random control is visible rather than subtracted away:

    unsteered baseline (alpha = 0)  ->  matched-norm random  ->  steered

Reading the figure:
  * Suppressed panel, RepNoise/GradDiff: the steered bar sits far above random,
    so the recovery is DIRECTION-SPECIFIC.
  * Suppressed panel, RMU/RR/RMU-LAT: random already lifts K_ext most of the way
    to where steering gets it, so their apparent recovery is NON-SPECIFIC -- a
    perturbation of that norm would do it too.
  * Forgotten panel: the two bars sit close together for nearly every method,
    which is the point of the comparison. The same construction applied where an
    independent probe finds no internal knowledge recovers far less.

alpha = 0 is the unsteered model: adding 0 x d gives the same value for the
steered and random conditions, which is why one grey bar serves as the baseline
for both.

Usage:  python plots/recovery_supp_vs_forg_levels.py
Outputs: plots/recovery_supp_vs_forg_levels.{csv,png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hk_utils import iclr_figsize, use_iclr_style  # noqa: E402

use_iclr_style()

REPO = Path(__file__).resolve().parent.parent
AV = REPO / "plots" / "activation_vectors"
METHODS = ["RepNoise", "GradDiff", "PB_J", "TAR", "RR", "ELM", "RMU", "RMU-LAT"]
DISP = {"PB_J": "PB&J"}

SUP_C, FORG_C = "#762a83", "#1b7837"
BASE_C = "#c7c7c7"
CHANCE = 0.5          # K_ext is a pairwise win-fraction over 3 distractors

# Injection rule. "_allnorm" = all 32 layers with the total injected norm matched
# to the fixed grid's budget; "" = the fixed {3,6,9,12,15} grid. The paper's 5.3
# uses allnorm, so that is the default here.
MODE = "_allnorm"

rows = []
for m in METHODS:
    d = pd.read_csv(AV / f"causal_recover_{m}{MODE}.csv")
    a1 = d[d.alpha == 1.0].set_index("condition")
    a0 = d[d.alpha == 0.0].set_index("condition")
    need = ["supp_dS", "supp_random", "forg_dF", "forg_random"]
    missing = [c for c in need if c not in a1.index or c not in a0.index]
    if missing:
        print(f"  {m}: missing {missing} -- skipped")
        continue
    # alpha=0 is the unsteered model, so the two conditions must agree there
    assert np.isclose(a0.loc["supp_dS", "kext"], a0.loc["supp_random", "kext"])
    assert np.isclose(a0.loc["forg_dF", "kext"], a0.loc["forg_random", "kext"])
    rows.append(dict(
        method=DISP.get(m, m),
        n_sup=int(a1.loc["supp_dS", "nq"]),
        sup_base=a0.loc["supp_dS", "kext"],
        sup_rand=a1.loc["supp_random", "kext"],
        sup_steer=a1.loc["supp_dS", "kext"],
        n_forg=int(a1.loc["forg_dF", "nq"]),
        forg_base=a0.loc["forg_dF", "kext"],
        forg_rand=a1.loc["forg_random", "kext"],
        forg_steer=a1.loc["forg_dF", "kext"],
    ))

t = pd.DataFrame(rows)
t["d_sup"] = t.sup_steer - t.sup_rand
t["d_forg"] = t.forg_steer - t.forg_rand
t = t.sort_values("d_sup", ascending=False).reset_index(drop=True)
t.to_csv(REPO / "plots" / "recovery_supp_vs_forg_levels.csv", index=False)

# ---------------------------------------------------------------- table -----
print(f"{'':<10}{'|':>3}{'SUPPRESSED':^34}{'|':>3}{'FORGOTTEN':^34}")
print(f"{'method':<10}{'|':>3}{'n':>5}{'base':>7}{'rand':>7}{'steer':>7}"
      f"{'delta':>8}{'|':>3}{'n':>5}{'base':>7}{'rand':>7}{'steer':>7}{'delta':>8}")
print("-" * 88)
for _, r in t.iterrows():
    print(f"{r.method:<10}{'|':>3}{r.n_sup:>5}{r.sup_base:>7.3f}{r.sup_rand:>7.3f}"
          f"{r.sup_steer:>7.3f}{r.d_sup:>+8.3f}{'|':>3}"
          f"{r.n_forg:>5}{r.forg_base:>7.3f}{r.forg_rand:>7.3f}"
          f"{r.forg_steer:>7.3f}{r.d_forg:>+8.3f}")
print("-" * 88)
print(f"{'MEAN':<10}{'|':>3}{'':>5}{t.sup_base.mean():>7.3f}"
      f"{t.sup_rand.mean():>7.3f}{t.sup_steer.mean():>7.3f}{t.d_sup.mean():>+8.3f}"
      f"{'|':>3}{'':>5}{t.forg_base.mean():>7.3f}{t.forg_rand.mean():>7.3f}"
      f"{t.forg_steer.mean():>7.3f}{t.d_forg.mean():>+8.3f}")
print(f"\nsuppressed delta exceeds forgotten delta in "
      f"{(t.d_sup > t.d_forg).sum()}/{len(t)} methods")
print(f"steered above chance ({CHANCE}): suppressed "
      f"{(t.sup_steer > CHANCE).sum()}/{len(t)}, forgotten "
      f"{(t.forg_steer > CHANCE).sum()}/{len(t)}")

# ---------------------------------------------------------------- figure ----
x = np.arange(len(t))
w = 0.27
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True,
                         figsize=iclr_figsize(aspect=4.9 / 7, width_frac=1.0))

panels = [
    (axes[0], "Suppressed set  (probe finds internal knowledge)", SUP_C,
     "sup_base", "sup_rand", "sup_steer", "n_sup", r"$d_S$"),
    (axes[1], "Forgotten set  (probe finds none)", FORG_C,
     "forg_base", "forg_rand", "forg_steer", "n_forg", r"$d_F$"),
]

for ax, title, col, cb, cr, cs, cn, vec in panels:
    ax.bar(x - w, t[cb], w, color=BASE_C, zorder=3,
           label="unsteered baseline")
    ax.bar(x, t[cr], w, color=col, alpha=0.38, hatch="///",
           edgecolor=col, linewidth=0.5, zorder=3,
           label="matched-norm random direction")
    ax.bar(x + w, t[cs], w, color=col, alpha=0.92, zorder=3,
           label=f"steered ({vec})")
    ax.axhline(CHANCE, color="black", ls="--", lw=0.9, alpha=0.65, zorder=4)
    ax.text(len(t) - 0.42, CHANCE + 0.018, "chance", fontsize=6.2,
            ha="right", va="bottom", alpha=0.75)
    ax.set_title(title, fontsize=8.2, pad=3)
    ax.set_ylabel(r"$K_\mathrm{ext}$", fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.6, ncol=3, loc="upper right", frameon=False,
              handlelength=1.3, columnspacing=1.0, borderpad=0.2)
    # Only the steered bar is labelled. Three labels per group collide at any
    # legible size on a 5.5in column; the random bar's height is readable
    # against the grey baseline beside it, which is the comparison that matters.
    for xi, v in zip(x + w, t[cs]):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                fontsize=7.0, color=col, weight="bold")

axes[1].set_xticks(x)
axes[1].set_xticklabels(
    [f"{m}\n$n$={ns}/{nf}" for m, ns, nf in zip(t.method, t.n_sup, t.n_forg)],
    fontsize=7.0)
axes[1].set_xlabel("method  ($n$ = held-out suppressed / forgotten questions)",
                   fontsize=7.5)

fig.tight_layout(h_pad=1.1)
out = REPO / "plots" / "recovery_supp_vs_forg_levels"
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: {out}.csv / .png / .pdf")
plt.close(fig)
