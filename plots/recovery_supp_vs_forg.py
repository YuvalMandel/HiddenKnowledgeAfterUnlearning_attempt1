"""Item 7a: recovery on the suppressed set vs recovery on the forgotten set.

Both arms are measured the same way -- steered K_ext minus a MATCHED-NORM RANDOM
control on the same held-out questions -- so the two are directly comparable.
The suppressed arm uses d_S (base minus ck8 centroids of the suppressed train
split); the forgotten arm uses d_F, built identically from forgotten questions.

The forgotten arm's random control (forg_random) existed for only 4 methods
until 2026-08-30; SLURM job 1352660 backfilled GradDiff/TAR/ELM/RMU-LAT on the
fixed grid. All pre-existing conditions reproduced to the digit.

Outputs: recovery_supp_vs_forg.csv and .png/.pdf
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

rows = []
for m in METHODS:
    d = pd.read_csv(AV / f"causal_recover_{m}.csv")
    d = d[d.alpha == 1.0].set_index("condition")
    need = ["supp_dS", "supp_random", "forg_dF", "forg_random"]
    if any(c not in d.index for c in need):
        print(f"  {m}: missing {[c for c in need if c not in d.index]}")
        continue
    rows.append(dict(
        method=DISP.get(m, m),
        n_sup=int(d.loc["supp_dS", "nq"]),
        sup_steer=d.loc["supp_dS", "kext"],
        sup_rand=d.loc["supp_random", "kext"],
        d_sup=d.loc["supp_dS", "kext"] - d.loc["supp_random", "kext"],
        n_forg=int(d.loc["forg_dF", "nq"]),
        forg_steer=d.loc["forg_dF", "kext"],
        forg_rand=d.loc["forg_random", "kext"],
        d_forg=d.loc["forg_dF", "kext"] - d.loc["forg_random", "kext"],
    ))

t = pd.DataFrame(rows).sort_values("d_sup", ascending=False).reset_index(drop=True)
t["gap"] = t.d_sup - t.d_forg
t.to_csv(REPO / "plots" / "recovery_supp_vs_forg.csv", index=False)

print(f"{'method':<10}{'n_sup':>6}{'steer':>7}{'rand':>7}{'D_sup':>8}   "
      f"{'n_forg':>7}{'steer':>7}{'rand':>7}{'D_forg':>8}{'  sup-forg':>10}")
for _, r in t.iterrows():
    print(f"{r.method:<10}{r.n_sup:>6}{r.sup_steer:>7.3f}{r.sup_rand:>7.3f}"
          f"{r.d_sup:>+8.3f}   {r.n_forg:>7}{r.forg_steer:>7.3f}"
          f"{r.forg_rand:>7.3f}{r.d_forg:>+8.3f}{r.gap:>+10.3f}")
print(f"{'MEAN':<10}{'':>6}{'':>7}{'':>7}{t.d_sup.mean():>+8.3f}   "
      f"{'':>7}{'':>7}{'':>7}{t.d_forg.mean():>+8.3f}{t.gap.mean():>+10.3f}")
print(f"\nsuppressed arm beats forgotten arm in {(t.gap > 0).sum()}/{len(t)} methods")
from scipy.stats import wilcoxon
try:
    st, p = wilcoxon(t.d_sup, t.d_forg)
    print(f"Wilcoxon signed-rank (paired, n={len(t)}): W={st:.1f}, p={p:.3f}")
except Exception as e:      # n too small / all-zero differences
    print(f"Wilcoxon unavailable: {e}")

x = np.arange(len(t))
w = 0.38
fig, ax = plt.subplots(figsize=iclr_figsize(aspect=4.2 / 7, width_frac=0.8))
ax.bar(x - w / 2, t.d_sup, w, color=SUP_C, alpha=0.88, zorder=3,
       label="Suppressed set (high hidden knowledge)")
ax.bar(x + w / 2, t.d_forg, w, color=FORG_C, alpha=0.80, zorder=3,
       label="Forgotten set (low hidden knowledge)")
ax.axhline(0, color="black", lw=0.9, alpha=0.6, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(t.method.tolist(), rotation=38, ha="right", fontsize=9)
ax.set_ylabel(r"Recovery: steered $K_\mathrm{ext}$ $-$ matched random (pp)",
              fontsize=8.5)
ax.tick_params(labelsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.55, zorder=0)
ax.set_axisbelow(True)
for xi, v in zip(x - w / 2, t.d_sup):
    ax.text(xi, v + 0.012, f"{v:+.2f}", ha="center", va="bottom",
            fontsize=6.0, color=SUP_C)
for xi, v in zip(x + w / 2, t.d_forg):
    ax.text(xi, v + 0.012, f"{v:+.2f}", ha="center", va="bottom",
            fontsize=6.0, color=FORG_C)
ax.legend(fontsize=7.6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
          frameon=False, columnspacing=1.4, handlelength=1.4)
fig.tight_layout()
out = REPO / "plots" / "recovery_supp_vs_forg"
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}.csv / .png / .pdf")
plt.close(fig)
