#!/usr/bin/env python3
"""Does steering recovery concentrate where internal knowledge is strong?

Claim #5 currently reads as a method lottery: four of eight methods significant,
and only two actually above chance. But K_int is quantised to {0, 1/3, 2/3, 1},
so the suppressed set splits into questions the probe ranks perfectly
(K_int_ck8 = 1.0) and questions one flipped comparison from the boundary
(K_int_ck8 = 2/3).

If recovery tracks internal strength rather than method identity, that turns a
lottery into a mechanism: steering restores what is still represented.

Input is the per-question output of causal_recover_pvalue.py at the
pre-registered alpha=1 (no alpha sweep, so no cherry-picking):
    plots/activation_vectors/causal_recover_pvalue_perq_<METHOD>.csv
paired with K_int at ck8 from inside_out_out/<METHOD>_ck8/k_scores.parquet.

Outputs: plots/v6_recovery_by_strength.csv
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "inside_out_out"
VEC_DIR = REPO / "plots" / "activation_vectors"
SAVE_DIR = Path(__file__).resolve().parent
SEL = dict(split_type="cv", domain="bio", clf="LR",
           probe_type="own", layer_config="best_layer")
# parquet uses PB_J, the recovery CSVs use PB_J too; paper prints PB&J
STRONG, MARGINAL = 1.0, 2.0 / 3.0


def load_k(model_id):
    d = pd.read_parquet(OUT_DIR / model_id / "k_scores.parquet")
    for k, v in SEL.items():
        d = d[d[k] == v]
    return (d[["question_idx", "k_internal", "k_external"]]
            .groupby("question_idx", as_index=False).mean()
            .set_index("question_idx"))


rows = []
for f in sorted(glob.glob(str(VEC_DIR / "causal_recover_pvalue_perq_*.csv"))):
    d = pd.read_csv(f)
    method = d.method.iloc[0]
    kk = load_k(f"{method}_ck8")

    s = d[d.qset == "suppressed"]
    wide = s.pivot_table(index="question_idx", columns="condition",
                         values="kext", aggfunc="first")
    wide = wide.join(kk["k_internal"].rename("kint_ck8"), how="inner")
    wide["recovery"] = wide["supp_dS"] - wide["baseline"]
    wide["vs_random"] = wide["supp_dS"] - wide["supp_random"]

    for name, sel in (("strong (K_int=1)", wide.kint_ck8.round(3) == 1.0),
                      ("marginal (K_int=2/3)", wide.kint_ck8.round(3) == 0.667)):
        g = wide[sel]
        if len(g) < 3:
            continue
        # paired: same questions, steered vs random direction of equal norm
        try:
            p = wilcoxon(g["supp_dS"], g["supp_random"],
                         alternative="greater").pvalue
        except ValueError:            # all-zero differences
            p = 1.0
        rows.append(dict(method=method, stratum=name, n=len(g),
                         baseline=g["baseline"].mean(),
                         steered=g["supp_dS"].mean(),
                         random=g["supp_random"].mean(),
                         recovery=g["recovery"].mean(),
                         vs_random=g["vs_random"].mean(), p=p))

df = pd.DataFrame(rows)
df.to_csv(SAVE_DIR / "v6_recovery_by_strength.csv", index=False)

print("Steering recovery on SUPPRESSED questions, alpha=1, by internal strength")
print("(kext: baseline = unsteered ck8; steered = +1*d_S; random = matched-norm "
      "control)\n")
hdr = (f"{'method':<10}{'stratum':<22}{'n':>4}{'base':>7}{'steer':>7}"
       f"{'rand':>7}{'steer-base':>12}{'steer-rand':>12}{'p':>9}")
print(hdr)
print("-" * len(hdr))
for m in sorted(df.method.unique()):
    for _, r in df[df.method == m].iterrows():
        print(f"{r.method:<10}{r.stratum:<22}{r.n:>4}{r.baseline:>7.3f}"
              f"{r.steered:>7.3f}{r['random']:>7.3f}{r.recovery:>+12.3f}"
              f"{r.vs_random:>+12.3f}{r.p:>9.3f}")

print("\n" + "=" * 78)
for name in ("strong (K_int=1)", "marginal (K_int=2/3)"):
    g = df[df.stratum == name]
    if g.empty:
        continue
    print(f"{name:<22} n={int(g.n.sum()):<4} "
          f"mean steered K_ext {g.steered.mean():.3f}   "
          f"vs baseline {g.recovery.mean():+.3f}   "
          f"vs random {g.vs_random.mean():+.3f}   "
          f"methods p<0.05: {int((g.p < 0.05).sum())}/{len(g)}")
