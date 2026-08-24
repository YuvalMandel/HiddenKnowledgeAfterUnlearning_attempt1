"""Do the three layer-selection rules actually give similar recovery?

YM: "we had multiple attempts at choosing which layers to pick, I think they gave
similar results, but some are more defensible than others."

Three rules exist in plots/activation_vectors/:
  fixed  -- one method-independent grid {3,6,9,12,15}          (headline)
  topk5  -- the 5 layers with highest suppressed-vs-retained AUC, per method
  band5  -- 5 consecutive layers centred on that profile's best layer, per method

Compared at the pre-registered alpha=1, condition supp_dS (steered) against
supp_random (matched-norm control). The control matters more than the raw K_ext:
a rule that lifts both equally has bought nothing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

VEC = Path("plots/activation_vectors")
METHODS = ["RepNoise", "GradDiff", "PB_J", "TAR", "RR", "ELM", "RMU", "RMU-LAT"]
VARIANTS = {"fixed": "", "topk5": "_topk5", "band5": "_band5"}


def get(method, suffix, cond, alpha=1.0):
    p = VEC / f"causal_recover_{method}{suffix}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    r = d[(d.condition == cond) & (np.isclose(d.alpha, alpha))]
    return float(r.kext.iloc[0]) if len(r) else None


rows = []
print(f"{'method':<10}" + "".join(f"{v:>22}" for v in VARIANTS))
print(f"{'':<10}" + "".join(f"{'steer':>8}{'rand':>7}{'diff':>7}" for _ in VARIANTS))
print("-" * 76)
for m in METHODS:
    line = f"{m:<10}"
    rec = {"method": m}
    for name, suf in VARIANTS.items():
        s = get(m, suf, "supp_dS")
        r = get(m, suf, "supp_random")
        if s is None:
            line += f"{'--':>22}"
            rec[name] = np.nan
            continue
        d = s - r if r is not None else np.nan
        rec[name] = d
        rec[name + "_steer"] = s
        line += f"{s:>8.3f}{r:>7.3f}{d:>+7.3f}"
    rows.append(rec)
    print(line)

df = pd.DataFrame(rows)
print("-" * 76)
print(f"{'MEAN diff':<10}" + "".join(
    f"{'':>15}{df[v].mean():>+7.3f}" for v in VARIANTS))
print()
print("Steered K_ext, by variant (mean over the 8 methods):")
for v in VARIANTS:
    c = v + "_steer"
    if c in df:
        print(f"  {v:<7} {df[c].mean():.3f}   "
              f"(available for {df[c].notna().sum()}/8 methods)")
print()
print("Pairwise agreement on the steer-minus-random advantage:")
for a in VARIANTS:
    for b in VARIANTS:
        if a < b:
            sub = df[[a, b]].dropna()
            if len(sub) > 2:
                print(f"  {a:>6} vs {b:<6} n={len(sub)}  "
                      f"corr={sub[a].corr(sub[b]):+.2f}  "
                      f"mean|delta|={np.abs(sub[a]-sub[b]).mean():.3f}")
print()
print("Rank of methods by advantage, per variant (best first):")
for v in VARIANTS:
    sub = df[["method", v]].dropna().sort_values(v, ascending=False)
    print(f"  {v:<7} " + " > ".join(sub.method))
