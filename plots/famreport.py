"""K_int / K_ext / gap / 4-subset profile for a model pair, from k_scores.parquet."""
import sys, pandas as pd, numpy as np
from pathlib import Path
from scipy import stats

def load(m):
    d = pd.read_parquet(Path("inside_out_out") / m / "k_scores.parquet")
    return d.drop_duplicates("question_idx").set_index("question_idx").sort_index()

def row(name, d):
    ki, ke = d.k_internal.values, d.k_external.values
    sup = (ki > .5) & (ke <= .5); ret = (ki > .5) & (ke > .5)
    forg = (ki <= .5) & (ke <= .5); luck = (ki <= .5) & (ke > .5)
    n = len(d)
    t = stats.wilcoxon(ki, ke)
    return dict(model=name, n=n, K_int=ki.mean(), K_ext=ke.mean(),
                gap=ki.mean() - ke.mean(), p=t.pvalue,
                retained=100*ret.mean(), suppressed=100*sup.mean(),
                forgotten=100*forg.mean(), lucky=100*luck.mean(),
                best_layer=d.best_layer.iloc[0], probe_auc=d.test_auc.iloc[0])

pairs = sys.argv[1:]
rows = [row(m, load(m)) for m in pairs]
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if len(pairs) == 2:
    b, u = load(pairs[0]), load(pairs[1])
    common = b.index.intersection(u.index)
    b, u = b.loc[common], u.loc[common]
    print(f"\n--- paired, n={len(common)} ---")
    print(f"K_ext  {pairs[0]} {b.k_external.mean():.4f} -> {pairs[1]} {u.k_external.mean():.4f}"
          f"   drop {b.k_external.mean()-u.k_external.mean():+.4f}")
    print(f"K_int  {pairs[0]} {b.k_internal.mean():.4f} -> {pairs[1]} {u.k_internal.mean():.4f}"
          f"   drop {b.k_internal.mean()-u.k_internal.mean():+.4f}")
    print(f"gap    {pairs[0]} {(b.k_internal-b.k_external).mean():+.4f}"
          f" -> {pairs[1]} {(u.k_internal-u.k_external).mean():+.4f}")
    # retention relative to base, 0.5 = chance floor
    rr = lambda x, base: (x - .5) / (base - .5)
    print(f"retention vs base: K_ext {100*rr(u.k_external.mean(), b.k_external.mean()):.1f}%"
          f"   K_int {100*rr(u.k_internal.mean(), b.k_internal.mean()):.1f}%")
    # where did the base-known questions go
    known = b.k_internal > .5
    print(f"\nof {known.sum()} questions the BASE knew internally, after unlearning:")
    for lab, m in [("still external (retained)", (u.k_internal > .5) & (u.k_external > .5)),
                   ("internal only (suppressed)", (u.k_internal > .5) & (u.k_external <= .5)),
                   ("gone (forgotten)", (u.k_internal <= .5) & (u.k_external <= .5)),
                   ("lucky", (u.k_internal <= .5) & (u.k_external > .5))]:
        print(f"  {lab:28s} {(known & m).sum():4d}  ({100*(known & m).sum()/known.sum():.1f}%)")
