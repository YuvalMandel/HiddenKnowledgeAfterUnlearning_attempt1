#!/usr/bin/env python3
"""Is the forgotten:lucky split 50:50, as chance on the external axis predicts?

Both subsets share K_int <= 1/2; they differ only in the sign of K_ext. So among
questions the probe cannot read, the split between them IS the external axis's
success rate. The prediction is exact, not approximate:

    K_ext averages |W| = 3 pairwise comparisons, so it takes values
    {0, 1/3, 2/3, 1} and can never equal 1/2. Under chance each comparison is a
    coin flip, and P(K_ext > 1/2) = P(>=2 of 3 heads) = 3/8 + 1/8 = 1/2 exactly.

So chance predicts lucky / (forgotten + lucky) = 0.5. This measures it, with a
test that respects the fact that the eight methods score the same 1,273
questions -- a naive binomial over method x question would treat those as
independent and understate the interval badly.

Usage: python plots/forgotten_vs_lucky.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "inside_out_out"
METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
AV = ROOT / "plots" / "activation_vectors"


def load(model_id):
    df = pd.read_parquet(OUT / model_id / "k_scores.parquet")
    cv = df[(df.split_type == "cv") & (df.domain == "bio") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    return cv.groupby("question_idx").agg(ki=("k_internal", "mean"),
                                          ke=("k_external", "mean"))


def main():
    rows, per_q = [], {}
    for m in METHODS:
        d = load(f"{m}_ck8")
        empty = d[d.ki <= 0.5]                    # probe finds nothing
        lucky = int((empty.ke > 0.5).sum())
        forg = int((empty.ke <= 0.5).sum())
        rows.append(dict(method=m, forgotten=forg, lucky=lucky,
                         n=forg + lucky,
                         lucky_share=round(lucky / (forg + lucky), 4)))
        per_q[m] = (empty.ke > 0.5).astype(int)
        print(f"  {m:<9} forgotten={forg:>4}  lucky={lucky:>4}  "
              f"lucky share={100*lucky/(forg+lucky):>5.1f}%")

    df = pd.DataFrame(rows)
    tot_f, tot_l = int(df.forgotten.sum()), int(df.lucky.sum())
    share = tot_l / (tot_f + tot_l)
    print(f"\npooled: forgotten={tot_f}  lucky={tot_l}  "
          f"lucky share={100*share:.2f}%  (chance predicts 50.00%)")

    # cluster the bootstrap on METHODS: the 8 series are the same questions
    rng = np.random.default_rng(0)
    shares = df.lucky.values / df.n.values
    boot = [np.mean(rng.choice(shares, len(shares), replace=True))
            for _ in range(20000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"mean of per-method shares = {100*shares.mean():.2f}% "
          f"[{100*lo:.2f}, {100*hi:.2f}] (bootstrap over the 8 methods)")
    print(f"chance (50%) inside the interval: {lo <= 0.5 <= hi}")
    print(f"per-method range: {100*shares.min():.1f}% - {100*shares.max():.1f}%")

    df.to_csv(AV / "forgotten_vs_lucky.csv", index=False)
    print(f"\nwrote {AV/'forgotten_vs_lucky.csv'}")


if __name__ == "__main__":
    main()
