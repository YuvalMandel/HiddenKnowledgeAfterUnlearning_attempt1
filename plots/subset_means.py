#!/usr/bin/env python3
"""E[K_int], E[K_ext] and E[HK] per method, per subset, at ck8.

The per-method breakdown behind tab:mechanism, which pools the eight methods and
omits the lucky cell.

Read the constraint before reading the numbers. Subsets are defined by
thresholding the same two scores at 1/2, and with |W|=3 the scores live in
{0, 1/3, 2/3, 1}. So within a cell the means are bounded by construction:

    retained    E[K_int] >= 2/3,  E[K_ext] >= 2/3
    suppressed  E[K_int] >= 2/3,  E[K_ext] <= 1/3,  E[HK] >= 1/3
    forgotten   E[K_int] <= 1/3,  E[K_ext] <= 1/3
    lucky       E[K_int] <= 1/3,  E[K_ext] >= 2/3,  E[HK] <= -1/3

The informative content is therefore where a cell sits *within* its allowed
band and how methods differ, not the level itself. E[HK] in the suppressed cell
cannot be near zero however little hidden knowledge there is.

Usage: python plots/subset_means.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "inside_out_out"
AV = ROOT / "plots" / "activation_vectors"
METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
SUBSETS = ["retained", "suppressed", "forgotten", "lucky"]


def load(model_id):
    df = pd.read_parquet(OUT / model_id / "k_scores.parquet")
    cv = df[(df.split_type == "cv") & (df.domain == "bio") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    return cv.groupby("question_idx").agg(ki=("k_internal", "mean"),
                                          ke=("k_external", "mean"))


def label(d):
    hi_i, hi_e = d.ki > .5, d.ke > .5
    return np.select([hi_i & hi_e, hi_i & ~hi_e, ~hi_i & ~hi_e],
                     SUBSETS[:3], default="lucky")


def main():
    rows = []
    for m in METHODS:
        d = load(f"{m}_ck8")
        d = d.assign(subset=label(d), hk=d.ki - d.ke)
        for s in SUBSETS:
            g = d[d.subset == s]
            rows.append(dict(method=m, subset=s, n=len(g),
                             k_int=g.ki.mean(), k_ext=g.ke.mean(),
                             hk=g.hk.mean()))
    df = pd.DataFrame(rows)
    df.to_csv(AV / "subset_means_ck8.csv", index=False)

    for s in SUBSETS:
        sub = df[df.subset == s]
        print(f"\n{s.upper():<11}{'n':>6}{'E[K_int]':>10}{'E[K_ext]':>10}{'E[HK]':>9}")
        print("-" * 46)
        for _, r in sub.iterrows():
            print(f"{r.method:<11}{r.n:>6}{r.k_int:>10.3f}{r.k_ext:>10.3f}"
                  f"{r.hk:>9.3f}")
        w = sub.n / sub.n.sum()           # pooled over questions, not methods
        print(f"{'pooled':<11}{sub.n.sum():>6}{(w*sub.k_int).sum():>10.3f}"
              f"{(w*sub.k_ext).sum():>10.3f}{(w*sub.hk).sum():>9.3f}")

    print("\nSpread across methods (max - min), by subset:")
    for s in SUBSETS:
        sub = df[df.subset == s]
        print(f"  {s:<11} K_int {sub.k_int.max()-sub.k_int.min():.3f}   "
              f"K_ext {sub.k_ext.max()-sub.k_ext.min():.3f}   "
              f"HK {sub.hk.max()-sub.hk.min():.3f}")
    print(f"\nwrote {AV / 'subset_means_ck8.csv'}")


if __name__ == "__main__":
    main()
