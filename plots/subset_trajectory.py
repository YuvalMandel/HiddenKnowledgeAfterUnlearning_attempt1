#!/usr/bin/env python3
"""Subset counts across ck1-ck8, with lucky and forgotten tracked together.

5.1 shows that within the probe-empty population (K_int <= 1/2) the split
between lucky and forgotten is at chance, so the two are one population -- the
questions the probe can no longer read -- and the coherent trajectory to report
is their union, not forgotten alone.

Counts are means over the eight methods, on all 1,273 questions, from the same
query that reproduces tab:subsets exactly.

Usage: python plots/subset_trajectory.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "inside_out_out"
AV = ROOT / "plots" / "activation_vectors"
METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]


def load(model_id):
    df = pd.read_parquet(OUT / model_id / "k_scores.parquet")
    cv = df[(df.split_type == "cv") & (df.domain == "bio") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    return cv.groupby("question_idx").agg(ki=("k_internal", "mean"),
                                          ke=("k_external", "mean"))


def counts(d):
    return dict(retained=int(((d.ki > .5) & (d.ke > .5)).sum()),
                suppressed=int(((d.ki > .5) & (d.ke <= .5)).sum()),
                forgotten=int(((d.ki <= .5) & (d.ke <= .5)).sum()),
                lucky=int(((d.ki <= .5) & (d.ke > .5)).sum()))


def main():
    rows = [dict(ck=0, **counts(load("base")))]
    for ck in range(1, 9):
        per = [counts(load(f"{m}_ck{ck}")) for m in METHODS]
        rows.append(dict(ck=ck, **{k: float(np.mean([p[k] for p in per]))
                                   for k in per[0]}))
    df = pd.DataFrame(rows)
    df["probe_empty"] = df.forgotten + df.lucky
    df.to_csv(AV / "subset_trajectory.csv", index=False)

    print(f"{'ck':>3}{'retained':>10}{'suppressed':>12}{'forgotten':>11}"
          f"{'lucky':>8}{'forg+lucky':>12}")
    print("-" * 56)
    for _, r in df.iterrows():
        tag = "base" if r.ck == 0 else f"ck{int(r.ck)}"
        print(f"{tag:>3}{r.retained:>10.0f}{r.suppressed:>12.0f}"
              f"{r.forgotten:>11.0f}{r.lucky:>8.0f}{r.probe_empty:>12.0f}")

    pe = df.probe_empty.values
    sup = df.suppressed.values
    print(f"\nprobe-empty (forgotten+lucky): {pe[0]:.0f} at base -> {pe[1]:.0f} at "
          f"ck1 -> {pe[-1]:.0f} at ck8")
    print(f"  ck1 accounts for {100*(pe[1]-pe[0])/(pe[-1]-pe[0]):.0f}% of the "
          f"total rise; range over ck1-ck8: {pe[1:].min():.0f}-{pe[1:].max():.0f}")
    print(f"  monotone over ck1-ck8: {bool(np.all(np.diff(pe[1:]) >= 0))}")
    print(f"suppressed: {sup[0]:.0f} -> peak {sup[1:].max():.0f} at "
          f"ck{int(df.ck[1:][sup[1:].argmax()+1])}, ck8 {sup[-1]:.0f}; "
          f"monotone ck1-ck7: {bool(np.all(np.diff(sup[1:8]) >= 0))}")


if __name__ == "__main__":
    main()
