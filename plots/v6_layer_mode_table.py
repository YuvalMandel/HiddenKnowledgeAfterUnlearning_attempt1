#!/usr/bin/env python3
"""Compare every injection-layer rule at the pre-registered alpha=1.

Rules, in order of how much per-method selection each involves:
  fixed  {3,6,9,12,15}   5 sites, early, identical for every method  (headline)
  late   {18,21,24,27,30} 5 sites, mirrored into the second half
  all    1..32           32 sites, no selection at all
  topk5  5 sites picked per method from its base-model AUC profile
  band5  5 consecutive sites centred on that profile's best layer

The column that matters is ADVANTAGE = steered - matched-norm random control.
A rule that lifts the control as much as the steered vector has demonstrated
nothing about the direction, however high its steered K_ext looks.

Outputs: prints the table; writes plots/v6_layer_mode_table.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

VEC = Path(__file__).resolve().parent / "activation_vectors"
SAVE = Path(__file__).resolve().parent
METHODS = ["RepNoise", "GradDiff", "PB_J", "TAR", "RR", "ELM", "RMU", "RMU-LAT"]
MODES = [("fixed", "", 5), ("late", "_late", 5), ("all", "_all", 32),
         ("topk5", "_topk5", 5), ("band5", "_band5", 5), ("allnorm", "_allnorm", 32), ("latenorm", "_latenorm", 5), ("bottomk5", "_bottomk5", 5)]


def val(method, suffix, cond, alpha=1.0):
    p = VEC / f"causal_recover_{method}{suffix}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    r = d[(d.condition == cond) & (np.isclose(d.alpha, alpha))]
    return float(r.kext.iloc[0]) if len(r) else None


rows = []
for m in METHODS:
    for name, suf, sites in MODES:
        s = val(m, suf, "supp_dS")
        r = val(m, suf, "supp_random")
        if s is None or r is None:
            continue
        rows.append(dict(method=m, mode=name, sites=sites, steered=s,
                         random=r, advantage=s - r))
df = pd.DataFrame(rows)
df.to_csv(SAVE / "v6_layer_mode_table.csv", index=False)

order = [n for n, _, _ in MODES]
print("Steered K_ext / matched-norm random / ADVANTAGE, at alpha=1\n")
w = 17
print(f"{'method':<10}" + "".join(f"{n:>{w}}" for n in order))
print("-" * (10 + w * len(order)))
for m in METHODS:
    line = f"{m:<10}"
    for n in order:
        g = df[(df.method == m) & (df["mode"] == n)]
        if g.empty:
            line += f"{'--':>{w}}"
        else:
            g = g.iloc[0]
            line += f"{g.steered:>6.3f}/{g['random']:.3f}{g.advantage:>+7.3f}"
    print(line)

print("-" * (10 + w * len(order)))
line = f"{'MEAN adv':<10}"
for n in order:
    g = df[df["mode"] == n]
    line += f"{'':>10}{g.advantage.mean():>+7.3f}" if len(g) else f"{'--':>{w}}"
print(line)
line = f"{'n methods':<10}"
for n in order:
    line += f"{'':>10}{len(df[df['mode']==n]):>7d}"
print(line)

print("\nPer-mode summary (only methods measured under that mode):")
for n in order:
    g = df[df["mode"] == n]
    if g.empty:
        print(f"  {n:<7} not yet run")
        continue
    print(f"  {n:<7} sites={g.sites.iloc[0]:<3} mean steered {g.steered.mean():.3f}"
          f"   mean random {g['random'].mean():.3f}"
          f"   mean advantage {g.advantage.mean():+.3f}"
          f"   ({len(g)} methods)")

common = (df.pivot_table(index="method", columns="mode", values="advantage")
          .dropna())
if len(common) > 1:
    print(f"\nRestricted to the {len(common)} methods measured under ALL modes "
          f"({', '.join(common.index)}):")
    for n in order:
        if n in common:
            print(f"  {n:<7} mean advantage {common[n].mean():+.3f}")
