#!/usr/bin/env python3
"""Section 4.4: Table 1 and where the Suppressed Knowledge gap comes from.

Each question is put in one of four states at the final checkpoint, by whether
K_int and K_ext are above 1/2:

    retained   K_int > 1/2, K_ext > 1/2      forgotten  K_int <= 1/2, K_ext <= 1/2
    suppressed K_int > 1/2, K_ext <= 1/2     lucky      K_int <= 1/2, K_ext > 1/2

Prints Table 1 (% of the 1,273 questions in each state, per method) and the
signed contribution of each state to the mean gap K_int - K_ext:

    contribution(s) = share(s) * mean gap within s,

which sum over the four states to the Suppressed Knowledge of Section 4.2.

Usage: python plots/states.py
"""
import pandas as pd

from common import K_CHANCE, METHODS, label, load_k

STATES = ["retained", "suppressed", "forgotten", "lucky"]


def assign_states(d: pd.DataFrame) -> pd.Series:
    hi_int, hi_ext = d.k_internal > K_CHANCE, d.k_external > K_CHANCE
    s = pd.Series("lucky", index=d.index)
    s[hi_int & hi_ext] = "retained"
    s[hi_int & ~hi_ext] = "suppressed"
    s[~hi_int & ~hi_ext] = "forgotten"
    return s


def main():
    share, contrib = {}, {}
    for m in METHODS:
        d = load_k(f"{m}_ck8")
        d = d.assign(state=assign_states(d), gap=d.k_internal - d.k_external)
        n = len(d)
        share[label(m)] = {s: 100 * (d.state == s).sum() / n for s in STATES}
        # share * mean gap within the state == sum of the state's gaps / n
        contrib[label(m)] = {s: 100 * d.gap[d.state == s].sum() / n for s in STATES}
    share, contrib = pd.DataFrame(share), pd.DataFrame(contrib)

    print("Table 1: % of questions in each state at the final checkpoint")
    print(share.round(1).to_string())
    above = share.loc["retained"] + share.loc["suppressed"]
    print(f"\nsuppressed            {share.loc['suppressed'].min():.1f}-"
          f"{share.loc['suppressed'].max():.1f}%")
    print(f"K_int above chance    {above.min():.1f}-{above.max():.1f}%   "
          "(retained + suppressed)")
    print(f"forgotten             {share.loc['forgotten'].min():.1f}-"
          f"{share.loc['forgotten'].max():.1f}%")

    print("\nContribution of each state to the mean gap (pp)")
    print(contrib.round(1).to_string())
    total = contrib.sum()
    print(f"\nsuppressed            {contrib.loc['suppressed'].min():+.1f} to "
          f"{contrib.loc['suppressed'].max():+.1f} pp")
    print(f"lucky                 {contrib.loc['lucky'].max():+.1f} to "
          f"{contrib.loc['lucky'].min():+.1f} pp")
    rf = contrib.loc["retained"] + contrib.loc["forgotten"]
    print(f"retained + forgotten  {rf.min():+.1f} to {rf.max():+.1f} pp")
    print(f"total (the gap)       {total.min():.1f} to {total.max():.1f} pp")
    print(f"suppressed / total    {(contrib.loc['suppressed'] / total).mean():.2f}x "
          "on average across methods")


if __name__ == "__main__":
    main()
