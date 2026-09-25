#!/usr/bin/env python3
"""Sections 4.5 and 4.6 (Figures 5 and 6): WMDP's RMU checkpoints on three architectures.

For Zephyr-7B, Mixtral-8x7B and Yi-34B, each RMU model against its own base:

  --domain bio    Section 4.5, Figure 5
  --domain cyber  Section 4.6, Figure 6

Prints the raw scores (published WMDP accuracy, K_ext, K_int, with the fold
standard deviation of K), each metric's share of its own base model's
above-chance signal retained (floored at 0, as in the figures), the gap
K_int - K_ext, and the share of questions in each state for both the base and
the RMU model.

Usage: python numbers/architectures.py --domain bio
       python numbers/architectures.py --domain cyber
"""
import argparse

import pandas as pd

from common import (K_CHANCE, WMDP_CHANCE, k_summary, published_accuracy,
                    retention, state_shares)

FAMILIES = [("Zephyr-7B", "zephyr"), ("Mixtral-8x7B", "mixtral"), ("Yi-34B", "yi")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["bio", "cyber"], default="bio")
    domain = ap.parse_args().domain

    acc = published_accuracy("wmdp", domain)
    rows = []
    for name, fam in FAMILIES:
        base_id, rmu_id = f"{fam}_base", f"{fam}_rmu"
        b, r = k_summary(base_id, domain), k_summary(rmu_id, domain)
        rows.append(dict(
            model=name, base_id=base_id, rmu_id=rmu_id,
            wmdp=acc[rmu_id], k_ext=r["k_ext"], k_int=r["k_int"],
            k_ext_sd=r["k_ext_sd"], k_int_sd=r["k_int_sd"],
            r_wmdp=max(0.0, retention(acc[rmu_id], WMDP_CHANCE, acc[base_id])),
            r_ext=max(0.0, retention(r["k_ext"], K_CHANCE, b["k_ext"])),
            r_int=max(0.0, retention(r["k_int"], K_CHANCE, b["k_int"])),
            gap_pp=100 * (r["k_int"] - r["k_ext"])))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(d.drop(columns=["base_id", "rmu_id"]).round(3).to_string(index=False))

    print("\n% of questions in each state (base -> RMU)")
    for r in d.itertuples():
        b, u = state_shares(r.base_id, domain), state_shares(r.rmu_id, domain)
        print(f"  {r.model:<13}" + "  ".join(
            f"{s} {b[s]:.1f}->{u[s]:.1f}" for s in b))


if __name__ == "__main__":
    main()
