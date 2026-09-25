#!/usr/bin/env python3
"""Section 4.2 (Figures 2 and 3): the eight methods at their final checkpoint.

Prints, for the base model and each method:
  - WMDP-Bio accuracy (published), K_ext and K_int, with the standard deviation
    of K across the five folds (Figure 2), and the gap K_int - K_ext;
  - each metric's share of the base model's above-chance signal retained,
    R(M) = 100 (M - chance) / (base - chance), chance 0.25 for WMDP accuracy
    and 0.5 for K (Figure 3; the figure floors R at 0, here it is unfloored);
  - the mean R over the eight methods, and the difference between the mean R
    of K_int and of WMDP-Bio (the 60.8 points of the abstract, the
    introduction and Figure 1).

Usage: python numbers/final_checkpoint.py
"""
import pandas as pd

from common import (K_CHANCE, METHODS, WMDP_CHANCE, k_summary, label,
                    published_accuracy, retention)


def main():
    acc = published_accuracy("llm-gat")
    base = k_summary("base")
    rows = []
    for mid in ["base"] + [f"{m}_ck8" for m in METHODS]:
        k = k_summary(mid)
        rows.append(dict(
            method="Base" if mid == "base" else label(mid[:-4]),
            wmdp=acc[mid], k_ext=k["k_ext"], k_int=k["k_int"],
            k_ext_sd=k["k_ext_sd"], k_int_sd=k["k_int_sd"],
            gap_pp=100 * (k["k_int"] - k["k_ext"]),
            r_wmdp=retention(acc[mid], WMDP_CHANCE, acc["base"]),
            r_ext=retention(k["k_ext"], K_CHANCE, base["k_ext"]),
            r_int=retention(k["k_int"], K_CHANCE, base["k_int"])))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(d.round(3).to_string(index=False))

    m = d.iloc[1:]
    print(f"\nK_ext  base {d.k_ext[0]:.3f} -> {m.k_ext.min():.3f}-{m.k_ext.max():.3f}")
    print(f"K_int  base {d.k_int[0]:.3f} -> {m.k_int.min():.3f}-{m.k_int.max():.3f}")
    print(f"gap    {m.gap_pp.min():.1f}-{m.gap_pp.max():.1f} pp")
    print(f"mean R over the eight methods: WMDP-Bio {m.r_wmdp.mean():.1f}%, "
          f"K_ext {m.r_ext.mean():.1f}%, K_int {m.r_int.mean():.1f}%")
    print(f"K_int keeps {m.r_int.mean() - m.r_wmdp.mean():.1f} points more of its "
          "above-chance signal than WMDP-Bio accuracy, on the mean")


if __name__ == "__main__":
    main()
