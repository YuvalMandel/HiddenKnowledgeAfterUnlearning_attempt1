#!/usr/bin/env python3
"""Appendix D (Figures 8 and 9): the nine OPTML unlearning models on Zephyr-7B-beta.

Prints the raw scores (published WMDP-Bio accuracy, K_ext, K_int), each
metric's share of the base model's above-chance signal retained (Figure 9
floors it at 0, here it is unfloored), and the numbers quoted in Appendix D:
the ranges of K_int and K_ext, their retention, the gap, and the suppressed and
forgotten shares against the base model's own.

Usage: python numbers/optml.py
"""
import pandas as pd

from common import (K_CHANCE, WMDP_CHANCE, k_summary, published_accuracy,
                    retention, state_shares)

BASE = "zephyr_base"
MODELS = [("zephyr_simnpo", "SimNPO"), ("zephyr_npo", "NPO"),
          ("zephyr_npo_sam", "NPO+SAM"), ("zephyr_npo_gp", "NPO+GP"),
          ("zephyr_npo_cr", "NPO+CR"), ("zephyr_npo_rs", "NPO+RS"),
          ("zephyr_npo_wa", "NPO+WA"), ("zephyr_graddiff", "GradDiff"),
          ("zephyr_graddiff_sam", "GradDiff+SAM")]


def main():
    acc = published_accuracy("optml")
    b = k_summary(BASE)
    rows = []
    for mid, lab in [(BASE, "Base")] + MODELS:
        k = k_summary(mid)
        rows.append(dict(
            model=lab, model_id=mid, wmdp=acc[mid], k_ext=k["k_ext"], k_int=k["k_int"],
            k_ext_sd=k["k_ext_sd"], k_int_sd=k["k_int_sd"],
            r_wmdp=retention(acc[mid], WMDP_CHANCE, acc[BASE]),
            r_ext=retention(k["k_ext"], K_CHANCE, b["k_ext"]),
            r_int=retention(k["k_int"], K_CHANCE, b["k_int"])))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(d.drop(columns="model_id").round(3).to_string(index=False))

    m = d.iloc[1:]
    shares = {mid: state_shares(mid) for mid in d.model_id}
    sup = [shares[mid]["suppressed"] for mid in m.model_id]
    forg = [shares[mid]["forgotten"] for mid in m.model_id]
    gap = 100 * (m.k_int - m.k_ext)
    print(f"\nK_int {m.k_int.min():.3f}-{m.k_int.max():.3f} (base {d.k_int[0]:.3f}); "
          f"K_ext {m.k_ext.min():.3f}-{m.k_ext.max():.3f} (base {d.k_ext[0]:.3f})")
    print(f"K_int retains {m.r_int.min():.0f}-{m.r_int.max():.0f}%; "
          f"K_ext {m.r_ext.min():.0f}-{m.r_ext.max():.0f}%")
    print(f"gap {gap.min():.1f}-{gap.max():.1f} pp")
    print(f"suppressed {min(sup):.1f}-{max(sup):.1f}%; forgotten "
          f"{min(forg):.1f}-{max(forg):.1f}% (base {shares[BASE]['forgotten']:.1f}%)")


if __name__ == "__main__":
    main()
