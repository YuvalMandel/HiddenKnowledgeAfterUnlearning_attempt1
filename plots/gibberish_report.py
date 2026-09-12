#!/usr/bin/env python3
"""Collect the gibberish runs and set them beside the published table.

Three columns matter:
  published   what tab:gibberish reports, on 573 questions x 4 options = 2,292
  reproduced  this run restricted to those same 2,292 prompts -- the gate
  full        this run over all 1,273 questions x 4 options = 5,092

The gate is the interesting one: base and ELM reproduce exactly, GradDiff to
0.1 pp, while the models that answer in repetition loops come out 1-5 pp lower.
Greedy decoding is not bit-reproducible across GPU type, batch size and dtype,
and a loop that runs past MAX_NEW_TOKENS in one run can emit a parseable token
in another -- which moves exactly the models whose outputs sit on that boundary.

Usage: python plots/gibberish_report.py
"""
import json
from pathlib import Path

import pandas as pd

AV = Path(__file__).resolve().parent / "activation_vectors"
ORDER = ["base", "GradDiff", "RepNoise", "RR", "PB_J", "RMU-LAT", "RMU", "TAR",
         "ELM"]
# tab:gibberish as printed in the paper: (N_gib, rate %) over 2,292 prompts
PUBLISHED = {"base": (2, 0.1), "GradDiff": (2285, 99.7), "RMU": (341, 14.9),
             "RMU-LAT": (349, 15.2), "RepNoise": (2135, 93.2), "ELM": (0, 0.0),
             "RR": (814, 35.5), "TAR": (2, 0.1), "PB_J": (545, 23.8)}


def main():
    rows = []
    for m in ORDER:
        f = AV / f"gibberish_full_{m}_quad.json"
        if not f.exists():
            print(f"  {m}: missing {f.name}")
            continue
        d = json.loads(f.read_text())
        pn, pr = PUBLISHED[m]
        full, test = d["full"], d["test_split_only"]
        rows.append(dict(
            method=m,
            pub_n=pn, pub_rate=pr,
            repro_n=test["gibberish"], repro_rate=round(100 * test["rate"], 2),
            delta_pp=round(100 * test["rate"] - pr, 2),
            full_n=full["gibberish"], full_rate=round(100 * full["rate"], 2),
            full_prompts=full["n"], questions=1273))
    df = pd.DataFrame(rows)
    df.to_csv(AV / "gibberish_summary.csv", index=False)

    print("published = tab:gibberish (2,292 prompts) | reproduced = same prompts,"
          " this run | full = all 1,273 questions (5,092 prompts)\n")
    hdr = (f"{'method':<10}{'pub N':>7}{'pub %':>8}{'repro N':>9}{'repro %':>9}"
           f"{'delta pp':>10}   ||{'full N':>8}{'full %':>8}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r.method:<10}{r.pub_n:>7}{r.pub_rate:>8.1f}{r.repro_n:>9}"
              f"{r.repro_rate:>9.2f}{r.delta_pp:>+10.2f}   ||"
              f"{r.full_n:>8}{r.full_rate:>8.2f}")
    print(f"\nmean |delta| on the gate: {df.delta_pp.abs().mean():.2f} pp; "
          f"exact matches: "
          f"{int((df.repro_n == df.pub_n).sum())}/{len(df)}")
    print(f"wrote {AV/'gibberish_summary.csv'}")


if __name__ == "__main__":
    main()
