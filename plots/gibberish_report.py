#!/usr/bin/env python3
"""Collect the float32 gibberish runs and set them beside the published table.

Three columns matter:
  published   what tab:gibberish reports, on 573 questions x 4 options = 2,292
  reproduced  this run restricted to those same 2,292 prompts -- the gate
  full        this run over all 1,273 questions x 4 options = 5,092

Everything here is float32, the dtype the LLM-GAT checkpoints are released in.
That matters: bfloat16 costs three mantissa bits and moved RepNoise by 4.6 pp
(88.2 -> 92.8), while float16 and float32 agree to 0.1 pp and batch size moves
nothing (<=1 prompt in 2,292). Use float32 where it fits, float16 otherwise;
never bfloat16 for this measurement.

After that correction five of nine reproduce the published value to within
0.5 pp. RR (-3.1), PB&J (-2.0) and RMU-LAT (-1.5) do not, and dtype is not the
cause -- their rates barely moved between bfloat16 and float32. That residual is
unexplained and should not be papered over.

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
        f = AV / f"gibberish_full_{m}_quad_f32.json"   # float32 = the released dtype
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
