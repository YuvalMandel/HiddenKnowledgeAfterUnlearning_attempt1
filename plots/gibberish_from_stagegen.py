#!/usr/bin/env python3
"""Gibberish straight from stage_gen's stored generations -- no GPU needed.

`inside_out_knowledge.stage_gen` already wrote every generation to
`inside_out_out/<model>/bio_gen_test.json`, for all 65 checkpoints, at float16
and with prompts byte-identical to `make_tf_prompt`. Each file holds 2,292
records (573 test questions x 4 options), which is exactly the denominator
`app:gibberish` reports -- so this is the source the published table was built
from, and recomputing from it needs no model and no GPU.

Two things fall out for free: an exact check on the published ck8 numbers, and
the same rate for ck1-ck7, which is the per-checkpoint generation data the
WMDP-trajectory question needed.

Usage: python plots/gibberish_from_stagegen.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "inside_out_out"
AV = ROOT / "plots" / "activation_vectors"
TF = re.compile(r"\b(True|False)\b", re.IGNORECASE)
METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
PUBLISHED = {"base": 2, "GradDiff": 2285, "RMU": 341, "RMU-LAT": 349,
             "RepNoise": 2135, "ELM": 0, "RR": 814, "TAR": 2, "PB_J": 545}


def rate(model_dir):
    f = OUT_DIR / model_dir / "bio_gen_test.json"
    if not f.exists():
        return None
    recs = json.loads(f.read_text(encoding="utf-8"))
    bad = sum(1 for r in recs if not TF.search(r["text"]))
    return len(recs), bad


def main():
    rows = []
    for m in ["base"] + [f"{x}_ck{c}" for x in METHODS for c in range(1, 9)]:
        r = rate(m)
        if r is None:
            continue
        n, bad = r
        name = m.replace("_ck", "|").split("|")
        rows.append(dict(model=m, method=name[0],
                         ck=int(name[1]) if len(name) > 1 else 0,
                         n=n, gibberish=bad, rate=round(100 * bad / n, 2)))
    df = pd.DataFrame(rows)
    df.to_csv(AV / "gibberish_stagegen_all.csv", index=False)

    print("ck8 against the published table (same 2,292 prompts, same fp16 source)\n")
    print(f"{'method':<10}{'published':>10}{'recomputed':>12}{'delta':>7}{'rate %':>9}")
    print("-" * 48)
    ck8 = df[(df.ck == 8) | (df.method == "base")]
    for _, r in ck8.iterrows():
        p = PUBLISHED.get(r.method)
        d = "" if p is None else f"{r.gibberish - p:+d}"
        print(f"{r.method:<10}{p if p is not None else '-':>10}"
              f"{r.gibberish:>12}{d:>7}{r.rate:>9.2f}")
    exact = sum(1 for _, r in ck8.iterrows()
                if PUBLISHED.get(r.method) == r.gibberish)
    print(f"\nexact matches: {exact}/{len(ck8)}")

    print("\ngibberish rate by checkpoint (%)\n")
    piv = df[df.ck > 0].pivot(index="method", columns="ck", values="rate")
    print(piv.to_string())
    print(f"\nwrote {AV/'gibberish_stagegen_all.csv'}")


if __name__ == "__main__":
    main()
