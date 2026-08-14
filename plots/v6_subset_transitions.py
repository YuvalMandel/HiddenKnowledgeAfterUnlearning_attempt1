#!/usr/bin/env python3
"""4x4 subset transition matrix: base (ck0) -> ck8, intermediate checkpoints ignored.

Rows are the subset a question belongs to at the base model, columns the subset
it belongs to at ck8. Cells count questions making that transition, so the
diagonal is questions whose classification is unchanged end-to-end.

Runs on ALL questions by default. Under the Q* pre-filter every question starts
as retained by construction, which collapses the matrix to a single populated
row -- pass --qstar if that degenerate view is wanted anyway.

Usage:
    python plots/v6_subset_transitions.py [METHOD] [--qstar]

Outputs: plots/v6_subset_transitions{,_qstar}.csv
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "inside_out_out"
SAVE_DIR = Path(__file__).resolve().parent
QSTAR_ONLY = "--qstar" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
METHOD = args[0] if args else "RepNoise"
KNOWS = 0.5
ORDER = ["Retained", "Suppressed", "Forgotten", "Lucky"]
SEL = dict(split_type="cv", domain="bio", clf="LR",
           probe_type="own", layer_config="best_layer")


def load(model_id):
    d = pd.read_parquet(OUT_DIR / model_id / "k_scores.parquet")
    for k, v in SEL.items():
        d = d[d[k] == v]
    return d.set_index("question_idx")[["k_internal", "k_external"]]


def classify(df):
    hi_i, hi_e = df.k_internal > KNOWS, df.k_external > KNOWS
    out = pd.Series("Forgotten", index=df.index)
    out[hi_i & hi_e] = "Retained"
    out[hi_i & ~hi_e] = "Suppressed"
    out[~hi_i & hi_e] = "Lucky"
    return out


base = load("base")
ck8 = load(f"{METHOD}_ck8")
pool = base.index
if QSTAR_ONLY:
    pool = base[(base.k_internal == 1.0) & (base.k_external == 1.0)].index
pool = pool.intersection(ck8.index)

a = classify(base.loc[pool])
b = classify(ck8.loc[pool])

mat = (pd.crosstab(a, b)
       .reindex(index=ORDER, columns=ORDER, fill_value=0))
mat.index.name = "base (ck0)"
mat.columns.name = "ck8"

stem = SAVE_DIR / ("v6_subset_transitions_qstar" if QSTAR_ONLY
                   else "v6_subset_transitions")
mat.to_csv(f"{stem}.csv")

scope = "Q*" if QSTAR_ONLY else "all questions"
print(f"{METHOD}: base (ck0) -> ck8 transitions, {scope}, n={len(pool)}\n")
show = mat.copy()
show["TOTAL"] = show.sum(axis=1)
show.loc["TOTAL"] = show.sum(axis=0)
print(show.to_string())

stayed = sum(mat.loc[s, s] for s in ORDER)
# a diagonal is only meaningful against what independent margins would give
exp_diag = sum(mat.loc[s].sum() * mat[s].sum() for s in ORDER) / len(pool)
print(f"\nunchanged (diagonal): {stayed}  ({stayed / len(pool):.1%})")
print(f"changed              : {len(pool) - stayed}  "
      f"({1 - stayed / len(pool):.1%})")
print(f"expected diagonal if base and ck8 labels were independent: "
      f"{exp_diag:.1f} ({exp_diag / len(pool):.1%})  -> observed/expected "
      f"{stayed / exp_diag:.2f}x")
print(f"\nrow % (where each base subset ends up)")
print((100 * mat.div(mat.sum(axis=1).replace(0, pd.NA), axis=0))
      .round(1).fillna(0).to_string())
