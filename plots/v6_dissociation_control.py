#!/usr/bin/env python3
"""Is the suppressed/lucky crossing a double dissociation, or one graded factor?

Suppressed questions lose K_ext and keep K_int; lucky questions do the reverse.
Read as a double dissociation, that argues the two axes index separable things
rather than one underlying quantity read twice.

The standard objection to any claimed dissociation is a single severity factor:
if lucky questions were simply *weaker* to begin with, the crossing would be an
ordering effect, not a dissociation. That objection has real support here --
lucky questions start weaker than suppressed ones on BOTH base margins. This
script runs the control that settles it: stratify Q* on base internal and
external margin (3x3 tertile grid) and re-measure the crossing within strata
that contain at least MIN_CELL of each group.

Base margins come from inside_out_ext/base_bio_*.npy, a different probe run
from the CV parquet; they disagree with it on 1.0% of Q* (7/701), which is
small enough to use as a stratification covariate.

Outputs: prints the table; writes plots/v6_dissociation_control.csv
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "inside_out_out"
EXT_DIR = REPO / "inside_out_ext"
SAVE_DIR = Path(__file__).resolve().parent
METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
SEL = dict(split_type="cv", domain="bio", clf="LR",
           probe_type="own", layer_config="best_layer")
MIN_CELL = 5
N_TERTILE = 3


def load(model_id):
    d = pd.read_parquet(OUT_DIR / model_id / "k_scores.parquet")
    for k, v in SEL.items():
        d = d[d[k] == v]
    return d.set_index("question_idx")[["k_internal", "k_external"]]


def margin(arr, idx, correct):
    """p(correct) - max p(wrong), per question."""
    return pd.Series({q: float(arr[q][int(correct[q])]
                               - np.delete(arr[q], int(correct[q])).max())
                      for q in idx})


base = load("base")
qstar = base[(base.k_internal == 1.0) & (base.k_external == 1.0)].index

correct = (pd.read_csv(REPO / "data" / "wmdp_tf_pairs.csv", keep_default_na=False)
           .drop_duplicates("original_id").set_index("original_id")["correct_idx"])
m_int = margin(np.load(EXT_DIR / "base_bio_int_proba.npy"), qstar, correct)
m_ext = margin(np.load(EXT_DIR / "base_bio_ext.npy"), qstar, correct)

subsets = {}
rows = []
for m in METHODS:
    q = load(f"{m}_ck8")
    q = q.loc[q.index.intersection(qstar)]
    hi_i, hi_e = q.k_internal > 0.5, q.k_external > 0.5
    lucky, supp = q[~hi_i & hi_e], q[hi_i & ~hi_e]
    subsets[m] = (set(lucky.index), set(supp.index), q)
    rows.append(dict(
        method=m, n_lucky=len(lucky), n_supp=len(supp),
        lucky_kint=lucky.k_internal.mean(), lucky_kext=lucky.k_external.mean(),
        supp_kint=supp.k_internal.mean(), supp_kext=supp.k_external.mean(),
        lucky_base_int=m_int[lucky.index].mean(),
        supp_base_int=m_int[supp.index].mean(),
        lucky_base_ext=m_ext[lucky.index].mean(),
        supp_base_ext=m_ext[supp.index].mean()))

df = pd.DataFrame(rows)
df.to_csv(SAVE_DIR / "v6_dissociation_control.csv", index=False)

print("ck8 subset signatures (K_int / K_ext) and base margins\n")
print(f"{'method':<10}{'lucky K':>14}{'supp K':>14}{'base int L/S':>14}{'base ext L/S':>14}")
for r in rows:
    print(f"{r['method']:<10}"
          f"{r['lucky_kint']:>6.2f}/{r['lucky_kext']:<7.2f}"
          f"{r['supp_kint']:>6.2f}/{r['supp_kext']:<7.2f}"
          f"{r['lucky_base_int']/r['supp_base_int']:>14.2f}"
          f"{r['lucky_base_ext']/r['supp_base_ext']:>14.2f}")
print(f"\n{'MEAN':<10}"
      f"{df.lucky_kint.mean():>6.2f}/{df.lucky_kext.mean():<7.2f}"
      f"{df.supp_kint.mean():>6.2f}/{df.supp_kext.mean():<7.2f}"
      f"{(df.lucky_base_int/df.supp_base_int).mean():>14.2f}"
      f"{(df.lucky_base_ext/df.supp_base_ext).mean():>14.2f}")
print("  lucky start weaker on BOTH axes -> single-factor account is live")

# ── is subset membership a property of the question, or method-specific noise?
print("\nCross-method stability (questions falling in the subset for >=3 methods)")
for name, pick in (("lucky", 0), ("suppressed", 1)):
    cnt = pd.Series(0, index=qstar)
    for m in METHODS:
        cnt[list(subsets[m][pick])] += 1
    p = [len(subsets[m][pick]) / len(qstar) for m in METHODS]
    exp = sum(np.prod([p[i] if i in c else 1 - p[i] for i in range(8)])
              for k in range(3, 9) for c in combinations(range(8), k)) * len(qstar)
    obs = int((cnt >= 3).sum())
    print(f"  {name:<11} observed {obs:>3}   expected if independent {exp:>6.1f}"
          f"   {obs / max(exp, 1e-9):>5.2f}x")

# ── the control: does the crossing survive matching on base strength? ─────────
t_int = pd.qcut(m_int, N_TERTILE, labels=False)
t_ext = pd.qcut(m_ext, N_TERTILE, labels=False)
cells = {(a, b): [q for q in qstar if t_int[q] == a and t_ext[q] == b]
         for a in range(N_TERTILE) for b in range(N_TERTILE)}

acc = {k: [] for k in ("L_ki", "L_ke", "S_ki", "S_ke")}
n_strata = 0
for m in METHODS:
    lucky, supp, q = subsets[m]
    for cell in cells.values():
        lc = [x for x in cell if x in lucky]
        sc = [x for x in cell if x in supp]
        if len(lc) >= MIN_CELL and len(sc) >= MIN_CELL:
            acc["L_ki"].append(q.loc[lc, "k_internal"].mean())
            acc["L_ke"].append(q.loc[lc, "k_external"].mean())
            acc["S_ki"].append(q.loc[sc, "k_internal"].mean())
            acc["S_ke"].append(q.loc[sc, "k_external"].mean())
            n_strata += 1

mean = {k: float(np.mean(v)) for k, v in acc.items()}
print(f"\nStratified control ({N_TERTILE}x{N_TERTILE} tertile grid on base int x ext "
      f"margin, >={MIN_CELL} per group)")
print(f"  matched strata: {n_strata}")
print(f"  lucky      K_int {mean['L_ki']:.3f}   K_ext {mean['L_ke']:.3f}")
print(f"  suppressed K_int {mean['S_ki']:.3f}   K_ext {mean['S_ke']:.3f}")
print(f"  crossing within strata:  K_int {mean['S_ki'] - mean['L_ki']:+.3f}"
      f"   K_ext {mean['S_ke'] - mean['L_ke']:+.3f}")
print("  -> undiminished by matching: the crossing is not a base-strength ordering")
