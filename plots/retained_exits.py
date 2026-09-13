"""Where do base-retained questions go, and is that consistent with 5.1?

5.2 reports two different things and they are easy to confuse:

  NET   change in cell sizes. retained 941 -> 587 is -354; suppressed, lucky and
        forgotten grow by 225/105/23, i.e. 64/30/6 percent of that 354. These
        are the paper's published figures and they are correct.
  GROSS individual questions leaving retained: 439 on average, splitting
        55/26/18. Larger than the net because questions also arrive.

tab:transitions counts gross, so it does not contradict the net prose. This
script computes the gross side; the net side is a difference of cell counts.
"""
import pandas as pd
from pathlib import Path

OUT = Path("inside_out_out")
M = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]


def load(mid):
    df = pd.read_parquet(OUT / mid / "k_scores.parquet")
    cv = df[(df.split_type == "cv") & (df.domain == "bio") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    return cv.groupby("question_idx").agg(ki=("k_internal", "mean"),
                                          ke=("k_external", "mean"))


def cell(ki, ke):
    if ki > 0.5:
        return "retained" if ke > 0.5 else "suppressed"
    return "lucky" if ke > 0.5 else "forgotten"


b = load("base")
base_ret = b[(b.ki > 0.5) & (b.ke > 0.5)].index
print(f"base retained: {len(base_ret)} questions\n")
print(f"{'method':<10}{'left':>6}{'supp':>7}{'lucky':>7}{'forg':>6}"
      f"{'lucky share of empty':>22}")
print("-" * 60)
rows = []
for m in M:
    d = load(f"{m}_ck8")
    cells = [cell(d.ki[q], d.ke[q]) for q in base_ret if q in d.index]
    left = [c for c in cells if c != "retained"]
    s = sum(c == "suppressed" for c in left)
    l = sum(c == "lucky" for c in left)
    f = sum(c == "forgotten" for c in left)
    share = l / (l + f) if (l + f) else float("nan")
    rows.append((len(left), s, l, f, share))
    print(f"{m:<10}{len(left):>6}{100*s/len(left):>6.0f}%{100*l/len(left):>6.0f}%"
          f"{100*f/len(left):>5.0f}%{100*share:>21.1f}%")

n = sum(r[0] for r in rows) / len(rows)
S = sum(r[1] for r in rows) / sum(r[0] for r in rows)
L = sum(r[2] for r in rows) / sum(r[0] for r in rows)
F = sum(r[3] for r in rows) / sum(r[0] for r in rows)
tot_l = sum(r[2] for r in rows)
tot_f = sum(r[3] for r in rows)
print(f"\nmean exits {n:.0f}: {100*S:.0f}% suppressed, {100*L:.0f}% lucky, "
      f"{100*F:.0f}% forgotten")
print(f"lucky share of the probe-empty exits: {100*tot_l/(tot_l+tot_f):.1f}%"
      f"   (5.1 reports 51.6% over ALL probe-empty questions)")
