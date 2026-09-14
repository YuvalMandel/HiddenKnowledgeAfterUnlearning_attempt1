#!/usr/bin/env python3
"""Figure 3 split into one panel per unlearning method.

v6_subset_flow_mean.py collapses the eight methods into a mean line with a
min-max band. That hides which method does what: the band at ck8 spans 481-716
retained, and a reader cannot tell whether the spread comes from one outlier or
from a real split in behaviour. These panels show the eight trajectories
separately, on shared axes so they can be compared directly.

Same data and same classification rule as Figure 3: every question is
re-classified at each checkpoint from that checkpoint's own K_int and K_ext, all
1,273 questions, no pre-filter, so ck0 is not all-retained.

Usage: python plots/v6_subset_flow_panels.py

Outputs: plots/v6_subset_flow_panels.pdf/.png/.csv
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hk_utils import iclr_figsize, use_iclr_style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "inside_out_out"
METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
LABEL = {"PB_J": "PB&J"}
N_CK = 8

# identical to v6_subset_flow_mean.py, so the panels read as the same figure
STYLE = {
    "Retained":   ("#56B4E9", "--", "s"),
    "Suppressed": ("#D55E00", "-",  "^"),
    "Forgotten":  ("#CC79A7", "-.", "v"),
    "Lucky":      ("#009E73", ":",  "D"),
}


def load(model_id):
    df = pd.read_parquet(OUT / model_id / "k_scores.parquet")
    cv = df[(df.split_type == "cv") & (df.domain == "bio") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    return cv.groupby("question_idx").agg(ki=("k_internal", "mean"),
                                          ke=("k_external", "mean"))


def counts(d):
    return {"Retained":   int(((d.ki > .5) & (d.ke > .5)).sum()),
            "Suppressed": int(((d.ki > .5) & (d.ke <= .5)).sum()),
            "Forgotten":  int(((d.ki <= .5) & (d.ke <= .5)).sum()),
            "Lucky":      int(((d.ki <= .5) & (d.ke > .5)).sum())}


def main():
    use_iclr_style()
    xs = list(range(N_CK + 1))

    base_c = counts(load("base"))
    series, rows = {}, []
    for m in METHODS:
        per_ck = [base_c] + [counts(load(f"{m}_ck{ck}")) for ck in range(1, N_CK + 1)]
        series[m] = {s: [c[s] for c in per_ck] for s in STYLE}
        for i, c in enumerate(per_ck):
            for s, v in c.items():
                rows.append(dict(method=LABEL.get(m, m),
                                 checkpoint="base" if i == 0 else f"ck{i}",
                                 subset=s, count=v))

    stem = str(Path(__file__).with_suffix(""))
    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)

    W = float(os.environ.get("FIG_WIDTH_FRAC", "1.0"))
    w, h = iclr_figsize(aspect=0.52, width_frac=W)
    fig, axes = plt.subplots(2, 4, figsize=(w, h * 1.55), sharex=True, sharey=True)

    top = max(max(series[m][s]) for m in METHODS for s in STYLE)
    for ax, m in zip(axes.ravel(), METHODS):
        for name, (colour, ls, marker) in STYLE.items():
            ax.plot(xs, series[m][name], ls, color=colour, marker=marker,
                    markersize=2.6, linewidth=1.3, label=name, zorder=3)
        ax.set_title(LABEL.get(m, m), fontsize=8.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(-15, top * 1.08)
        ax.set_xticks(xs)
        ax.set_xticklabels(["0"] + [str(c) for c in range(1, N_CK + 1)],
                           fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

    for ax in axes[:, 0]:
        ax.set_ylabel("questions", fontsize=8)
    fig.supxlabel("unlearning checkpoint (0 = base)", fontsize=9, y=0.045)

    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), fontsize=8, columnspacing=1.6,
               handlelength=2.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {Path(stem).name}.pdf  (8 panels, all 1,273 questions)\n")

    print(f"{'method':<10}" + "".join(f"{s[:5]:>8}" for s in STYLE) + "   (at ck8)")
    print("-" * 52)
    for m in METHODS:
        print(f"{LABEL.get(m, m):<10}"
              + "".join(f"{series[m][s][-1]:>8}" for s in STYLE))
    print("\nspread at ck8 (max - min across methods):")
    for s in STYLE:
        v = [series[m][s][-1] for m in METHODS]
        print(f"  {s:<11}{min(v):>5} - {max(v):<5}  (range {max(v)-min(v)})")


if __name__ == "__main__":
    main()
