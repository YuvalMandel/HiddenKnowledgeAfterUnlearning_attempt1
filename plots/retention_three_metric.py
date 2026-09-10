#!/usr/bin/env python3
"""The three-metric retention figure: WMDP vs K_ext vs K_int on one scale.

Table `tab:wmdp-k-comparison` says the same thing in seven columns. The single
comparable quantity across three metrics with three different chance floors and
three different base-model ceilings is the fraction of the base model's
above-chance signal that survives unlearning:

    R(M) = 100 * (M - M_chance) / (M_base - M_chance)

so every metric runs 0 (at chance) to 100 (base model) and the three become
directly comparable. Each method is one column carrying three marks; the column
reads bottom to top as "what the standard benchmark sees" -> "what the model's
own readout gives away" -> "what is actually still in there".

A grouped bar chart of the same numbers needs 24 bars and reads as texture; the
dot-and-track form keeps one column per method and makes the WMDP->K_int span
the visual quantity, which is the claim.

The numeric strip beneath prints every plotted value and all three gaps, so the
figure is self-contained. Those gaps are differences of the plotted numbers, not
new measurements -- and the third is the sum of the first two, carried only
because Table 1 reports it.

K_ext and K_int come from the same parquet query as the figure this replaced, so
figure and table cannot disagree. WMDP accuracies are LLM-GAT's published values.

Usage: python plots/retention_three_metric.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hk_utils import iclr_figsize, use_iclr_style  # noqa: E402

use_iclr_style()

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "inside_out_out"
IMGS = ROOT / "overleaf_claims" / "imgs"

MODELS = [("base", "Base"), ("GradDiff_ck8", "GradDiff"), ("PB_J_ck8", "PB&J"),
          ("RMU_ck8", "RMU"), ("RMU-LAT_ck8", "RMU-LAT"),
          ("RepNoise_ck8", "RepNoise"), ("ELM_ck8", "ELM"), ("RR_ck8", "RR"),
          ("TAR_ck8", "TAR")]
# data/LLM-GAT_summary_table.csv -- GAT's published 4-way MCQ accuracies
WMDP = {"GradDiff": 0.25, "RMU": 0.26, "RMU-LAT": 0.32, "RepNoise": 0.29,
        "ELM": 0.24, "RR": 0.26, "TAR": 0.28, "PB&J": 0.31, "Base": 0.70}

INT_C, EXT_C, WMDP_C = "#2166ac", "#d6604d", "#4d4d4d"


def load():
    frames = [pd.read_parquet(OUT_DIR / m / "k_scores.parquet")
              for m, _ in MODELS if (OUT_DIR / m / "k_scores.parquet").exists()]
    df = pd.concat(frames, ignore_index=True)
    cv = df[(df.split_type == "cv") & (df.domain == "bio") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    rows = {}
    for mid, label in MODELS:
        sub = cv[cv.model_id == mid]
        if sub.empty:
            continue
        f = sub.groupby("fold").agg(ki=("k_internal", "mean"),
                                    ke=("k_external", "mean"))
        rows[label] = dict(k_int=f.ki.mean(), k_ext=f.ke.mean(),
                           k_int_sd=f.ki.std(ddof=1), k_ext_sd=f.ke.std(ddof=1))
    return rows


def main():
    raw = load()
    base = raw["Base"]

    def R(v, chance, top):
        return 100.0 * (v - chance) / (top - chance)

    rows = []
    for _, label in MODELS:
        if label == "Base" or label not in raw:
            continue
        rows.append(dict(
            label=label,
            wmdp=R(WMDP[label], 0.25, WMDP["Base"]),
            k_ext=R(raw[label]["k_ext"], 0.5, base["k_ext"]),
            k_int=R(raw[label]["k_int"], 0.5, base["k_int"]),
            # denominator held fixed, so this is the method's fold spread and
            # not the ratio's full uncertainty
            k_ext_sd=100 * raw[label]["k_ext_sd"] / (base["k_ext"] - 0.5),
            k_int_sd=100 * raw[label]["k_int_sd"] / (base["k_int"] - 0.5)))
    d = (pd.DataFrame(rows).sort_values("k_int", ascending=False)
         .reset_index(drop=True))
    mean = d[["wmdp", "k_ext", "k_int"]].mean()
    print(d.round(1).to_string(index=False))
    print("\nmean:", mean.round(1).to_dict(),
          "  (table says 5.8 / 32.1 / 66.7)")

    # ---- methods across, retention up; numeric strip underneath
    cols = list(d.label) + ["mean"]
    vw = d.wmdp.tolist() + [mean.wmdp]
    ve = d.k_ext.tolist() + [mean.k_ext]
    vi = d.k_int.tolist() + [mean.k_int]
    se = d.k_ext_sd.tolist() + [float("nan")]
    si = d.k_int_sd.tolist() + [float("nan")]
    x = list(range(len(cols)))
    x[-1] += 0.6                                    # set the mean column apart
    LEFT = x[0] - 0.9                               # row-label gutter

    fig, (ax, tb) = plt.subplots(
        2, 1, figsize=iclr_figsize(aspect=5.4 / 7, width_frac=0.80),
        gridspec_kw=dict(height_ratios=[2.3, 1.7], hspace=0.05))

    for xi, a, c in zip(x, vw, vi):
        ax.plot([xi, xi], [a, c], color="0.90", lw=4.0, zorder=1,
                solid_capstyle="round")
    ax.errorbar(x, ve, yerr=se, fmt="none", ecolor=EXT_C, elinewidth=1.0,
                capsize=0, alpha=0.9, zorder=2)
    ax.errorbar(x, vi, yerr=si, fmt="none", ecolor=INT_C, elinewidth=1.0,
                capsize=0, alpha=0.9, zorder=2)
    ax.scatter(x, vw, s=24, marker="o", facecolors="white", edgecolors=WMDP_C,
               linewidths=1.1, zorder=3, label="WMDP-Bio accuracy")
    ax.scatter(x, ve, s=24, marker="D", color=EXT_C, zorder=3,
               label=r"$K_\mathrm{ext}$ (logit margin)")
    ax.scatter(x, vi, s=32, marker="o", color=INT_C, zorder=3,
               label=r"$K_\mathrm{int}$ (best-layer probe)")

    ax.axhline(0, color="0.35", ls=":", lw=0.8, zorder=0)
    ax.axhline(100, color="0.35", ls="--", lw=0.8, zorder=0)
    ax.text(LEFT - 0.5, 101, "base", fontsize=6.5, color="0.35",
            va="bottom", ha="left")
    ax.text(LEFT - 0.5, 1.5, "chance", fontsize=6.5, color="0.35",
            va="bottom", ha="left")
    ax.set_ylim(-14, 114)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of base model's\nabove-chance signal", fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(LEFT - 0.6, x[-1] + 0.6)
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, handletextpad=0.35, columnspacing=1.2)

    strip = [("WMDP-Bio", vw, WMDP_C, False),
             (r"$K_\mathrm{ext}$", ve, EXT_C, False),
             (r"$K_\mathrm{int}$", vi, INT_C, False),
             (r"$K_\mathrm{ext}\!-\!$WMDP", [b - a for a, b in zip(vw, ve)],
              "0.15", True),
             (r"$K_\mathrm{int}\!-\!K_\mathrm{ext}$",
              [c - b for b, c in zip(ve, vi)], "0.15", True),
             (r"$K_\mathrm{int}\!-\!$WMDP", [c - a for a, c in zip(vw, vi)],
              "0.15", True)]
    tb.set_xlim(ax.get_xlim())
    tb.set_ylim(len(strip) + 2.7, -0.9)
    tb.axis("off")
    for xi, c in zip(x, cols):
        tb.text(xi, len(strip) - 0.15, c, fontsize=6.8, ha="center", va="top",
                rotation=90, style="italic" if c == "mean" else "normal")
    for yr in (-0.55, 2.5):
        tb.plot([LEFT - 0.35, x[-1] + 0.45], [yr, yr], color="0.8", lw=0.6)
    for r, (name, vals, colr, gap) in enumerate(strip):
        tb.text(LEFT, r, name, fontsize=6.4, ha="right", va="center",
                color=colr)
        for xi, v in zip(x, vals):
            tb.text(xi, r, f"{v:+.0f}" if gap else f"{v:.0f}", fontsize=6.4,
                    ha="center", va="center", color=colr)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = ROOT / "plots" / f"retention_three_metric.{ext}"
        fig.savefig(p, bbox_inches="tight",
                    **({"dpi": 200} if ext == "png" else {}))
        print("wrote", p)
    if IMGS.is_dir():
        fig.savefig(IMGS / "retention_three_metric.pdf", bbox_inches="tight")
        print("wrote", IMGS / "retention_three_metric.pdf")


if __name__ == "__main__":
    main()
