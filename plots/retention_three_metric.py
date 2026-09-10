#!/usr/bin/env python3
"""The three-metric retention figure: WMDP vs K_ext vs K_int on one scale.

Table `tab:wmdp-k-comparison` says the same thing in seven columns. The single
comparable quantity across three metrics with three different chance floors and
three different base-model ceilings is the fraction of the base model's
above-chance signal that survives unlearning:

    R(M) = 100 * (M - M_chance) / (M_base - M_chance)

so every metric runs 0 (at chance) to 100 (base model) and the three become
directly comparable.

On this scale the three quantities are additive, so the figure is a stacked bar:
the grey base is what the standard benchmark still sees, the red segment is what
the model's own readout gives away on top of that, and the blue segment is what
only a probe recovers. Segment heights ARE the two gaps, and the full bar is
K_int. Raw scores stack nowhere -- they are three measurements on two different
scales -- which is why the companion figure keeps them side by side.

Every number is printed on the plot: each segment carries its own height, inside
when the segment is tall enough and just outside when it is not, and the bar
total is printed above.

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

    # ---- stacked: grey = WMDP, red = K_ext-WMDP, blue = K_int-K_ext
    cols = list(d.label) + ["mean"]
    vw = d.wmdp.tolist() + [mean.wmdp]
    ve = d.k_ext.tolist() + [mean.k_ext]
    vi = d.k_int.tolist() + [mean.k_int]
    g1 = [b - a for a, b in zip(vw, ve)]            # K_ext - WMDP
    g2 = [c - b for b, c in zip(ve, vi)]            # K_int - K_ext
    x = list(range(len(cols)))
    x[-1] += 0.6                                    # set the mean column apart
    W = 0.62

    fig, ax = plt.subplots(figsize=iclr_figsize(aspect=5.0 / 7,
                                                width_frac=0.80))
    # zorder above the other two: where WMDP retention is negative (ELM,
    # whose WMDP accuracy sits below the four-way chance floor) the red
    # segment also starts below zero, and would otherwise paint over the
    # grey. Drawn on top, the grey reads 0 to -2 as it should.
    ax.bar(x, vw, W, color=WMDP_C, zorder=5, label="WMDP-Bio accuracy")
    ax.bar(x, g1, W, bottom=vw, color=EXT_C, zorder=3,
           label=r"$+\;K_\mathrm{ext}$ (logit margin)")
    ax.bar(x, g2, W, bottom=ve, color=INT_C, zorder=3,
           label=r"$+\;K_\mathrm{int}$ (best-layer probe)")

    def seg(xi, lo, hi, txt, colr):
        """Number inside its segment, or just outside when it will not fit."""
        h = hi - lo
        if abs(h) >= 9:
            ax.text(xi, (lo + hi) / 2, txt, ha="center", va="center",
                    fontsize=6.1, color="white", zorder=6)
        else:
            ax.text(xi + W / 2 + 0.06, (lo + hi) / 2, txt, ha="left",
                    va="center", fontsize=6.1, color=colr, zorder=6)

    for xi, a, b, c in zip(x, vw, ve, vi):
        seg(xi, 0, a, f"{a:.0f}", "#5a5a5a")
        seg(xi, a, b, f"+{b - a:.0f}", EXT_C)
        seg(xi, b, c, f"+{c - b:.0f}", INT_C)
        ax.text(xi, c + 2.5, f"{c:.0f}", ha="center", va="bottom",
                fontsize=6.6, color=INT_C, zorder=6)

    ax.axhline(0, color="0.35", ls=":", lw=0.8, zorder=1)
    ax.axhline(100, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.text(-0.95, 101, "base model", fontsize=6.3, color="0.35",
            va="bottom", ha="left")
    ax.set_ylim(-9, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of base model's above-chance\nsignal retained",
                  fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=34, ha="right", fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(-1.0, x[-1] + 0.75)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.6)
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
