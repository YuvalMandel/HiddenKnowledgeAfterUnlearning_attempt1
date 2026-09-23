#!/usr/bin/env python3
"""WMDP-bio: WMDP's own RMU checkpoints on three model families, and the same
plot with LLM-GAT's Llama-3 RMU added.

Retention view, so all four are comparable despite four different bases:

    R(M) = 100 * (M - M_chance) / (M_base - M_chance)

Each bar is normalised against ITS OWN base, which is why "only the three RMU
models" still carries the base comparison -- the 100 line is each model's own
starting point, not a shared one.

Stacked: grey = what the four-way benchmark still sees, red = what the model's
own logits give away on top of that, blue = what only a probe recovers. Bar
total is K_int retention.

Writes two figures:
  plots/rmu_three_families.png/.pdf            -- the three WMDP RMU models
  plots/rmu_three_families_plus_gat.png/.pdf   -- plus LLM-GAT's Llama-3 RMU

NOTE the fourth bar is a DIFFERENT GROUP's implementation: cais publishes no
Llama-3 RMU, so LLM-GAT's is a reimplementation on another base. Also, WMDP's
RMU removed bio AND cyber; LLM-GAT's suite targets bio.

Usage: python plots/rmu_three_families.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hk_utils import use_iclr_style      # noqa: E402

use_iclr_style()

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "inside_out_out"

# (label, rmu_id, rmu_wmdp_acc, base_id, base_wmdp_acc, source)
WMDP_RMU = [
    ("Zephyr-7B",    "zephyr_rmu",  0.312, "zephyr_base",  0.637, "cais"),
    ("Mixtral-8x7B", "mixtral_rmu", 0.340, "mixtral_base", 0.748, "cais"),
    ("Yi-34B",       "yi_rmu",      0.307, "yi_base",      0.753, "cais"),
]
GAT_RMU = ("Llama-3-8B", "RMU_ck8", 0.26, "base", 0.70, "LLM-GAT")

INT_C, EXT_C, WMDP_C = "#2166ac", "#d6604d", "#4d4d4d"


def k_of(mid):
    df = pd.read_parquet(OUT_DIR / mid / "k_scores.parquet")
    cv = df[(df.split_type == "cv") & (df.domain == "bio") & (df.clf == "LR")
            & (df.probe_type == "own") & (df.layer_config == "best_layer")]
    f = cv.groupby("fold").agg(ki=("k_internal", "mean"),
                               ke=("k_external", "mean"))
    return dict(k_int=f.ki.mean(), k_ext=f.ke.mean(),
                k_int_sd=f.ki.std(ddof=1), k_ext_sd=f.ke.std(ddof=1))


def R(v, chance, top):
    """Retention, floored at 0: below chance is no evidence, not
    negative evidence."""
    return max(0.0, 100.0 * (v - chance) / (top - chance))


def row(label, rmu, acc, bid, bacc, src):
    r, b = k_of(rmu), k_of(bid)
    return dict(label=label, src=src,
                wmdp=R(acc, 0.25, bacc),
                k_ext=R(r["k_ext"], 0.5, b["k_ext"]),
                k_int=R(r["k_int"], 0.5, b["k_int"]),
                raw_ke=r["k_ext"], raw_ki=r["k_int"],
                ke_sd=r["k_ext_sd"], ki_sd=r["k_int_sd"],
                base_ke=b["k_ext"], base_ki=b["k_int"],
                raw_wmdp=acc, base_wmdp=bacc)


def draw(d, out_stem, title):
    n = len(d)
    fig, ax = plt.subplots(figsize=(1.35 * n + 2.2, 4.2))
    x = list(range(n))
    W = 0.44
    vw, ve, vi = d.wmdp.tolist(), d.k_ext.tolist(), d.k_int.tolist()
    ax.bar(x, vw, W, color=WMDP_C, zorder=5, label="WMDP-Bio accuracy")
    ax.bar(x, [b - a for a, b in zip(vw, ve)], W, bottom=vw, color=EXT_C,
           zorder=3, label=r"$+\;K_\mathrm{ext}$ (logit margin)")
    ax.bar(x, [c - b for b, c in zip(ve, vi)], W, bottom=ve, color=INT_C,
           zorder=3, label=r"$+\;K_\mathrm{int}$ (best-layer probe)")

    for xi, a, b, c in zip(x, vw, ve, vi):
        for lo, hi, colr in ((a, b, EXT_C), (b, c, INT_C)):
            txt = f"{hi - lo:+.0f}"
            if abs(hi - lo) >= 9:
                ax.text(xi, (lo + hi) / 2, txt, ha="center", va="center",
                        fontsize=7.2, color="white", zorder=6)
            else:
                ax.text(xi + W / 2 + 0.04, (lo + hi) / 2, txt, ha="left",
                        va="center", fontsize=7.2, color=colr, zorder=6)
        ax.plot([xi - W / 2 - 0.07, xi - W / 2], [a, a], color="#5a5a5a",
                lw=1.0, zorder=6, clip_on=False)
        ax.text(xi - W / 2 - 0.10, a, f"{a:.0f}", ha="right", va="center",
                fontsize=7.0, color="#5a5a5a", zorder=6)
        ax.plot([xi - W / 2 - 0.07, xi - W / 2], [b, b], color=EXT_C,
                lw=1.0, zorder=6, clip_on=False)
        ax.text(xi - W / 2 - 0.10, b, f"{b:.0f}", ha="right", va="center",
                fontsize=7.0, color=EXT_C, zorder=6)
        ax.text(xi, c + 2.2, f"{c:.0f}", ha="center", va="bottom",
                fontsize=8.0, color=INT_C, fontweight="bold", zorder=6)

    ax.axhline(0, color="0.35", ls=":", lw=0.8, zorder=1)
    ax.axhline(100, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.text(-0.62, 101, "own base model", fontsize=6.8, color="0.35",
            va="bottom", ha="left")
    ax.set_ylim(-16, 116)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of its own base model's\nabove-chance signal retained",
                  fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nRMU" + ("" if s == "cais" else "\n(LLM-GAT)")
                        for l, s in zip(d.label, d.src)], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlim(-0.72, n - 0.32)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=9, pad=14)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = ROOT / "plots" / f"{out_stem}.{ext}"
        fig.savefig(p, bbox_inches="tight",
                    **({"dpi": 200} if ext == "png" else {}))
        print("wrote", p)
    plt.close(fig)


def draw_raw(d, out_stem, title):
    """Figure 3 style: raw pairwise K on the left axis, four-way WMDP accuracy
    on the right, each read against its own chance floor (0.5 and 0.25).

    Raw K is not comparable across bases, so each group carries a dashed marker
    at ITS OWN base value for all three metrics -- the drop from that marker to
    the bar is what the group actually says."""
    n = len(d)
    fig, ax = plt.subplots(figsize=(1.55 * n + 2.4, 4.3))
    ax2 = ax.twinx()
    x = list(range(n))
    W = 0.26
    xw = [i - W for i in x]
    xe = list(x)
    xi_ = [i + W for i in x]

    ax2.bar(xw, d.raw_wmdp, W, color=WMDP_C, zorder=3,
            label="WMDP-Bio accuracy (right axis)")
    ax.bar(xe, d.raw_ke, W, color=EXT_C, zorder=3,
           label=r"$K_\mathrm{ext}$ (logit margin)")
    ax.bar(xi_, d.raw_ki, W, color=INT_C, zorder=3,
           label=r"$K_\mathrm{int}$ (best-layer probe)")
    ax.errorbar(xe, d.raw_ke, yerr=d.ke_sd, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=1.8, alpha=0.65, zorder=5)
    ax.errorbar(xi_, d.raw_ki, yerr=d.ki_sd, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=1.8, alpha=0.65, zorder=5)

    # dashed base markers, one per metric per group
    for xx, col, vals in ((xw, "#5a5a5a", d.base_wmdp),
                          (xe, EXT_C, d.base_ke), (xi_, INT_C, d.base_ki)):
        for xi2, vv in zip(xx, vals):
            (ax2 if col == "#5a5a5a" else ax).plot(
                [xi2 - W / 2, xi2 + W / 2], [vv, vv], color=col, lw=1.3,
                ls=(0, (2.2, 1.4)), alpha=0.85, zorder=7)

    for xx, v, sd, c in ((xe, d.raw_ke, d.ke_sd, EXT_C),
                         (xi_, d.raw_ki, d.ki_sd, INT_C)):
        for xi2, vv, ss in zip(xx, v, sd):
            ax.text(xi2, vv + ss + 0.012, f"{vv:.3f}", ha="center",
                    va="bottom", fontsize=6.4, color=c, rotation=90, zorder=6)
    for xi2, vv in zip(xw, d.raw_wmdp):
        ax2.text(xi2, vv + 0.010, f"{vv:.2f}", ha="center", va="bottom",
                 fontsize=6.4, color="#5a5a5a", rotation=90, zorder=6)

    ax.axhline(0.5, color=EXT_C, ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax2.axhline(0.25, color="#5a5a5a", ls=":", lw=1.0, alpha=0.85, zorder=1)
    ax.text(-0.92, 0.506, "chance 0.50", fontsize=6.6, color=EXT_C,
            va="bottom", ha="left")
    ax2.text(-0.92, 0.256, "chance 0.25", fontsize=6.6, color="#5a5a5a",
             va="bottom", ha="left")
    ax.plot([], [], color="0.35", lw=1.3, ls=(0, (2.2, 1.4)),
            label="its own base model")

    ax.set_ylim(0.15, 0.95)
    ax2.set_ylim(0.15, 0.83)
    ax.set_ylabel("Knowledge: pairwise $K$", fontsize=8)
    ax2.set_ylabel("Accuracy: WMDP-Bio (4-way)", fontsize=8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticks(x)
    NL = chr(10)
    labs = [lab + NL + "RMU" + ("" if src == "cais" else NL + "(LLM-GAT)")
            for lab, src in zip(d.label, d.src)]
    ax.set_xticklabels(labs, fontsize=8)
    ax.tick_params(labelsize=8)
    ax2.tick_params(labelsize=8)
    ax.set_xlim(-0.95, n - 0.35)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=9, pad=14)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h2 + h1, l2 + l1, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False,
              handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        pth = ROOT / "plots" / f"{out_stem}.{ext}"
        fig.savefig(pth, bbox_inches="tight",
                    **({"dpi": 200} if ext == "png" else {}))
        print("wrote", pth)
    plt.close(fig)


def main():
    three = pd.DataFrame([row(*a) for a in WMDP_RMU])
    four = pd.DataFrame([row(*a) for a in WMDP_RMU] + [row(*GAT_RMU)])

    print("\n=== raw scores (WMDP-bio, ck8/final) ===")
    for _, r in four.iterrows():
        print(f"{r.label:14s} base  WMDP {r.base_wmdp:.3f}  "
              f"K_ext {r.base_ke:.3f}  K_int {r.base_ki:.3f}")
        print(f"{'':14s} RMU   WMDP {r.raw_wmdp:.3f}  "
              f"K_ext {r.raw_ke:.3f}  K_int {r.raw_ki:.3f}   "
              f"gap {r.raw_ki - r.raw_ke:+.3f}")
    print("\n=== retention, % of own base above-chance signal ===")
    print(four[["label", "src", "wmdp", "k_ext", "k_int"]]
          .round(1).to_string(index=False))

    draw(three, "rmu_three_families",
         "WMDP's own RMU checkpoints, WMDP-Bio")
    draw(four, "rmu_three_families_plus_gat",
         "RMU on four base models, WMDP-Bio "
         "(rightmost is LLM-GAT's implementation)")
    draw_raw(three, "rmu_three_families_raw",
             "WMDP's own RMU checkpoints, WMDP-Bio (raw scores)")
    draw_raw(four, "rmu_three_families_raw_plus_gat",
             "RMU on four base models, WMDP-Bio (raw scores; "
             "rightmost is LLM-GAT's implementation)")


if __name__ == "__main__":
    main()
