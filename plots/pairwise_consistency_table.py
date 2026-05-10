#!/usr/bin/env python3
"""
pairwise_consistency_table.py

For every pair of unlearning methods, compute the fraction of test questions
on which both methods assign the same knowledge subset:
  retained / suppressed / emerged / always_wrong
  (subset is determined relative to the base model K_external)

Outputs:
  - Ranked list printed to stdout (most → least agreement)
  - plots/pairwise_consistency_matrix.png  — 8×8 heatmap
  - plots/pairwise_consistency_ranked.png  — horizontal bar chart of all 28 pairs
"""

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datasets import load_dataset

REPO    = Path(__file__).resolve().parent.parent
EXT_DIR = REPO / "inside_out_ext"
OUT_DIR = REPO / "plots"

SEED       = 42
TRAIN_SIZE = 500
VAL_SIZE   = 200
N_OPTIONS  = 4
KNOWS_THRESHOLD = 0.5

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]


def get_test_indices(n_q):
    return np.random.default_rng(SEED).permutation(n_q)[TRAIN_SIZE + VAL_SIZE:]


def compute_k_ext(ext, correct_idx, te):
    k = np.zeros(len(te), dtype=np.float32)
    for i, qi in enumerate(te):
        c  = correct_idx[qi]
        cs = ext[qi, c]
        ws = [ext[qi, j] for j in range(N_OPTIONS) if j != c]
        k[i] = sum(float(cs > w) for w in ws) / len(ws)
    return k


def classify(base_knows, method_knows):
    subsets = np.full(len(base_knows), "always_wrong", dtype=object)
    subsets[ base_knows &  method_knows] = "retained"
    subsets[ base_knows & ~method_knows] = "suppressed"
    subsets[~base_knows &  method_knows] = "emerged"
    return subsets


def main():
    print("Loading WMDP-bio ...")
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    correct_idx = np.array([int(ex["answer"]) for ex in ds])
    n_q = len(correct_idx)
    te  = get_test_indices(n_q)
    n_te = len(te)
    print(f"  {n_te} test questions")

    base_ext   = np.load(EXT_DIR / "base_bio_ext.npy")
    base_k     = compute_k_ext(base_ext, correct_idx, te)
    base_knows = base_k > KNOWS_THRESHOLD

    # Per-method subset classification
    method_subsets = {}
    available = []
    for method in METHODS:
        fname    = method.replace("/", "_").replace("&", "_")
        ext_path = EXT_DIR / f"{fname}_ck8_bio_ext.npy"
        if not ext_path.exists():
            print(f"  [skip] {method}")
            continue
        mk = compute_k_ext(np.load(ext_path), correct_idx, te)
        method_subsets[method] = classify(base_knows, mk > KNOWS_THRESHOLD)
        available.append(method)
        print(f"  {method}: retained={int((method_subsets[method]=='retained').sum())}  "
              f"suppressed={int((method_subsets[method]=='suppressed').sum())}  "
              f"emerged={int((method_subsets[method]=='emerged').sum())}  "
              f"always_wrong={int((method_subsets[method]=='always_wrong').sum())}")

    # Pairwise agreement
    pairs = []
    for m1, m2 in combinations(available, 2):
        agree = float((method_subsets[m1] == method_subsets[m2]).mean())
        pairs.append((m1, m2, agree))

    pairs.sort(key=lambda x: -x[2])

    print("\nRanked pairwise agreement (fraction of questions with same subset):")
    print(f"{'Rank':>4}  {'Method A':<12}  {'Method B':<12}  {'Agreement':>9}")
    print("-" * 44)
    for rank, (m1, m2, ag) in enumerate(pairs, 1):
        print(f"{rank:>4}  {m1:<12}  {m2:<12}  {ag:>8.1%}")

    # 8x8 agreement matrix
    n = len(available)
    mat = np.full((n, n), np.nan)
    for m1, m2, ag in pairs:
        i, j = available.index(m1), available.index(m2)
        mat[i, j] = mat[j, i] = ag
    np.fill_diagonal(mat, 1.0)

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap=cmap, aspect="equal")
    plt.colorbar(im, ax=ax, label="Fraction agreeing on subset", shrink=0.82)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(available, fontsize=9)
    ax.set_title("Pairwise subset-classification agreement (ck8, WMDP-bio test)", fontsize=10)

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                val = mat[i, j]
                color = "black" if 0.65 < val < 0.95 else "white"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.tight_layout()
    out_path = OUT_DIR / "pairwise_consistency_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_path.name}")

    # --- Ranked bar chart ---
    labels = [f"{m1} / {m2}" for m1, m2, _ in pairs]
    values = [ag for _, _, ag in pairs]

    # Colour bars by the dominant group
    GROUP = {
        "GradDiff": "strong", "RepNoise": "strong", "TAR": "strong",
        "RMU": "moderate", "RMU-LAT": "moderate", "ELM": "moderate",
        "RR": "preserver", "PB_J": "preserver",
    }
    GROUP_COLOR = {"strong": "#d62728", "moderate": "#ff7f0e", "preserver": "#1f77b4"}

    def bar_color(m1, m2):
        g1, g2 = GROUP.get(m1, "?"), GROUP.get(m2, "?")
        if g1 == g2:
            return GROUP_COLOR[g1]
        return "#888888"

    colors = [bar_color(m1, m2) for m1, m2, _ in pairs]

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    y = np.arange(len(pairs))
    ax2.barh(y, values, color=colors, edgecolor="white", linewidth=0.4)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Fraction of questions with same subset classification", fontsize=9)
    ax2.set_title("Ranked pairwise subset-classification agreement (ck8)", fontsize=10)
    ax2.axvline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlim(0, 1)

    for xi, val in enumerate(values):
        ax2.text(val + 0.005, xi, f"{val:.1%}", va="center", fontsize=7)

    # Legend
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=GROUP_COLOR["strong"],    label="Both strong compressors"),
        mpatches.Patch(color=GROUP_COLOR["moderate"],  label="Both moderate compressors"),
        mpatches.Patch(color=GROUP_COLOR["preserver"], label="Both preservers"),
        mpatches.Patch(color="#888888",                label="Cross-group pair"),
    ]
    ax2.legend(handles=legend_handles, fontsize=8, loc="lower right")

    ax2.invert_yaxis()
    fig2.tight_layout()
    out_path2 = OUT_DIR / "pairwise_consistency_ranked.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved -> {out_path2.name}")


if __name__ == "__main__":
    main()
