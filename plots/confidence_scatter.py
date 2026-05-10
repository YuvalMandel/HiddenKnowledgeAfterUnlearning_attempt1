#!/usr/bin/env python3
"""
confidence_scatter.py

For each unlearning method (ck8), scatter plot of base vs post-unlearning
pairwise confidence. Each dot is one (question, wrong-option) comparison.

  X = sigmoid(ext_base[qi, correct]   - ext_base[qi, wrong_j])
  Y = sigmoid(ext_method[qi, correct] - ext_method[qi, wrong_j])

Colour = knowledge subset (retained / suppressed / emerged / always_wrong).
Quadrant lines at 0.5 / 0.5 divide the plot into the 4 subset regions.

573 questions × 3 comparisons = 1719 dots per figure.

Usage:
  python plots/confidence_scatter.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset

REPO    = Path(__file__).resolve().parent.parent
EXT_DIR = REPO / "inside_out_ext"
OUT_DIR = REPO / "plots" / "confidence_scatter"
OUT_DIR.mkdir(exist_ok=True)

SEED       = 42
TRAIN_SIZE = 500
VAL_SIZE   = 200
N_OPTIONS  = 4

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]

SUBSET_COLORS = {
    "retained":    "#2ca02c",
    "suppressed":  "#d62728",
    "emerged":     "#1f77b4",
    "always_wrong":"#888888",
}
SUBSET_LABELS = {
    "retained":    "Retained",
    "suppressed":  "Suppressed",
    "emerged":     "Emerged",
    "always_wrong":"Always wrong",
}
ORDER = ["retained", "suppressed", "emerged", "always_wrong"]


def get_test_indices(n_q):
    return np.random.default_rng(SEED).permutation(n_q)[TRAIN_SIZE + VAL_SIZE:]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))



def pairwise_sigmoid(ext, correct_idx, te):
    """
    Returns:
      conf   (n_te * 3,)  sigmoid pairwise margin per comparison
      q_map  (n_te * 3,)  index into te for each comparison
    """
    conf, q_map = [], []
    for i, qi in enumerate(te):
        c  = correct_idx[qi]
        cs = ext[qi, c]
        for j in range(N_OPTIONS):
            if j != c:
                conf.append(float(sigmoid(np.array(cs - ext[qi, j]))))
                q_map.append(i)
    return np.array(conf, dtype=np.float32), np.array(q_map, dtype=np.int32)



def classify_pairs(base_conf, method_conf):
    """Classify each pairwise comparison by its own (X, Y) quadrant."""
    subsets = np.full(len(base_conf), "always_wrong", dtype=object)
    subsets[ (base_conf > 0.5) &  (method_conf > 0.5)] = "retained"
    subsets[ (base_conf > 0.5) & (method_conf <= 0.5)] = "suppressed"
    subsets[(base_conf <= 0.5) &  (method_conf > 0.5)] = "emerged"
    return subsets


def plot_method(method_name, base_ext, correct_idx, te,
                base_conf, q_map):
    fname    = method_name.replace("/", "_").replace("&", "_")
    ext_path = EXT_DIR / f"{fname}_ck8_bio_ext.npy"
    if not ext_path.exists():
        print(f"  [skip] {ext_path.name} not found")
        return

    method_ext  = np.load(ext_path)
    method_conf, _ = pairwise_sigmoid(method_ext, correct_idx, te)

    pair_subsets = classify_pairs(base_conf, method_conf)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    for subset in ORDER:
        mask = pair_subsets == subset
        n    = int(mask.sum())
        ax.scatter(
            base_conf[mask],
            method_conf[mask],
            c=SUBSET_COLORS[subset],
            s=6,
            alpha=0.35,
            linewidths=0,
            label=f"{SUBSET_LABELS[subset]} (n={n})",
            rasterized=True,
        )

    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.6, linestyle=":", alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Base model pairwise confidence\nsigmoid(logit_correct − logit_wrong)",
                  fontsize=10)
    ax.set_ylabel(f"{method_name} pairwise confidence\nsigmoid(logit_correct − logit_wrong)",
                  fontsize=10)
    ax.set_title(f"{method_name} (ck8) — base vs post-unlearning confidence",
                 fontsize=11)

    ax.legend(fontsize=8, loc="upper left", markerscale=2)
    fig.tight_layout()

    out_path = OUT_DIR / f"confidence_scatter_{fname}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


def main():
    print("Loading WMDP-bio correct_idx ...")
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    correct_idx = np.array([int(ex["answer"]) for ex in ds])
    n_q = len(correct_idx)
    te  = get_test_indices(n_q)
    print(f"  {len(te)} test questions, {len(te)*3} comparisons per figure")

    print("Loading base ext ...")
    base_ext  = np.load(EXT_DIR / "base_bio_ext.npy")
    base_conf, q_map = pairwise_sigmoid(base_ext, correct_idx, te)

    for method in METHODS:
        print(f"\n{method}")
        plot_method(method, base_ext, correct_idx, te, base_conf, q_map)

    print("\nDone.")


if __name__ == "__main__":
    main()
