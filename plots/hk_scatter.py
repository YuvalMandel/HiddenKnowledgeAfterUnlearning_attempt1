#!/usr/bin/env python3
"""
hk_scatter.py

For each unlearning method, scatter plot of:

  X = base model pairwise external confidence (per question, mean over 3 comparisons)
        mean_j [ sigmoid(ext_base[qi, correct] - ext_base[qi, wrong_j]) ]  ∈ (0, 1)

  Y = hidden knowledge gap (per question)
        K_internal_method - K_external_method  ∈ {-1, -⅔, -⅓, 0, ⅓, ⅔, 1}
        K_internal: multi-layer LR probe (from parquet), layer-26 LR for RMU
        K_external: fraction of pairwise comparisons won by method logits

  573 dots per figure (one per test question), jitter on Y to separate discrete levels.
  Colour = subset: retained / suppressed / emerged / always_wrong.

Output: plots/hk_scatter/hk_scatter_{method}.png

Usage:
  python plots/hk_scatter.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

REPO     = Path(__file__).resolve().parent.parent
EXT_DIR  = REPO / "inside_out_ext"
OUT_DIR  = REPO / "plots" / "hk_scatter"
OUT_DIR.mkdir(exist_ok=True)

SEED       = 42
TRAIN_SIZE = 500
VAL_SIZE   = 200
N_OPTIONS  = 4
KNOWS_THRESHOLD = 0.5

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
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def compute_k_ext(ext, correct_idx, te):
    k = np.zeros(len(te), np.float32)
    for i, qi in enumerate(te):
        c  = correct_idx[qi]
        ws = [ext[qi, j] for j in range(N_OPTIONS) if j != c]
        k[i] = sum(float(ext[qi, c] > w) for w in ws) / len(ws)
    return k


def compute_k_int_from_proba(proba, correct_idx, te):
    k = np.zeros(len(te), np.float32)
    for i, qi in enumerate(te):
        c  = correct_idx[qi]
        ws = [proba[qi, j] for j in range(N_OPTIONS) if j != c]
        k[i] = sum(float(proba[qi, c] > w) for w in ws) / len(ws)
    return k


def mean_base_conf_per_question(base_ext, correct_idx, te):
    """Mean sigmoid pairwise margin across 3 comparisons, per test question."""
    x = np.zeros(len(te), np.float32)
    for i, qi in enumerate(te):
        c = correct_idx[qi]
        margins = [float(sigmoid(base_ext[qi, c] - base_ext[qi, j]))
                   for j in range(N_OPTIONS) if j != c]
        x[i] = float(np.mean(margins))
    return x


def load_k_int_from_parquet(model_id, te, orig_to_pos, parquet_multi):
    sub = parquet_multi[parquet_multi["model_id"] == model_id][
        ["question_idx", "k_internal"]
    ]
    k = np.full(len(te), np.nan, dtype=np.float32)
    for _, row in sub.iterrows():
        pos = orig_to_pos.get(int(row["question_idx"]))
        if pos is not None:
            k[pos] = float(row["k_internal"])
    return k


def plot_method(method_name, x, y, subsets, k_int_source):
    fname = method_name.replace("/", "_").replace("&", "_")
    rng   = np.random.default_rng(SEED)
    jitter = rng.uniform(-0.04, 0.04, size=len(y)).astype(np.float32)
    y_jit  = y + jitter

    fig, ax = plt.subplots(figsize=(6.5, 6))

    for subset in ORDER:
        mask = subsets == subset
        n_q  = int(mask.sum())
        ax.scatter(
            x[mask],
            y_jit[mask],
            c=SUBSET_COLORS[subset],
            s=12,
            alpha=0.45,
            linewidths=0,
            label=f"{SUBSET_LABELS[subset]} (n={n_q})",
            rasterized=True,
        )

    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.axvline(0.5, color="black", linewidth=0.9, linestyle="--", alpha=0.6)

    # Mark discrete Y levels
    for yval, ylabel in [(-1, "−1"), (-2/3, "−⅔"), (-1/3, "−⅓"),
                          (0, "0"), (1/3, "⅓"), (2/3, "⅔"), (1, "1")]:
        ax.axhline(yval, color="#cccccc", linewidth=0.5, linestyle=":", alpha=0.7)
        ax.text(1.02, yval, ylabel, va="center", fontsize=7,
                color="#888888", transform=ax.get_yaxis_transform())

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.12, 1.12)
    ax.set_xlabel(
        "Base model external confidence (mean pairwise)\n"
        r"mean$_j\,[\,\sigma(\mathrm{ext}_{correct} - \mathrm{ext}_{wrong_j})\,]$",
        fontsize=9,
    )
    ax.set_ylabel(
        f"Hidden knowledge gap  ({method_name} ck8)\n"
        r"$K_{internal} - K_{external}$",
        fontsize=9,
    )
    probe_note = f"K_int source: {k_int_source}"
    ax.set_title(
        f"{method_name} (ck8)  —  base confidence vs hidden knowledge gap\n"
        f"({probe_note})",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left", markerscale=1.8)
    fig.tight_layout()

    out_path = OUT_DIR / f"hk_scatter_{fname}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


def main():
    print("Loading WMDP-bio correct_idx ...")
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    correct_idx = np.array([int(ex["answer"]) for ex in ds])
    n_q = len(correct_idx)
    te  = get_test_indices(n_q)
    orig_to_pos = {int(qi): i for i, qi in enumerate(te)}
    print(f"  {len(te)} test questions")

    print("Loading base ext ...")
    base_ext   = np.load(EXT_DIR / "base_bio_ext.npy")
    base_k     = compute_k_ext(base_ext, correct_idx, te)
    base_knows = base_k > KNOWS_THRESHOLD
    x_base     = mean_base_conf_per_question(base_ext, correct_idx, te)

    print("Loading parquet multi-layer K_internal ...")
    df = pd.read_parquet(REPO / "plots" / "all_k_scores.parquet")
    parquet_multi = df[
        (df["layer_config"] == "multi") &
        (df["clf"]          == "LR")    &
        (df["split_type"]   == "single")&
        (df["domain"]       == "bio")
    ]
    available_multi = set(parquet_multi["model_id"].unique())

    for method in METHODS:
        fname    = method.replace("/", "_").replace("&", "_")
        ext_path = EXT_DIR / f"{fname}_ck8_bio_ext.npy"
        if not ext_path.exists():
            print(f"\n{method}: [skip] ext file not found")
            continue
        print(f"\n{method}")

        # K_external of method
        method_ext = np.load(ext_path)
        mk_ext = compute_k_ext(method_ext, correct_idx, te)

        # K_internal of method
        model_id = f"{fname}_ck8"
        if model_id in available_multi:
            mk_int = load_k_int_from_parquet(model_id, te, orig_to_pos, parquet_multi)
            k_int_source = "multi-layer LR (parquet)"
        else:
            int_path = EXT_DIR / f"{fname}_ck8_bio_int_proba.npy"
            if not int_path.exists():
                print(f"  [skip] no K_int data available")
                continue
            proba  = np.load(int_path)
            mk_int = compute_k_int_from_proba(proba, correct_idx, te)
            k_int_source = "layer-26 LR (int_proba)"

        hk_gap = mk_int - mk_ext

        mmk = mk_ext > KNOWS_THRESHOLD
        subsets = np.full(len(te), "always_wrong", dtype=object)
        subsets[ base_knows &  mmk] = "retained"
        subsets[ base_knows & ~mmk] = "suppressed"
        subsets[~base_knows &  mmk] = "emerged"

        for s in ORDER:
            n = int((subsets == s).sum())
            mean_hk = float(hk_gap[subsets == s].mean()) if n > 0 else float("nan")
            print(f"  {s:<14}: n={n:3d}  mean_hk_gap={mean_hk:+.3f}")

        plot_method(method, x_base, hk_gap, subsets, k_int_source)

    print("\nDone.")


if __name__ == "__main__":
    main()
