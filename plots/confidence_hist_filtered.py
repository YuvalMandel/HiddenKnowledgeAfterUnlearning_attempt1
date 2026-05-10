#!/usr/bin/env python3
"""
confidence_hist_filtered.py

Pre-filter: only questions where base model K_int_multi=1 AND K_ext=1 (307/573).

For each unlearning method, 2x2 grid of histograms (one subplot per post-unlearning subset):
  retained   (int K>0.5, ext K>0.5)   -- top-left,  green
  suppressed (int K>0.5, ext K<=0.5)  -- top-right, red
  forgotten  (int K<=0.5, ext K<=0.5) -- bottom-left, grey
  lucky      (int K<=0.5, ext K>0.5)  -- bottom-right, orange

Each subplot: two histograms on dual Y axes.
  Left  axis (colored): correct option P(True) = sigmoid(base_ext[qi, correct])
  Right axis (grey):    wrong   option P(True) = sigmoid(base_ext[qi, wrong_j])
  Right ylim = 3 * left ylim  so same-proportion distributions appear same height.

Output: plots/confidence_hist_filtered/confidence_hist_{method}.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

REPO    = Path(__file__).resolve().parent.parent
EXT_DIR = REPO / "inside_out_ext"
OUT_DIR = REPO / "plots" / "confidence_hist_filtered"
OUT_DIR.mkdir(exist_ok=True)

SEED, TRAIN_SIZE, VAL_SIZE, N_OPTIONS = 42, 500, 200, 4
KNOWS_THRESHOLD = 0.5
BINS = np.linspace(0, 1, 31)

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]

SUBSETS = [
    ("retained",   "Retained\n(int K>0.5, ext K>0.5)",    "#2ca02c"),
    ("suppressed", "Suppressed\n(int K>0.5, ext K<=0.5)", "#d62728"),
    ("forgotten",  "Forgotten\n(int K<=0.5, ext K<=0.5)", "#7f7f7f"),
    ("lucky",      "Lucky\n(int K<=0.5, ext K>0.5)",      "#ff7f0e"),
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def compute_k_ext(ext, te_subset):
    k = np.zeros(len(te_subset), np.float32)
    for i, qi in enumerate(te_subset):
        c  = correct_idx[qi]
        ws = [ext[qi, j] for j in range(N_OPTIONS) if j != c]
        k[i] = sum(float(ext[qi, c] > w) for w in ws) / len(ws)
    return k


# ── Load dataset and split ────────────────────────────────────────────────────
print("Loading WMDP-bio ...")
ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
correct_idx = np.array([int(ex["answer"]) for ex in ds])
n_q = len(correct_idx)
te  = np.random.default_rng(SEED).permutation(n_q)[TRAIN_SIZE + VAL_SIZE:]
n_te = len(te)
orig_to_pos = {int(qi): i for i, qi in enumerate(te)}

# ── Pre-filter ────────────────────────────────────────────────────────────────
print("Loading parquet ...")
df = pd.read_parquet(REPO / "plots" / "all_k_scores.parquet")
parquet_multi = df[
    (df["layer_config"] == "multi") & (df["clf"] == "LR") &
    (df["split_type"]   == "single") & (df["domain"] == "bio")
]

base_single = df[
    (df["model_id"]     == "base") &
    (df["split_type"]   == "single") &
    (df["layer_config"] == "multi") &
    (df["clf"]          == "LR") &
    (df["domain"]       == "bio")
][["question_idx", "k_internal", "k_external"]]

filt_mask = np.zeros(n_te, dtype=bool)
for _, row in base_single.iterrows():
    pos = orig_to_pos.get(int(row["question_idx"]))
    if pos is not None and row["k_internal"] == 1.0 and row["k_external"] == 1.0:
        filt_mask[pos] = True

filt_te = te[filt_mask]
n_filt  = len(filt_te)
filt_orig_to_pos = {int(qi): i for i, qi in enumerate(filt_te)}
print(f"  Pre-filter: {n_filt} / {n_te} questions")

# ── Base ext confidences (per option, not pairwise) ───────────────────────────
print("Loading base ext ...")
base_ext = np.load(EXT_DIR / "base_bio_ext.npy")


def collect_option_confs(subset_labels):
    corr_conf = {s: [] for s, *_ in SUBSETS}
    wrong_conf = {s: [] for s, *_ in SUBSETS}
    for i, qi in enumerate(filt_te):
        s = subset_labels[i]
        c = correct_idx[qi]
        corr_conf[s].append(float(sigmoid(base_ext[qi, c])))
        for j in range(N_OPTIONS):
            if j != c:
                wrong_conf[s].append(float(sigmoid(base_ext[qi, j])))
    return corr_conf, wrong_conf


def make_figure(method_name, corr_conf, wrong_conf, k_int_src):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_map = {
        "retained":   axes[0, 0],
        "suppressed": axes[0, 1],
        "forgotten":  axes[1, 0],
        "lucky":      axes[1, 1],
    }

    for s_id, s_title, color in SUBSETS:
        ax  = ax_map[s_id]
        ax2 = ax.twinx()

        cc = np.array(corr_conf[s_id])
        wc = np.array(wrong_conf[s_id])
        nc, nw = len(cc), len(wc)

        ax2.hist(wc, bins=BINS, color="#888888", alpha=0.45, label=f"Wrong (n={nw})")
        ax.hist( cc, bins=BINS, color=color,     alpha=0.75, label=f"Correct (n={nc})")

        # Enforce 3x right scale relative to left
        left_max = ax.get_ylim()[1]
        ax2.set_ylim(0, 3 * left_max)

        ax.set_xlim(0, 1)
        ax.set_xlabel(
            "P(True) from base model logits\n"
            r"$\sigma(\mathrm{ext}_{base}[q,\,j])$",
            fontsize=8.5,
        )
        ax.set_ylabel("Count (correct options)", fontsize=8.5, color=color)
        ax2.set_ylabel("Count (wrong options)",  fontsize=8.5, color="#555555")
        ax.tick_params(axis="y", labelcolor=color)
        ax2.tick_params(axis="y", labelcolor="#555555")
        ax.set_title(s_title, fontsize=9)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    fig.suptitle(
        f"Base model P(True) per option — {method_name} (ck8)\n"
        f"Pre-filter: base K_int=K_ext=1  ({n_filt}/573 questions)\n"
        f"Subsets: post-unlearning K_int vs K_ext  |  K_int source: {k_int_src}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    fname   = method_name.replace("/", "_").replace("&", "_")
    out     = OUT_DIR / f"confidence_hist_{fname}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


# ── Main loop ─────────────────────────────────────────────────────────────────
for method in METHODS:
    fname = method.replace("/", "_").replace("&", "_")
    ext_path = EXT_DIR / f"{fname}_ck8_bio_ext.npy"
    if not ext_path.exists():
        print(f"\n{method}: [skip] ext file not found")
        continue
    print(f"\n{method}")

    method_ext = np.load(ext_path)
    mk_ext = compute_k_ext(method_ext, filt_te)

    model_id = f"{fname}_ck8"
    sub = parquet_multi[parquet_multi["model_id"] == model_id][
        ["question_idx", "k_internal"]
    ]
    if not sub.empty:
        mk_int = np.full(n_filt, np.nan, np.float32)
        for _, row in sub.iterrows():
            pos = filt_orig_to_pos.get(int(row["question_idx"]))
            if pos is not None:
                mk_int[pos] = float(row["k_internal"])
        k_int_src = "multi-layer LR"
    else:
        proba_path = EXT_DIR / f"{fname}_ck8_bio_int_proba.npy"
        if not proba_path.exists():
            print("  [skip] no K_int data")
            continue
        proba  = np.load(proba_path)
        mk_int = np.zeros(n_filt, np.float32)
        for i, qi in enumerate(filt_te):
            c  = correct_idx[qi]
            ws = [proba[qi, j] for j in range(N_OPTIONS) if j != c]
            mk_int[i] = sum(float(proba[qi, c] > w) for w in ws) / len(ws)
        k_int_src = "layer-26 LR"

    mk_i = mk_int > KNOWS_THRESHOLD
    mk_e = mk_ext > KNOWS_THRESHOLD
    subset_labels = np.full(n_filt, "forgotten", dtype=object)
    subset_labels[ mk_i &  mk_e] = "retained"
    subset_labels[ mk_i & ~mk_e] = "suppressed"
    subset_labels[~mk_i &  mk_e] = "lucky"

    for s_id, *_ in SUBSETS:
        n = int((subset_labels == s_id).sum())
        print(f"  {s_id:<12}: {n:3d}")

    corr_conf, wrong_conf = collect_option_confs(subset_labels)
    make_figure(method, corr_conf, wrong_conf, k_int_src)

print("\nDone.")
