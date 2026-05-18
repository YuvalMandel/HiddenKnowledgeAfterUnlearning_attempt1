#!/usr/bin/env python3
"""
ckpt_k_subset_heatmap.py

For each unlearning method: 2x2 heatmap grid, one subplot per post-unlearning subset.

Pre-filter: questions where base K_int_full=1 AND K_ext=1 (312/573).
Subsets (defined at ck8, fixed for all checkpoints):
  retained   (int K>0.5, ext K>0.5)   -- top-left,  green
  suppressed (int K>0.5, ext K<=0.5)  -- top-right, red
  forgotten  (int K<=0.5, ext K<=0.5) -- bottom-left, grey
  lucky      (int K<=0.5, ext K>0.5)  -- bottom-right, orange

Each heatmap:
  X = layer (1-32)
  Y = checkpoint (base, ck1 ... ck8)
  color = mean K_internal (fold-mean then question-mean within subset)

Color scale shared across all 4 subsets per method figure.

Output: plots/ckpt_k_subset_heatmap/ckpt_k_subset_{method}.{pdf,png}
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
OUT_DIR = REPO / "plots" / "ckpt_k_subset_heatmap"
OUT_DIR.mkdir(exist_ok=True)

SEED, TRAIN_SIZE, VAL_SIZE, N_OPTIONS = 42, 500, 200, 4
KNOWS_THRESHOLD = 0.5
METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]

SUBSETS = [
    ("retained",   "Retained\n(int K>0.5, ext K>0.5)",    "#2ca02c"),
    ("suppressed", "Suppressed\n(int K>0.5, ext K<=0.5)", "#d62728"),
    ("forgotten",  "Forgotten\n(int K<=0.5, ext K<=0.5)", "#7f7f7f"),
    ("lucky",      "Lucky\n(int K<=0.5, ext K>0.5)",      "#ff7f0e"),
]
SUBSET_GRID = [("retained", "suppressed"), ("forgotten", "lucky")]  # row, col order


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def compute_k_ext(ext, te_subset):
    k = np.zeros(len(te_subset), np.float32)
    for i, qi in enumerate(te_subset):
        c  = correct_idx[qi]
        ws = [ext[qi, j] for j in range(N_OPTIONS) if j != c]
        k[i] = sum(float(ext[qi, c] > w) for w in ws) / len(ws)
    return k


# ── Dataset / split ───────────────────────────────────────────────────────────
print("Loading WMDP-bio ...")
ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
correct_idx = np.array([int(ex["answer"]) for ex in ds])
n_q  = len(correct_idx)
te   = np.random.default_rng(SEED).permutation(n_q)[TRAIN_SIZE + VAL_SIZE:]
n_te = len(te)
orig_to_pos = {int(qi): i for i, qi in enumerate(te)}

# ── Parquet ───────────────────────────────────────────────────────────────────
print("Loading parquet ...")
df = pd.read_parquet(REPO / "plots" / "all_k_scores.parquet")

parquet_multi = df[
    (df["layer_config"] == "full") & (df["clf"] == "LR") &
    (df["split_type"]   == "single") & (df["domain"] == "bio")
]

# CV per-layer data for all models
cv_all = df[
    (df["split_type"]   == "cv") &
    (df["domain"]       == "bio") &
    (df["clf"]          == "LR") &
    (df["probe_type"]   == "own") &
    df["layer_config"].str.match(r"^layer_\d+$")
].copy()
cv_all["layer_idx"] = cv_all["layer_config"].str.extract(r"layer_(\d+)").astype(int)
cv_all = cv_all[cv_all["layer_idx"] > 0]  # skip embedding layer

# ── Pre-filter: base K_int_full=1 AND K_ext=1 ───────────────────────────────
base_single = df[
    (df["model_id"]     == "base") &
    (df["split_type"]   == "single") &
    (df["layer_config"] == "full") &
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
filt_qidx = set(int(qi) for qi in filt_te)
print(f"  Pre-filter: {n_filt} / {n_te} questions")


def get_mean_k_per_layer_ckpt(model_id, qidx_set):
    """
    Returns DataFrame: index=checkpoint_label, columns=layer_idx, values=mean K_internal.
    Averages across 5 CV folds, then across questions in qidx_set.
    """
    sub = cv_all[
        (cv_all["model_id"] == model_id) &
        cv_all["question_idx"].isin(qidx_set)
    ]
    if sub.empty:
        return pd.DataFrame()
    fold_means = (
        sub.groupby(["fold", "layer_idx"])["k_internal"]
        .mean().reset_index()
    )
    result = fold_means.groupby("layer_idx")["k_internal"].mean()
    return result  # Series: layer_idx -> mean K_internal


def build_pivot(method_fname, subset_qidx):
    """
    Build pivot table: rows=checkpoint, cols=layer, values=mean K_internal.
    Includes base model as first row.
    """
    rows = {}

    # Base model row
    base_series = get_mean_k_per_layer_ckpt("base", subset_qidx)
    if not base_series.empty:
        rows["base"] = base_series

    # Method checkpoints ck1-ck8
    for ck in range(1, 9):
        model_id = f"{method_fname}_ck{ck}"
        series = get_mean_k_per_layer_ckpt(model_id, subset_qidx)
        if not series.empty:
            rows[f"ck{ck}"] = series

    if not rows:
        return pd.DataFrame()
    pivot = pd.DataFrame(rows).T  # rows=checkpoints, cols=layers
    pivot.index.name   = "checkpoint"
    pivot.columns.name = "layer"
    return pivot


# ── Main loop ─────────────────────────────────────────────────────────────────
for method in METHODS:
    fname = method.replace("/", "_").replace("&", "_")
    print(f"\n{method}")

    ext_path = EXT_DIR / f"{fname}_ck8_bio_ext.npy"
    if not ext_path.exists():
        print("  [skip] ext file not found")
        continue

    # K_external at ck8
    method_ext = np.load(ext_path)
    mk_ext = compute_k_ext(method_ext, filt_te)

    # K_internal at ck8
    model_id = f"{fname}_ck8"
    sub_ki = parquet_multi[parquet_multi["model_id"] == model_id][
        ["question_idx", "k_internal"]
    ]
    if not sub_ki.empty:
        mk_int = np.full(n_filt, np.nan, np.float32)
        for _, row in sub_ki.iterrows():
            pos = filt_orig_to_pos.get(int(row["question_idx"]))
            if pos is not None:
                mk_int[pos] = float(row["k_internal"])
        k_int_src = "multi-LR"
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
        k_int_src = "layer-26"

    # Classify subsets
    mk_i = mk_int > KNOWS_THRESHOLD
    mk_e = mk_ext > KNOWS_THRESHOLD
    subset_labels = np.full(n_filt, "forgotten", dtype=object)
    subset_labels[ mk_i &  mk_e] = "retained"
    subset_labels[ mk_i & ~mk_e] = "suppressed"
    subset_labels[~mk_i &  mk_e] = "lucky"

    # Question indices per subset
    subset_qidx = {}
    for s_id, *_ in SUBSETS:
        mask = subset_labels == s_id
        subset_qidx[s_id] = set(int(filt_te[i]) for i in np.where(mask)[0])
        print(f"  {s_id:<12}: {len(subset_qidx[s_id]):3d}")

    # Build pivots for all 4 subsets
    pivots = {}
    for s_id, *_ in SUBSETS:
        if not subset_qidx[s_id]:
            continue
        p = build_pivot(fname, subset_qidx[s_id])
        if not p.empty:
            pivots[s_id] = p

    if not pivots:
        print("  [skip] no pivot data")
        continue

    # Shared color scale
    all_vals = np.concatenate([p.values.ravel() for p in pivots.values()])
    all_vals  = all_vals[~np.isnan(all_vals)]
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax_map = {
        "retained":   axes[0, 0],
        "suppressed": axes[0, 1],
        "forgotten":  axes[1, 0],
        "lucky":      axes[1, 1],
    }
    color_map = {s_id: color for s_id, _, color in SUBSETS}

    for s_id, s_title, s_color in SUBSETS:
        ax = ax_map[s_id]
        n  = len(subset_qidx.get(s_id, []))

        if s_id not in pivots:
            ax.text(0.5, 0.5, f"No data\n(n={n})", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(f"{s_title}  (n={n})", fontsize=9, color=s_color)
            continue

        pivot = pivots[s_id]
        layers = sorted(pivot.columns.astype(int))
        ckpts  = list(pivot.index)
        data   = pivot[layers].values  # shape: (n_ckpts, n_layers)

        im = ax.imshow(
            data, aspect="auto", cmap="RdYlGn",
            vmin=vmin, vmax=vmax,
            origin="upper", interpolation="nearest",
        )

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=5.5, rotation=90)
        ax.set_yticks(range(len(ckpts)))
        ax.set_yticklabels(ckpts, fontsize=7)
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("Checkpoint", fontsize=8)
        ax.set_title(f"{s_title}  (n={n})", fontsize=9, color=s_color,
                     fontweight="bold")

        fig.colorbar(im, ax=ax, label="Mean K_internal", shrink=0.85,
                     format="%.2f")

    fig.suptitle(
        f"K_internal per layer × checkpoint — {method} · Bio · LR · 5-fold CV\n"
        f"Pre-filter: base K_int=K_ext=1  ({n_filt}/573)  |  "
        f"Subsets at ck8  |  K_int source: {k_int_src}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    stem = OUT_DIR / f"ckpt_k_subset_{fname}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {stem.name}.png")

print("\nDone.")
