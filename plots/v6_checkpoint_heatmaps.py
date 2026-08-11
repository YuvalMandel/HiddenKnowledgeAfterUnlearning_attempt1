#!/usr/bin/env python3
"""Per-layer probe-AUC heatmaps over unlearning checkpoints, one per method.

Replaces the `plot_layer_accuracy.py --mode checkpoints` route, whose raw
per-checkpoint hidden states (`checkpoints/sweep_<M>/ck<N>/hs_test.npy`) no
longer exist.  The same quantity is already precomputed in the v6 pipeline's
`inside_out_out/<model_id>/k_scores_layer.parquet`, so we read those instead:
same 500/200/573 single split, same LR probe, same own-probe source, so the
numbers are directly comparable to the previous figures.

Outputs: plots/heatmap_checkpoints_auc_lr_method_bio_<METHOD>.pdf/.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hk_utils import METHODS, ICLR_TEXTWIDTH_IN, use_iclr_style

use_iclr_style()

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "inside_out_out"
SAVE_DIR = Path(__file__).resolve().parent
N_LAYERS = 32          # layer 0 is the embedding, skipped -- matches the old figure
N_CKPTS = 8
LABELS = {"PB_J": "PB&J"}


def layer_auc(model_id):
    """Per-layer test AUC for one model, as an array over layers 1..N_LAYERS."""
    p = OUT_DIR / model_id / "k_scores_layer.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["layer_config", "test_auc"])
    # test_auc is a model-level metric broadcast across the 573 test rows
    per_layer = df.groupby("layer_config")["test_auc"].first()

    out = np.full(N_LAYERS, np.nan)
    for cfg, v in per_layer.items():
        if not str(cfg).startswith("layer_"):
            continue
        l = int(str(cfg).split("_")[1])
        if 1 <= l <= N_LAYERS:
            out[l - 1] = float(v)
    return out


def build(method):
    """Matrix of shape (1 + N_CKPTS, N_LAYERS): row 0 = base, then ck1..ck8."""
    rows, names = [], []
    base = layer_auc("base")
    if base is None:
        raise SystemExit("missing inside_out_out/base/k_scores_layer.parquet")
    rows.append(base)
    names.append("Base")
    for ck in range(1, N_CKPTS + 1):
        v = layer_auc(f"{method}_ck{ck}")
        if v is None:
            print(f"  [skip] {method}_ck{ck} has no k_scores_layer.parquet")
            continue
        rows.append(v)
        names.append(f"ck{ck}")
    return np.vstack(rows), names


mats = {m: build(m) for m in METHODS}

# One colour scale across every method so the panels stay comparable.
allv = np.concatenate([mat.ravel() for mat, _ in mats.values()])
vmin, vmax = np.nanmin(allv), np.nanmax(allv)
print(f"Global colour scale: vmin={vmin:.4f} vmax={vmax:.4f}")

for method, (mat, names) in mats.items():
    label = LABELS.get(method, method)
    fig, ax = plt.subplots(figsize=(ICLR_TEXTWIDTH_IN, ICLR_TEXTWIDTH_IN * 0.62))
    im = ax.imshow(mat, aspect="auto", origin="upper", vmin=vmin, vmax=vmax,
                   cmap="cividis", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)

    ax.set_title(f"{label} — probe AUC over training checkpoints (bio, LR)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC-ROC")
    # Layer index is a discrete identifier, so every tick is labelled; rotating
    # them means each needs only its font height (~8pt) of horizontal room.
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels(range(1, N_LAYERS + 1), fontsize=8, rotation=90)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)

    stem = SAVE_DIR / f"heatmap_checkpoints_auc_lr_method_bio_{method}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem.name}.pdf ({mat.shape[0]} rows x {mat.shape[1]} layers)")
