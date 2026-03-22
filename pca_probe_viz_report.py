#!/usr/bin/env python3
"""
pca_probe_viz_report.py — PCA visualization: base (pre) vs ELM checkpoints (post).

For each ELM training checkpoint 1-8 produces a 2-panel PNG:
  Left:  base model hidden states (fixed) in PCA 2D + LR decision boundary
  Right: checkpoint N hidden states in the same PCA space + its own LR boundary

Also saves an animated GIF of all frames.

Paths read (from main pipeline):
  checkpoints/base_hs_test.npy
  checkpoints/sweep_ELM/ck{1..8}/hs_test.npy
  data/wmdp_tf_pairs.csv        (for labels — test split)

Usage:
  python pca_probe_viz_report.py [--out_dir pca_viz] [--layer_start 10] [--layer_end 22]
  python pca_probe_viz_report.py --out_dir pca_viz --layer_start 10 --layer_end 22
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR       = Path("data")
WMDP_CSV_PATH  = DATA_DIR / "wmdp_tf_pairs.csv"

N_CHECKPOINTS  = 8
BAND_DEFAULT_START = 10
BAND_DEFAULT_END   = 22   # inclusive


# ── Data loading ──────────────────────────────────────────────────────────────

def load_labels_from_csv(csv_path: Path, split: str = "test") -> np.ndarray:
    """Read labels (0/1) in CSV order for the given split."""
    labels = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            labels.append(1 if row["label"].strip().lower() == "true" else 0)
    return np.array(labels, dtype=np.int32)


def band_mean(hs: np.ndarray, start: int, end: int) -> np.ndarray:
    """Average hidden states over layers [start, end] inclusive. hs: (N, L, H) → (N, H)."""
    return hs[:, start: end + 1, :].mean(axis=1)


def load_hs(path: Path) -> np.ndarray | None:
    if not path.exists():
        print(f"  [warn] Missing: {path}")
        return None
    hs = np.load(path)
    print(f"  Loaded {path.name}  shape={hs.shape}")
    return hs


# ── PCA + LR helpers ──────────────────────────────────────────────────────────

def fit_pca_scaler(X: np.ndarray, n_components: int = 2):
    """Fit StandardScaler + PCA on X. Returns (Z, scaler, pca)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    Z = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_
    print(f"  PCA var explained: PC1={evr[0]:.3f}  PC2={evr[1]:.3f}  total={evr.sum():.3f}")
    return Z, scaler, pca


def project(X: np.ndarray, scaler, pca) -> np.ndarray:
    """Project new X into the fitted PCA space."""
    return pca.transform(scaler.transform(X))


def fit_lr(Z: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(Z, y)
    acc = (clf.predict(Z) == y).mean()
    print(f"  LR train acc = {acc:.3f}")
    return clf


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {1: "#e05b4b", 0: "#4b7be0"}   # red=True, blue=False
ALPHA  = 0.55
S      = 14
GRID_N = 300


def _draw_panel(ax, Z: np.ndarray, y: np.ndarray, clf: LogisticRegression,
                title: str, auc: float | None = None):
    """Draw scatter + LR decision boundary on ax."""
    # Decision boundary via meshgrid
    x_min, x_max = Z[:, 0].min() - 0.5, Z[:, 0].max() + 0.5
    y_min, y_max = Z[:, 1].min() - 0.5, Z[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, GRID_N),
        np.linspace(y_min, y_max, GRID_N),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    ax.contourf(xx, yy, proba, levels=[0, 0.5, 1],
                colors=["#c8d9f7", "#f7c8c8"], alpha=0.35)
    ax.contour(xx, yy, proba, levels=[0.5], colors=["#333333"],
               linewidths=1.2, linestyles="--")

    for val, name in [(1, "True"), (0, "False")]:
        idx = y == val
        ax.scatter(Z[idx, 0], Z[idx, 1], color=COLORS[val],
                   label=name, alpha=ALPHA, s=S, linewidths=0)

    train_acc = (clf.predict(Z) == y).mean()
    subtitle = f"LR acc = {train_acc:.2f}"
    if auc is not None:
        subtitle += f"  AUC = {auc:.2f}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper right")


def save_frame(Z_base, y_base, clf_base,
               Z_ck, y_ck, clf_ck,
               ck_num: int, out_path: Path):
    """Two-panel figure: left=base (pre), right=checkpoint (post)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"ELM — hidden-state PCA  |  checkpoint {ck_num}", fontsize=12)

    _draw_panel(axes[0], Z_base, y_base, clf_base, "Pre-unlearning (base)")
    _draw_panel(axes[1], Z_ck,   y_ck,   clf_ck,
                f"Post-unlearning (ck {ck_num})")

    # Shared legend patch
    patches = [mpatches.Patch(color=COLORS[1], label="True (correct answer)"),
               mpatches.Patch(color=COLORS[0], label="False (wrong answer)")]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",      default="pca_viz")
    ap.add_argument("--layer_start",  type=int, default=BAND_DEFAULT_START)
    ap.add_argument("--layer_end",    type=int, default=BAND_DEFAULT_END)
    ap.add_argument("--split",        default="test")
    ap.add_argument("--gif",          action="store_true", default=True,
                    help="Also save animated GIF (requires Pillow)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Labels ────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading labels from {WMDP_CSV_PATH} (split={args.split})")
    y_all = load_labels_from_csv(WMDP_CSV_PATH, args.split)
    print(f"  Labels: {y_all.sum()} True  {(y_all==0).sum()} False  total={len(y_all)}")

    # ── Base hidden states ────────────────────────────────────────────────────
    print("\n[2/4] Loading base hidden states")
    base_hs = load_hs(CHECKPOINT_DIR / "base_hs_test.npy")
    if base_hs is None:
        raise FileNotFoundError(f"base_hs_test.npy not found in {CHECKPOINT_DIR}/")

    # Align label count with hs count (test split)
    N = len(base_hs)
    y = y_all[:N]
    if len(y) != N:
        raise ValueError(f"Label count mismatch: hs has {N} rows, CSV has {len(y_all)} test labels")

    # ── Feature extraction + PCA fitted on base ───────────────────────────────
    print(f"\n[3/4] Fitting PCA on base  (band mean layers {args.layer_start}–{args.layer_end})")
    X_base = band_mean(base_hs, args.layer_start, args.layer_end)
    Z_base, scaler, pca = fit_pca_scaler(X_base)

    clf_base = fit_lr(Z_base, y)

    # ── Checkpoints ───────────────────────────────────────────────────────────
    print("\n[4/4] Generating frames")
    frame_paths = []

    for ck in range(1, N_CHECKPOINTS + 1):
        ck_hs = load_hs(CHECKPOINT_DIR / f"sweep_ELM" / f"ck{ck}" / "hs_test.npy")
        if ck_hs is None:
            print(f"  [skip] ck{ck} — hs_test.npy missing")
            continue

        N_ck = len(ck_hs)
        y_ck = y_all[:N_ck]

        X_ck = band_mean(ck_hs, args.layer_start, args.layer_end)
        Z_ck = project(X_ck, scaler, pca)   # same PCA space as base
        clf_ck = fit_lr(Z_ck, y_ck)

        # Re-project base into same axes (already Z_base, but recompute y alignment)
        y_base_ck = y_all[:min(N, N_ck)]
        Z_base_aligned = Z_base[:len(y_base_ck)]

        frame_path = out / f"frame_ck{ck:02d}.png"
        save_frame(Z_base_aligned, y_base_ck, clf_base,
                   Z_ck[:len(y_base_ck)], y_ck[:len(y_base_ck)], clf_ck,
                   ck, frame_path)
        frame_paths.append(frame_path)

    # ── GIF ───────────────────────────────────────────────────────────────────
    if args.gif and frame_paths:
        try:
            from PIL import Image
            imgs = [Image.open(p) for p in frame_paths]
            gif_path = out / "elm_sweep.gif"
            imgs[0].save(
                gif_path, save_all=True, append_images=imgs[1:],
                loop=0, duration=800,
            )
            print(f"\nGIF saved → {gif_path}")
        except ImportError:
            print("\n[warn] Pillow not installed — skipping GIF. Install with: pip install Pillow")

    print(f"\nDone. {len(frame_paths)} frames in {out}/")


if __name__ == "__main__":
    main()
