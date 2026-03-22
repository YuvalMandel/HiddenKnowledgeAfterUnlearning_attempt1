#!/usr/bin/env python3
"""
pca_probe_viz_report.py — PCA visualization: base (pre) vs unlearning checkpoints (post).

For each training checkpoint 1-8 of a given method produces a 2-panel PNG:
  Left:  base model hidden states (fixed) in PCA 2D + LR decision boundary
  Right: checkpoint N hidden states in the same PCA space + its own LR boundary

Axes are fixed across all frames (global limits computed before plotting).
Also saves an animated GIF of all frames.

Paths read (from main pipeline):
  checkpoints/base_hs_test.npy
  checkpoints/sweep_<METHOD>/ck{1..8}/hs_test.npy
  data/wmdp_tf_pairs.csv        (for labels — test split)

Valid method names: GradDiff, RMU, RMU-LAT, RepNoise, ELM, RR, TAR, PB_J

Usage:
  python pca_probe_viz_report.py --method ELM
  python pca_probe_viz_report.py --method RMU --pca_basis post
  python pca_probe_viz_report.py --method ELM --layer_start 10 --layer_end 22
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR     = Path("checkpoints")
DATA_DIR           = Path("data")
WMDP_CSV_PATH      = DATA_DIR / "wmdp_tf_pairs.csv"

VALID_METHODS      = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
N_CHECKPOINTS      = 8
BAND_DEFAULT_START = 10
BAND_DEFAULT_END   = 22   # inclusive

COLORS = {1: "#e05b4b", 0: "#4b7be0"}   # red=True, blue=False
ALPHA  = 0.55
S      = 14
GRID_N = 300
MARGIN = 0.6   # axis padding beyond data range


# ── Data loading ──────────────────────────────────────────────────────────────

def load_labels_from_csv(csv_path: Path, split: str = "test") -> np.ndarray:
    labels = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            labels.append(1 if row["label"].strip().lower() == "true" else 0)
    return np.array(labels, dtype=np.int32)


def band_mean(hs: np.ndarray, start: int, end: int) -> np.ndarray:
    """hs: (N, L, H) → (N, H) mean over layers [start, end] inclusive."""
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
    return pca.transform(scaler.transform(X))


def fit_lr(Z: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(Z, y)
    acc = (clf.predict(Z) == y).mean()
    print(f"  LR train acc = {acc:.3f}")
    return clf


def global_limits(all_Z: list[np.ndarray], margin: float = MARGIN):
    """Compute x/y limits that cover all projected arrays."""
    x_min = min(Z[:, 0].min() for Z in all_Z) - margin
    x_max = max(Z[:, 0].max() for Z in all_Z) + margin
    y_min = min(Z[:, 1].min() for Z in all_Z) - margin
    y_max = max(Z[:, 1].max() for Z in all_Z) + margin
    return (x_min, x_max), (y_min, y_max)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_panel(ax, Z: np.ndarray, y: np.ndarray, clf: LogisticRegression,
                title: str, xlim: tuple, ylim: tuple):
    """Draw scatter + LR decision boundary with fixed axis limits."""
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], GRID_N),
        np.linspace(ylim[0], ylim[1], GRID_N),
    )
    proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    ax.contourf(xx, yy, proba, levels=[0, 0.5, 1],
                colors=["#c8d9f7", "#f7c8c8"], alpha=0.35)
    ax.contour(xx, yy, proba, levels=[0.5], colors=["#333333"],
               linewidths=1.2, linestyles="--")

    for val in [1, 0]:
        idx = y == val
        ax.scatter(Z[idx, 0], Z[idx, 1], color=COLORS[val],
                   alpha=ALPHA, s=S, linewidths=0)

    train_acc = (clf.predict(Z) == y).mean()
    ax.set_title(f"{title}\nLR acc = {train_acc:.3f}", fontsize=10)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=7)


def save_frame(Z_base, y_base, clf_base,
               Z_ck, y_ck, clf_ck,
               ck_num: int, out_path: Path,
               method: str, pca_basis: str,
               xlim: tuple, ylim: tuple):
    """Two-panel figure: left=base (pre), right=checkpoint (post)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    basis_label = f"PCA basis: {pca_basis}"
    fig.suptitle(
        f"{method} — hidden-state PCA  |  checkpoint {ck_num}  ({basis_label})",
        fontsize=12
    )

    _draw_panel(axes[0], Z_base, y_base, clf_base,
                "Pre-unlearning (base)", xlim, ylim)
    _draw_panel(axes[1], Z_ck,   y_ck,   clf_ck,
                f"Post-unlearning (ck {ck_num})", xlim, ylim)

    # Shared legend with background box
    patches = [
        mpatches.Patch(color=COLORS[1], label="True (correct answer)"),
        mpatches.Patch(color=COLORS[0], label="False (wrong answer)"),
    ]
    leg = fig.legend(
        handles=patches, loc="lower center", ncol=2, fontsize=9,
        frameon=True, fancybox=True, framealpha=0.85,
        edgecolor="#aaaaaa", bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",      default="ELM", choices=VALID_METHODS,
                    help="Unlearning method (default: ELM)")
    ap.add_argument("--pca_basis",   default="pre", choices=["pre", "post"],
                    help="pre: PCA fitted on base model; "
                         "post: PCA fitted on last available checkpoint")
    ap.add_argument("--out_dir",     default=None,
                    help="Output directory (default: pca_viz_<METHOD>_<basis>)")
    ap.add_argument("--layer_start", type=int, default=BAND_DEFAULT_START)
    ap.add_argument("--layer_end",   type=int, default=BAND_DEFAULT_END)
    ap.add_argument("--split",       default="test")
    ap.add_argument("--gif",         action="store_true", default=True)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = f"pca_viz_{args.method}_{args.pca_basis}"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Labels ────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading labels  (split={args.split})")
    y_all = load_labels_from_csv(WMDP_CSV_PATH, args.split)
    print(f"  {y_all.sum()} True  {(y_all==0).sum()} False  total={len(y_all)}")

    # ── Load all hidden states ────────────────────────────────────────────────
    print("\n[2/4] Loading hidden states")
    base_hs = load_hs(CHECKPOINT_DIR / "base_hs_test.npy")
    if base_hs is None:
        raise FileNotFoundError("base_hs_test.npy not found in checkpoints/")

    N_base = len(base_hs)
    X_base = band_mean(base_hs, args.layer_start, args.layer_end)

    ck_data = []   # list of (ck_num, X_ck, y_ck)
    for ck in range(1, N_CHECKPOINTS + 1):
        hs = load_hs(CHECKPOINT_DIR / f"sweep_{args.method}" / f"ck{ck}" / "hs_test.npy")
        if hs is None:
            continue
        X_ck = band_mean(hs, args.layer_start, args.layer_end)
        y_ck = y_all[:len(hs)]
        ck_data.append((ck, X_ck, y_ck))

    if not ck_data:
        raise FileNotFoundError(f"No sweep checkpoints found for {args.method}")

    # ── Fit PCA ───────────────────────────────────────────────────────────────
    print(f"\n[3/4] Fitting PCA  (basis={args.pca_basis}, "
          f"layers {args.layer_start}–{args.layer_end} mean)")

    if args.pca_basis == "pre":
        fit_X = X_base
    else:   # "post" — use last available checkpoint
        fit_X = ck_data[-1][1]
        print(f"  PCA fitted on ck{ck_data[-1][0]} (last checkpoint)")

    y_base = y_all[:N_base]
    Z_base, scaler, pca = fit_pca_scaler(fit_X)

    if args.pca_basis == "pre":
        # Z_base already computed above
        pass
    else:
        Z_base = project(X_base, scaler, pca)

    clf_base = fit_lr(Z_base, y_base)

    # Project all checkpoints
    projected = []   # (ck_num, Z_ck, y_ck, clf_ck)
    for ck_num, X_ck, y_ck in ck_data:
        Z_ck = project(X_ck, scaler, pca)
        clf_ck = fit_lr(Z_ck, y_ck)
        projected.append((ck_num, Z_ck, y_ck, clf_ck))

    # ── Global axis limits (fixed across all frames) ──────────────────────────
    all_Z = [Z_base] + [Z_ck for _, Z_ck, _, _ in projected]
    xlim, ylim = global_limits(all_Z)
    print(f"  Global axis limits: x={xlim}  y={ylim}")

    # ── Generate frames ───────────────────────────────────────────────────────
    print(f"\n[4/4] Generating frames")
    frame_paths = []

    for ck_num, Z_ck, y_ck, clf_ck in projected:
        N_common = min(N_base, len(Z_ck))
        frame_path = out / f"frame_ck{ck_num:02d}.png"
        save_frame(
            Z_base[:N_common], y_base[:N_common], clf_base,
            Z_ck[:N_common],   y_ck[:N_common],   clf_ck,
            ck_num, frame_path,
            method=args.method, pca_basis=args.pca_basis,
            xlim=xlim, ylim=ylim,
        )
        frame_paths.append(frame_path)

    # ── GIF ───────────────────────────────────────────────────────────────────
    if args.gif and frame_paths:
        try:
            from PIL import Image
            imgs = [Image.open(p) for p in frame_paths]
            gif_path = out / f"{args.method}_{args.pca_basis}_sweep.gif"
            imgs[0].save(
                gif_path, save_all=True, append_images=imgs[1:],
                loop=0, duration=800,
            )
            print(f"\nGIF saved → {gif_path}")
        except ImportError:
            print("\n[warn] Pillow not installed — skipping GIF. pip install Pillow")

    print(f"\nDone. {len(frame_paths)} frames in {out}/")


if __name__ == "__main__":
    main()
