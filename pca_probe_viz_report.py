#!/usr/bin/env python3
"""
pca_probe_viz_report.py — PCA visualization: base (pre) vs unlearning checkpoints (post).

For each training checkpoint 1-8 of a given method produces a 2-panel PNG:
  Left:  base model hidden states (fixed) in PCA 2D + LR decision boundary
  Right: checkpoint N hidden states in the same PCA space + its own LR boundary

Axes are fixed across all frames (global limits computed before plotting).
Also saves an animated GIF of all frames.

--show_probe:
  Also overlays the original pipeline probe (per-layer LR from the main
  training run) on each panel.  The probe boundary is computed by inverting
  the visualization transform (PCA-2 → original feature space) and running
  probe.predict_proba on each grid point, so the probe "experiences" the
  same projection as the dots.  The visualization automatically switches to
  single-layer features (the probe's best layer) so both are consistent.

Paths read (from main pipeline):
  checkpoints/base_hs_test.npy
  checkpoints/base_probes.pkl               (when --show_probe)
  checkpoints/sweep_<METHOD>/ck{N}/hs_test.npy
  checkpoints/sweep_<METHOD>/ck{N}/probes.pkl  (when --show_probe)
  data/wmdp_tf_pairs.csv

Valid method names: GradDiff, RMU, RMU-LAT, RepNoise, ELM, RR, TAR, PB_J

Usage:
  python pca_probe_viz_report.py --method ELM
  python pca_probe_viz_report.py --method ELM --show_probe
  python pca_probe_viz_report.py --method ELM --show_probe --probe_layer 18
  python pca_probe_viz_report.py --method RMU --pca_basis post
"""

import argparse
import csv
import pickle
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
MARGIN = 0.6


# ── Data loading ──────────────────────────────────────────────────────────────

def load_labels_from_csv(csv_path: Path, split: str = "test") -> np.ndarray:
    labels = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            labels.append(1 if row["label"].strip().lower() == "true" else 0)
    return np.array(labels, dtype=np.int32)


def load_hs(path: Path) -> np.ndarray | None:
    if not path.exists():
        print(f"  [warn] Missing: {path}")
        return None
    hs = np.load(path)
    print(f"  Loaded {path.name}  shape={hs.shape}")
    return hs


def load_probe_set(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [warn] Missing probe: {path}")
        return None
    with open(path, "rb") as f:
        ps = pickle.load(f)
    return ps


def extract_features(hs: np.ndarray, layer: int | None,
                     band_start: int, band_end: int) -> np.ndarray:
    """Single layer if layer is set, otherwise band mean."""
    if layer is not None:
        return hs[:, layer, :]
    return hs[:, band_start: band_end + 1, :].mean(axis=1)


# ── PCA + LR helpers ──────────────────────────────────────────────────────────

def fit_pca_scaler(X: np.ndarray, n_components: int = 2):
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


def global_limits(all_Z: list, margin: float = MARGIN):
    x_min = min(Z[:, 0].min() for Z in all_Z) - margin
    x_max = max(Z[:, 0].max() for Z in all_Z) + margin
    y_min = min(Z[:, 1].min() for Z in all_Z) - margin
    y_max = max(Z[:, 1].max() for Z in all_Z) + margin
    return (x_min, x_max), (y_min, y_max)


def grid_proba(clf, xx, yy) -> np.ndarray:
    """Evaluate clf.predict_proba on a meshgrid, returns reshaped array."""
    return clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)


def grid_proba_via_probe(probe_pipe, xx, yy, viz_scaler, viz_pca) -> np.ndarray:
    """
    Evaluate the pipeline probe on grid points by inverting the visualization
    transform.  Each (z1, z2) is mapped back to original feature space via
    PCA inverse → StandardScaler inverse, then probe.predict_proba is called.
    The probe has its own internal scaler so it handles the raw features.
    """
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    x_scaled_approx = viz_pca.inverse_transform(grid_2d)        # → scaled feature space
    x_orig_approx   = viz_scaler.inverse_transform(x_scaled_approx)  # → original feature space
    proba = probe_pipe.predict_proba(x_orig_approx)[:, 1]
    return proba.reshape(xx.shape)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_panel(ax, Z: np.ndarray, y: np.ndarray, clf_viz: LogisticRegression,
                title: str, xlim: tuple, ylim: tuple,
                probe_pipe=None, viz_scaler=None, viz_pca=None):
    """Draw scatter + visualization LR boundary, optionally + original probe boundary."""
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], GRID_N),
        np.linspace(ylim[0], ylim[1], GRID_N),
    )

    # ── Background shading: visualization LR ──────────────────────────────────
    proba_viz = grid_proba(clf_viz, xx, yy)
    ax.contourf(xx, yy, proba_viz, levels=[0, 0.5, 1],
                colors=["#c8d9f7", "#f7c8c8"], alpha=0.30)
    ax.contour(xx, yy, proba_viz, levels=[0.5],
               colors=["#333333"], linewidths=1.2, linestyles="--",
               zorder=3)

    # ── Original probe boundary (solid colored line) ───────────────────────────
    if probe_pipe is not None:
        proba_probe = grid_proba_via_probe(probe_pipe, xx, yy, viz_scaler, viz_pca)
        ax.contour(xx, yy, proba_probe, levels=[0.5],
                   colors=["#228B22"], linewidths=1.8, linestyles="-",
                   zorder=4)

    # ── Scatter ────────────────────────────────────────────────────────────────
    for val in [1, 0]:
        idx = y == val
        ax.scatter(Z[idx, 0], Z[idx, 1], color=COLORS[val],
                   alpha=ALPHA, s=S, linewidths=0, zorder=5)

    train_acc = (clf_viz.predict(Z) == y).mean()
    ax.set_title(f"{title}\nLR acc = {train_acc:.3f}", fontsize=10)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=7)


def save_frame(Z_base, y_base, clf_base_viz,
               Z_ck, y_ck, clf_ck_viz,
               ck_num: int, out_path: Path,
               method: str, pca_basis: str,
               xlim: tuple, ylim: tuple,
               probe_base=None, probe_ck=None,
               viz_scaler=None, viz_pca=None,
               show_probe: bool = False):

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    basis_label = f"PCA basis: {pca_basis}"
    fig.suptitle(
        f"{method} — hidden-state PCA  |  checkpoint {ck_num}  ({basis_label})",
        fontsize=12,
    )

    _draw_panel(axes[0], Z_base, y_base, clf_base_viz,
                "Pre-unlearning (base)", xlim, ylim,
                probe_pipe=probe_base if show_probe else None,
                viz_scaler=viz_scaler, viz_pca=viz_pca)

    _draw_panel(axes[1], Z_ck, y_ck, clf_ck_viz,
                f"Post-unlearning (ck {ck_num})", xlim, ylim,
                probe_pipe=probe_ck if show_probe else None,
                viz_scaler=viz_scaler, viz_pca=viz_pca)

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=COLORS[1], label="True (correct answer)"),
        mpatches.Patch(color=COLORS[0], label="False (wrong answer)"),
        mpatches.Patch(color="#333333", label="Viz LR boundary (dashed)"),
    ]
    if show_probe:
        patches.append(
            mpatches.Patch(color="#228B22", label="Original pipeline probe boundary")
        )
    fig.legend(
        handles=patches, loc="lower center", ncol=len(patches),
        fontsize=8, frameon=True, fancybox=True, framealpha=0.85,
        edgecolor="#aaaaaa", bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",       default="ELM", choices=VALID_METHODS)
    ap.add_argument("--pca_basis",    default="pre", choices=["pre", "post"],
                    help="pre: PCA on base; post: PCA on last checkpoint")
    ap.add_argument("--show_probe",   action="store_true",
                    help="Overlay original pipeline probe boundary (green solid line)")
    ap.add_argument("--probe_layer",  type=int, default=None,
                    help="Layer to use for probe + visualization when --show_probe "
                         "(default: auto from base probe best_layers[LR])")
    ap.add_argument("--out_dir",      default=None)
    ap.add_argument("--layer_start",  type=int, default=BAND_DEFAULT_START)
    ap.add_argument("--layer_end",    type=int, default=BAND_DEFAULT_END)
    ap.add_argument("--split",        default="test")
    ap.add_argument("--gif",          action="store_true", default=True)
    args = ap.parse_args()

    suffix = "probe" if args.show_probe else args.pca_basis
    if args.out_dir is None:
        args.out_dir = f"pca_viz_{args.method}_{suffix}"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Labels ────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading labels  (split={args.split})")
    y_all = load_labels_from_csv(WMDP_CSV_PATH, args.split)
    print(f"  {y_all.sum()} True  {(y_all==0).sum()} False  total={len(y_all)}")

    # ── Probe setup ───────────────────────────────────────────────────────────
    probe_layer = None
    probe_set_base = None
    if args.show_probe:
        probe_set_base = load_probe_set(CHECKPOINT_DIR / "base_probes.pkl")
        if probe_set_base is None:
            raise FileNotFoundError("base_probes.pkl not found — needed for --show_probe")
        if args.probe_layer is not None:
            probe_layer = args.probe_layer
        else:
            probe_layer = probe_set_base["best_layers"]["LR"]
        print(f"\n  Probe: per-layer LR at layer {probe_layer} "
              f"(auto={'yes' if args.probe_layer is None else 'no'})")

    # ── Load all hidden states ────────────────────────────────────────────────
    print("\n[2/4] Loading hidden states")
    base_hs = load_hs(CHECKPOINT_DIR / "base_hs_test.npy")
    if base_hs is None:
        raise FileNotFoundError("base_hs_test.npy not found in checkpoints/")
    N_base = len(base_hs)

    X_base = extract_features(base_hs, probe_layer, args.layer_start, args.layer_end)
    feat_desc = (f"layer {probe_layer}" if probe_layer is not None
                 else f"layers {args.layer_start}–{args.layer_end} mean")
    print(f"  Features: {feat_desc}  shape={X_base.shape}")

    ck_data = []   # (ck_num, X_ck, y_ck, probe_pipe_ck_or_None)
    for ck in range(1, N_CHECKPOINTS + 1):
        hs = load_hs(CHECKPOINT_DIR / f"sweep_{args.method}" / f"ck{ck}" / "hs_test.npy")
        if hs is None:
            continue
        X_ck = extract_features(hs, probe_layer, args.layer_start, args.layer_end)
        y_ck = y_all[:len(hs)]

        ck_probe = None
        if args.show_probe:
            ps = load_probe_set(
                CHECKPOINT_DIR / f"sweep_{args.method}" / f"ck{ck}" / "probes.pkl"
            )
            if ps is not None and "per_layer" in ps and "LR" in ps["per_layer"]:
                ck_probe = ps["per_layer"]["LR"].get(probe_layer)
                if ck_probe is None:
                    print(f"  [warn] ck{ck} probe missing layer {probe_layer}")
            else:
                print(f"  [warn] ck{ck} probes.pkl missing or stale")

        ck_data.append((ck, X_ck, y_ck, ck_probe))

    if not ck_data:
        raise FileNotFoundError(f"No sweep checkpoints found for {args.method}")

    # ── Fit visualization PCA ─────────────────────────────────────────────────
    print(f"\n[3/4] Fitting visualization PCA  (basis={args.pca_basis}, {feat_desc})")
    if args.pca_basis == "pre":
        fit_X = X_base
    else:
        fit_X = ck_data[-1][1]
        print(f"  PCA fitted on ck{ck_data[-1][0]} (last checkpoint)")

    y_base = y_all[:N_base]
    Z_base, viz_scaler, viz_pca = fit_pca_scaler(fit_X)
    if args.pca_basis != "pre":
        Z_base = project(X_base, viz_scaler, viz_pca)

    clf_base_viz = fit_lr(Z_base, y_base)

    # Project all checkpoints
    projected = []   # (ck_num, Z_ck, y_ck, clf_ck_viz, probe_ck)
    for ck_num, X_ck, y_ck, ck_probe in ck_data:
        Z_ck = project(X_ck, viz_scaler, viz_pca)
        clf_ck_viz = fit_lr(Z_ck, y_ck)
        projected.append((ck_num, Z_ck, y_ck, clf_ck_viz, ck_probe))

    # ── Global axis limits ────────────────────────────────────────────────────
    all_Z = [Z_base] + [Z_ck for _, Z_ck, _, _, _ in projected]
    xlim, ylim = global_limits(all_Z)
    print(f"  Global axis limits: x={xlim}  y={ylim}")

    # ── Base probe pipeline ───────────────────────────────────────────────────
    probe_base_pipe = None
    if args.show_probe and probe_set_base is not None:
        probe_base_pipe = probe_set_base["per_layer"]["LR"].get(probe_layer)
        if probe_base_pipe is None:
            print(f"  [warn] Base probe missing layer {probe_layer} — skipping probe overlay")

    # ── Generate frames ───────────────────────────────────────────────────────
    print(f"\n[4/4] Generating frames")
    frame_paths = []

    for ck_num, Z_ck, y_ck, clf_ck_viz, ck_probe in projected:
        N_common = min(N_base, len(Z_ck))
        frame_path = out / f"frame_ck{ck_num:02d}.png"
        save_frame(
            Z_base[:N_common], y_base[:N_common], clf_base_viz,
            Z_ck[:N_common],   y_ck[:N_common],   clf_ck_viz,
            ck_num, frame_path,
            method=args.method, pca_basis=args.pca_basis,
            xlim=xlim, ylim=ylim,
            probe_base=probe_base_pipe,
            probe_ck=ck_probe,
            viz_scaler=viz_scaler,
            viz_pca=viz_pca,
            show_probe=args.show_probe,
        )
        frame_paths.append(frame_path)

    # ── GIF ───────────────────────────────────────────────────────────────────
    if args.gif and frame_paths:
        try:
            from PIL import Image
            imgs = [Image.open(p) for p in frame_paths]
            gif_path = out / f"{args.method}_{suffix}_sweep.gif"
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                         loop=0, duration=800)
            print(f"\nGIF saved → {gif_path}")
        except ImportError:
            print("\n[warn] Pillow not installed — skipping GIF. pip install Pillow")

    print(f"\nDone. {len(frame_paths)} frames in {out}/")


if __name__ == "__main__":
    main()
