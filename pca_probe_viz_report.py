#!/usr/bin/env python3
"""
pca_probe_viz_report.py — PCA visualization: base (pre) vs unlearning checkpoints (post).

For each training checkpoint 1-8 of a given method produces a 2-panel PNG:
  Left:  base model hidden states in PCA 2D + selected probe boundaries
  Right: checkpoint N hidden states in the same PCA space + probe boundaries

Axes are fixed across all frames.  Saves an animated GIF of all frames.

Probe boundaries are shown by inverting the PCA transform back to original
feature space and running each probe's predict_proba on the grid.
  - Per-layer probe: exact (probe and viz share the same H-dim space).
  - Multi-layer probes (mid_band, full_layer, etc.): approximate — the
    inverted band-mean vector is tiled to reconstruct the concatenated
    feature space expected by the probe.

Accuracy printed in the subtitle is always computed on the actual (non-
approximated) test hidden states.

Paths read:
  checkpoints/base_hs_test.npy
  checkpoints/base_probes.pkl
  checkpoints/sweep_<METHOD>/ck{N}/hs_test.npy
  checkpoints/sweep_<METHOD>/ck{N}/probes.pkl
  data/wmdp_tf_pairs.csv

Usage:
  python pca_probe_viz_report.py --method ELM
  python pca_probe_viz_report.py --method ELM --pca_basis post
  python pca_probe_viz_report.py --method ELM --probes per_layer mid_band full_layer
  python pca_probe_viz_report.py --method ELM --probes all
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
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR     = Path("checkpoints")
DATA_DIR           = Path("data")
WMDP_CSV_PATH      = DATA_DIR / "wmdp_tf_pairs.csv"

VALID_METHODS      = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
N_CHECKPOINTS      = 8
BAND_DEFAULT_START = 10
BAND_DEFAULT_END   = 22

# ── Probe config ──────────────────────────────────────────────────────────────
# Order determines draw order (last = on top).
# "range_key": key in probe_set for the (start, end) layer tuple.
# "dict_key":  key in probe_set for the {clf_name: pipe} dict.
# "per_layer": if True, probe operates on hs[:, layer, :] (single layer, exact inversion).
#              if False, probe operates on hs[:, s:e+1, :].reshape(N,-1) (tiled approx).
PROBE_CONFIGS = {
    "init_band":     {"color": "#CC6600", "label": "Init-band LR (1–9)",
                      "dict_key": "init_band",     "range_key": "init_band_range",
                      "per_layer": False},
    "init_band_emb": {"color": "#888888", "label": "Init-band+emb LR (0–9)",
                      "dict_key": "init_band_emb", "range_key": "init_band_emb_range",
                      "per_layer": False},
    "ib_no_pca":     {"color": "#FF66AA", "label": "Init-band LR no-PCA (1–6)",
                      "dict_key": "ib_no_pca",     "range_key": "ib_no_pca_range",
                      "per_layer": False},
    "end_band":      {"color": "#009999", "label": "End-band LR (23+)",
                      "dict_key": "end_band",       "range_key": "end_band_range",
                      "per_layer": False},
    "full_layer":    {"color": "#9933CC", "label": "Full-layer LR (all)",
                      "dict_key": "full_layer",     "range_key": "full_layer_range",
                      "per_layer": False},
    "mid_band":      {"color": "#FF6600", "label": "Mid-band LR (10–22)",
                      "dict_key": "mid_band",       "range_key": "mid_band_range",
                      "per_layer": False},
    "per_layer":     {"color": "#228B22", "label": "Per-layer LR (best layer)",
                      "dict_key": "per_layer",      "range_key": None,
                      "per_layer": True},
}
ALL_PROBE_KEYS   = list(PROBE_CONFIGS.keys())
DEFAULT_PROBES   = ["per_layer"]

COLORS = {1: "#e05b4b", 0: "#4b7be0"}
ALPHA  = 0.55
S      = 14
GRID_N = 300
MARGIN = 0.6


# ── Data helpers ──────────────────────────────────────────────────────────────

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
        return pickle.load(f)


def band_mean(hs: np.ndarray, start: int, end: int) -> np.ndarray:
    return hs[:, start: end + 1, :].mean(axis=1)


# ── Probe feature extraction ──────────────────────────────────────────────────

def get_probe_pipe(probe_set: dict, probe_key: str, best_layer: int):
    """Return the LR pipeline for a given probe type, or None if missing."""
    cfg = PROBE_CONFIGS[probe_key]
    dk  = cfg["dict_key"]
    if dk not in probe_set:
        return None
    d = probe_set[dk]
    if cfg["per_layer"]:
        return d.get("LR", {}).get(best_layer)
    return d.get("LR")


def get_probe_features(hs: np.ndarray, probe_set: dict,
                       probe_key: str, best_layer: int):
    """
    Return (X_feat, n_tile) where:
      X_feat  — features as the probe was trained on  (N, D)
      n_tile  — how many copies of the band-mean vec were concatenated
                (1 for per-layer, >1 for multi-layer)
    """
    cfg = PROBE_CONFIGS[probe_key]
    if cfg["per_layer"]:
        return hs[:, best_layer, :], 1
    rk = cfg["range_key"]
    s, e = probe_set[rk]
    n = e - s + 1
    return hs[:, s: e + 1, :].reshape(len(hs), -1), n


# ── PCA helpers ───────────────────────────────────────────────────────────────

def fit_pca_scaler(X: np.ndarray, n_components: int = 2):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    Z = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_
    print(f"  PCA var explained: PC1={evr[0]:.3f}  PC2={evr[1]:.3f}  total={evr.sum():.3f}")
    return Z, scaler, pca


def project(X, scaler, pca):
    return pca.transform(scaler.transform(X))


def global_limits(all_Z: list, margin: float = MARGIN):
    return (
        (min(Z[:, 0].min() for Z in all_Z) - margin,
         max(Z[:, 0].max() for Z in all_Z) + margin),
        (min(Z[:, 1].min() for Z in all_Z) - margin,
         max(Z[:, 1].max() for Z in all_Z) + margin),
    )


def grid_proba_via_probe(probe_pipe, xx, yy,
                          viz_scaler, viz_pca, n_tile: int = 1) -> np.ndarray:
    """
    Evaluate probe on a 2D grid by inverting the visualization transform.
    Per-layer probes (n_tile=1): exact.
    Multi-layer probes (n_tile>1): approximate — band-mean vector is tiled
    to reconstruct the concatenated feature space.
    """
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    x_scaled = viz_pca.inverse_transform(grid_2d)
    x_orig   = viz_scaler.inverse_transform(x_scaled)
    if n_tile > 1:
        x_orig = np.tile(x_orig, n_tile)
    return probe_pipe.predict_proba(x_orig)[:, 1].reshape(xx.shape)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_panel(ax, Z: np.ndarray, y: np.ndarray, title: str,
                xlim: tuple, ylim: tuple,
                probe_entries: list,   # list of (probe_pipe, X_orig, n_tile, color, label)
                viz_scaler, viz_pca):
    """
    Draw scatter + one boundary + shading per probe entry.
    probe_entries is drawn in order; last entry's shading wins.
    """
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], GRID_N),
        np.linspace(ylim[0], ylim[1], GRID_N),
    )

    # Draw probes back-to-front (per_layer on top)
    for probe_pipe, X_orig, n_tile, color, _ in probe_entries:
        if probe_pipe is None:
            continue
        proba = grid_proba_via_probe(probe_pipe, xx, yy, viz_scaler, viz_pca, n_tile)
        ax.contourf(xx, yy, proba, levels=[0, 0.5, 1],
                    colors=["#c8d9f7", "#f7c8c8"], alpha=0.20)
        ax.contour(xx, yy, proba, levels=[0.5],
                   colors=[color], linewidths=1.6, linestyles="-", zorder=3)

    # Scatter
    for val in [1, 0]:
        idx = y == val
        ax.scatter(Z[idx, 0], Z[idx, 1], color=COLORS[val],
                   alpha=ALPHA, s=S, linewidths=0, zorder=5)

    # Accuracy subtitle (one line per probe)
    acc_lines = []
    for probe_pipe, X_orig, _, _, label in probe_entries:
        if probe_pipe is None or X_orig is None:
            continue
        acc = (probe_pipe.predict(X_orig) == y[:len(X_orig)]).mean()
        acc_lines.append(f"{label}: {acc:.3f}")

    subtitle = "  |  ".join(acc_lines) if acc_lines else ""
    ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=7)


def save_frame(Z_base, y_base, Z_ck, y_ck,
               ck_num: int, out_path: Path,
               method: str, pca_basis: str,
               xlim: tuple, ylim: tuple,
               probe_entries_base: list,
               probe_entries_ck: list,
               viz_scaler, viz_pca):

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{method} — hidden-state PCA  |  checkpoint {ck_num}  (PCA basis: {pca_basis})",
        fontsize=12,
    )

    _draw_panel(axes[0], Z_base, y_base,
                "Pre-unlearning (base)",
                xlim, ylim, probe_entries_base, viz_scaler, viz_pca)

    _draw_panel(axes[1], Z_ck, y_ck,
                f"Post-unlearning (ck {ck_num})",
                xlim, ylim, probe_entries_ck, viz_scaler, viz_pca)

    # Legend
    patches = [
        mpatches.Patch(color=COLORS[1], label="True (correct answer)"),
        mpatches.Patch(color=COLORS[0], label="False (wrong answer)"),
    ]
    shown_labels = set()
    for entries in [probe_entries_base, probe_entries_ck]:
        for pipe, _, _, color, label in entries:
            if pipe is not None and label not in shown_labels:
                patches.append(mpatches.Patch(color=color, label=label))
                shown_labels.add(label)

    fig.legend(
        handles=patches, loc="lower center", ncol=min(len(patches), 4),
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
    ap.add_argument("--method",      default="ELM", choices=VALID_METHODS)
    ap.add_argument("--pca_basis",   default="pre", choices=["pre", "post"],
                    help="pre: PCA on base model; post: PCA on last checkpoint")
    ap.add_argument("--probes",      nargs="+", default=DEFAULT_PROBES,
                    help=f"Probe types to overlay. Use 'all' for all types. "
                         f"Choices: {ALL_PROBE_KEYS}")
    ap.add_argument("--out_dir",     default=None)
    ap.add_argument("--layer_start", type=int, default=BAND_DEFAULT_START,
                    help="Band-mean start layer for PCA visualization")
    ap.add_argument("--layer_end",   type=int, default=BAND_DEFAULT_END)
    ap.add_argument("--split",       default="test")
    ap.add_argument("--gif",         action="store_true", default=True)
    args = ap.parse_args()

    # Resolve "all" shorthand
    if args.probes == ["all"]:
        args.probes = ALL_PROBE_KEYS
    for p in args.probes:
        if p not in PROBE_CONFIGS:
            ap.error(f"Unknown probe type '{p}'. Choices: {ALL_PROBE_KEYS}")

    probe_suffix = "_".join(args.probes) if len(args.probes) <= 3 else "multi"
    if args.out_dir is None:
        args.out_dir = f"pca_viz_{args.method}_{args.pca_basis}_{probe_suffix}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Labels ────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading labels  (split={args.split})")
    y_all = load_labels_from_csv(WMDP_CSV_PATH, args.split)
    print(f"  {y_all.sum()} True  {(y_all==0).sum()} False  total={len(y_all)}")

    # ── Base hidden states + probes ───────────────────────────────────────────
    print("\n[2/4] Loading base hidden states and probes")
    base_hs = load_hs(CHECKPOINT_DIR / "base_hs_test.npy")
    if base_hs is None:
        raise FileNotFoundError("base_hs_test.npy not found in checkpoints/")
    N_base   = len(base_hs)
    y_base   = y_all[:N_base]
    X_viz_base = band_mean(base_hs, args.layer_start, args.layer_end)

    probe_set_base = load_probe_set(CHECKPOINT_DIR / "base_probes.pkl")
    best_layer = probe_set_base["best_layers"]["LR"] if probe_set_base else 0
    print(f"  Best per-layer LR layer: {best_layer}")

    # ── Sweep checkpoints ─────────────────────────────────────────────────────
    ck_data = []   # (ck_num, hs, y_ck, probe_set_ck)
    for ck in range(1, N_CHECKPOINTS + 1):
        hs = load_hs(CHECKPOINT_DIR / f"sweep_{args.method}" / f"ck{ck}" / "hs_test.npy")
        if hs is None:
            continue
        ps = load_probe_set(
            CHECKPOINT_DIR / f"sweep_{args.method}" / f"ck{ck}" / "probes.pkl"
        )
        ck_data.append((ck, hs, y_all[:len(hs)], ps))

    if not ck_data:
        raise FileNotFoundError(f"No sweep checkpoints found for {args.method}")

    # ── Fit visualization PCA ─────────────────────────────────────────────────
    print(f"\n[3/4] Fitting visualization PCA  "
          f"(basis={args.pca_basis}, band mean layers {args.layer_start}–{args.layer_end})")
    if args.pca_basis == "pre":
        fit_X = X_viz_base
    else:
        fit_X = band_mean(ck_data[-1][1], args.layer_start, args.layer_end)
        print(f"  PCA fitted on ck{ck_data[-1][0]} (last checkpoint)")

    Z_base, viz_scaler, viz_pca = fit_pca_scaler(fit_X)
    if args.pca_basis != "pre":
        Z_base = project(X_viz_base, viz_scaler, viz_pca)

    projected = []   # (ck_num, Z_ck, hs_ck, y_ck, probe_set_ck)
    for ck_num, hs, y_ck, ps in ck_data:
        Z_ck = project(band_mean(hs, args.layer_start, args.layer_end), viz_scaler, viz_pca)
        projected.append((ck_num, Z_ck, hs, y_ck, ps))

    all_Z  = [Z_base] + [Z_ck for _, Z_ck, _, _, _ in projected]
    xlim, ylim = global_limits(all_Z)
    print(f"  Global axis limits: x={xlim}  y={ylim}")

    # ── Helper: build probe entries list ──────────────────────────────────────
    def build_entries(hs_arr, probe_set, y):
        entries = []
        for pk in args.probes:
            cfg   = PROBE_CONFIGS[pk]
            pipe  = get_probe_pipe(probe_set, pk, best_layer) if probe_set else None
            if pipe is None:
                print(f"  [warn] probe '{pk}' not found in probe_set")
            X_f, n_tile = (get_probe_features(hs_arr, probe_set, pk, best_layer)
                           if probe_set and pipe is not None else (None, 1))
            entries.append((pipe, X_f, n_tile, cfg["color"], cfg["label"]))
        return entries

    # ── Generate frames ───────────────────────────────────────────────────────
    print(f"\n[4/4] Generating frames  (probes: {args.probes})")
    frame_paths = []

    entries_base = build_entries(base_hs, probe_set_base, y_base)

    for ck_num, Z_ck, hs_ck, y_ck, ps_ck in projected:
        N_common = min(N_base, len(Z_ck))
        entries_ck = build_entries(hs_ck, ps_ck, y_ck)

        frame_path = out / f"frame_ck{ck_num:02d}.png"
        save_frame(
            Z_base[:N_common], y_base[:N_common],
            Z_ck[:N_common],   y_ck[:N_common],
            ck_num, frame_path,
            method=args.method, pca_basis=args.pca_basis,
            xlim=xlim, ylim=ylim,
            probe_entries_base=entries_base,
            probe_entries_ck=entries_ck,
            viz_scaler=viz_scaler,
            viz_pca=viz_pca,
        )
        frame_paths.append(frame_path)

    # ── GIF ───────────────────────────────────────────────────────────────────
    if args.gif and frame_paths:
        try:
            from PIL import Image
            imgs = [Image.open(p) for p in frame_paths]
            gif_path = out / f"{args.method}_{args.pca_basis}_sweep.gif"
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                         loop=0, duration=800)
            print(f"\nGIF saved → {gif_path}")
        except ImportError:
            print("\n[warn] Pillow not installed — pip install Pillow")

    print(f"\nDone. {len(frame_paths)} frames in {out}/")


if __name__ == "__main__":
    main()
