#!/usr/bin/env python3
"""
pca_probe_viz_band_report.py

PCA probe visualization with multi-method / multi-checkpoint sweeps.

Single-run mode  (explicit paths):
  --base_hs        Base hidden states .npy, shape (N, L, D)
  --post_hs        Optional post hidden states (same shape/order)
  --tf_pairs_csv   CSV with label column (and optional split column)

Multi-run mode   (auto-resolve sweep paths):
  --methods        One or more method names, or "all"
                   Known: GradDiff RMU RMU-LAT RepNoise ELM RR TAR PB_J
  --checkpoints    Checkpoint numbers/ranges e.g. 1 2-4 8  or "all" (=1-8)

Shared options:
  --split / --layers / --band_mode / --pca_components / --max_pc_curve
  --out_dir        Base output dir (each run writes to its own subdir)
  --workers        Parallel worker processes (default: auto = nproc on SLURM)

Per-run outputs  (inside out_dir/<tag>/):
  pca_scatter_boundary__<tag>.png
  acc_vs_pc__<tag>.png
  truth_axis_hist__<tag>.png
  report__<tag>.json / .pdf

Combined outputs (inside out_dir/), created when >1 run:
  combined_scatter__<combined_tag>.png
  combined_acc_vs_pc__<combined_tag>.png
  combined_truth_axis_hist__<combined_tag>.png
  combined_centers_of_mass__<combined_tag>.png
  combined_trajectories__<combined_tag>.png   (only when >=2 checkpoints per method)

Marker convention throughout:
  × (x)  = False class
  ● (o)  = True  class
  Base   = gray markers + gray boundary
  Methods = rainbow-colored markers + colored boundary
"""

import argparse
import json
import math
import zipfile
import traceback
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------------------------------------------------------- #
#  Project layout                                                               #
# --------------------------------------------------------------------------- #
_SCRIPT_DIR     = Path(__file__).resolve().parent
_CHECKPOINT_DIR = _SCRIPT_DIR / "checkpoints"
_DATA_DIR       = _SCRIPT_DIR / "data"

ALL_METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
ALL_CKS     = list(range(1, 9))

# Number of pairs to show in trajectory plot (stratified by class)
N_TRAJ_PER_CLASS = 20


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #
def parse_ck_specs(specs: list) -> list:
    result = set()
    for s in specs:
        s = str(s).strip()
        if s == "all":
            result.update(ALL_CKS)
        elif "-" in s:
            a, b = s.split("-", 1)
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(s))
    return sorted(result)


def resolve_post_hs(method: str, ck: int) -> Path:
    return _CHECKPOINT_DIR / f"sweep_{method}" / f"ck{ck}" / "hs_train.npy"


def make_tag(method: str, ck: int, layers: str, band_mode: str, pca_k: int) -> str:
    return f"{method}_ck{ck}_layers{layers.replace('-','to')}_{band_mode}_pca{pca_k}"


def make_combined_tag(methods, cks, layers: str, band_mode: str, pca_k: int) -> str:
    m_str = "all" if sorted(methods) == ALL_METHODS else "-".join(methods)
    c_str = "all" if sorted(cks) == ALL_CKS else "-".join(str(c) for c in sorted(cks))
    return f"combined_m[{m_str}]_ck[{c_str}]_layers{layers.replace('-','to')}_{band_mode}_pca{pca_k}"


def load_hidden_states(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".npy":
        return np.load(p)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            npy_files = [f for f in z.namelist() if f.lower().endswith(".npy")]
            if len(npy_files) != 1:
                raise ValueError(f"Zip must contain exactly one .npy; found: {npy_files}")
            npy = npy_files[0]
            extract_dir = p.parent / (p.stem + "_extract")
            extract_dir.mkdir(exist_ok=True)
            z.extract(npy, extract_dir)
            return np.load(extract_dir / npy)
    raise ValueError(f"Unsupported format: {path}")


def parse_layers(spec: str):
    spec = str(spec).strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        a, b = int(a), int(b)
        if b < a:
            raise ValueError("layers range must be start<=end, e.g. 12-22")
        return list(range(a, b + 1))
    return [int(spec)]


def extract_features(hs: np.ndarray, layers, mode: str) -> np.ndarray:
    if hs.ndim != 3:
        raise ValueError(f"Expected (N,L,D), got {hs.shape}")
    if len(layers) == 1:
        return hs[:, layers[0], :]
    band = hs[:, layers, :]
    if mode == "mean":
        return band.mean(axis=1)
    if mode == "concat":
        N, bandL, D = band.shape
        return band.reshape(N, bandL * D)
    raise ValueError("band_mode must be concat or mean")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d == 0 else float(np.dot(a, b) / d)


def _build_mesh(all_X2_list):
    """Build a 300×300 meshgrid covering the union of all point clouds."""
    all_x = np.concatenate([X[:, 0] for X in all_X2_list])
    all_y = np.concatenate([X[:, 1] for X in all_X2_list])
    x0, x1 = np.percentile(all_x, 0.5), np.percentile(all_x, 99.5)
    y0, y1 = np.percentile(all_y, 0.5), np.percentile(all_y, 99.5)
    px = 0.05 * (x1 - x0 + 1e-9);  py = 0.05 * (y1 - y0 + 1e-9)
    xx, yy = np.meshgrid(np.linspace(x0 - px, x1 + px, 300),
                         np.linspace(y0 - py, y1 + py, 300))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def _spread(pts: np.ndarray) -> float:
    """Mean distance of points from their centroid (concentration proxy)."""
    if len(pts) < 2:
        return 1.0
    c = pts.mean(axis=0)
    return float(np.mean(np.linalg.norm(pts - c, axis=1))) or 1e-6


# --------------------------------------------------------------------------- #
#  Individual plot functions                                                    #
# --------------------------------------------------------------------------- #
def plot_boundary(X2, y, lr2, outpath, overlay=None, title="PCA-2D + LR boundary"):
    """Single-run scatter with LR boundary.
    Base = gray ×/● ; overlay (post) = colored ×/●
    """
    fig = plt.figure(figsize=(8.5, 6.5))
    m1, m0 = y == 1, y == 0
    # Base: gray, × for False, ● for True
    plt.scatter(X2[m0, 0], X2[m0, 1], s=14, alpha=0.5,
                color="gray", marker="x", label="Base: False")
    plt.scatter(X2[m1, 0], X2[m1, 1], s=14, alpha=0.5,
                color="gray", marker="o", label="Base: True")

    xx, yy, grid = _build_mesh([X2])
    zz = lr2.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, zz, levels=[0.5], linewidths=2, colors=["gray"])

    if overlay is not None:
        X2b, yb, lr2b, tag = overlay
        mb1, mb0 = yb == 1, yb == 0
        plt.scatter(X2b[mb0, 0], X2b[mb0, 1], s=12, alpha=0.35,
                    marker="x", label=f"{tag}: False")
        plt.scatter(X2b[mb1, 0], X2b[mb1, 1], s=12, alpha=0.35,
                    marker="o", label=f"{tag}: True")
        zz2 = lr2b.predict_proba(grid)[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, zz2, levels=[0.5], linewidths=2, linestyles="--")

    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=9); plt.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)


def plot_acc_vs_pc(Xp, y, max_k, outpath, title="Accuracy vs PC count (diagnostic)"):
    ks = list(range(1, max_k + 1))
    accs = []
    for k in ks:
        lr = LogisticRegression(max_iter=2000, solver="lbfgs")
        lr.fit(Xp[:, :k], y)
        accs.append(accuracy_score(y, lr.predict(Xp[:, :k])))
    fig = plt.figure(figsize=(8.5, 4.5))
    plt.plot(ks, accs); plt.xlabel("Number of PCs used")
    plt.ylabel("Train accuracy (LR)"); plt.title(title)
    plt.ylim(0.0, 1.0); plt.grid(True, alpha=0.25); plt.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)
    return ks, accs


def plot_truth_axis_hist(z, y, outpath, z_post=None, title="Truth-axis projection"):
    fig = plt.figure(figsize=(8.5, 4.8))
    bins = 40
    plt.hist(z[y == 0], bins=bins, alpha=0.6, density=True, label="Base: False")
    plt.hist(z[y == 1], bins=bins, alpha=0.6, density=True, label="Base: True")
    if z_post is not None:
        plt.hist(z_post[y == 0], bins=bins, alpha=0.25, density=True, histtype="step",
                 linewidth=2, linestyle="--", label="Post: False (on base axis)")
        plt.hist(z_post[y == 1], bins=bins, alpha=0.25, density=True, histtype="step",
                 linewidth=2, linestyle="--", label="Post: True (on base axis)")
    plt.title(title); plt.xlabel("Score z = w·x + b"); plt.ylabel("Density")
    plt.legend(loc="best", fontsize=9); plt.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)


def fig_text_page(lines, title="Run Summary"):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=14)
    fig.text(0.05, 0.95, "\n".join(lines), va="top", ha="left",
             family="monospace", fontsize=9)
    plt.axis("off")
    return fig


# --------------------------------------------------------------------------- #
#  Core: single-run analysis                                                    #
# --------------------------------------------------------------------------- #
def run_one(cfg: argparse.Namespace) -> dict:
    """Run full PCA probe analysis for one (base_hs, post_hs) config."""
    tag = getattr(cfg, "_tag", "run")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(cfg.layers)

    df = pd.read_csv(cfg.tf_pairs_csv)
    if "split" in df.columns:
        df = df[df["split"] == cfg.split]
    if df.empty:
        raise ValueError("No rows after split filtering.")
    y = (df["label"].astype(str).str.lower().str.strip() == "true").astype(int).to_numpy()

    hs_base = load_hidden_states(cfg.base_hs)
    if hs_base.shape[0] != len(y):
        raise ValueError(f"N mismatch: hs={hs_base.shape[0]} labels={len(y)}")
    X_base = extract_features(hs_base, layers, cfg.band_mode)

    K = int(cfg.pca_components)
    if K < 2:
        raise ValueError("--pca_components must be >= 2")
    if K > X_base.shape[1]:
        raise ValueError(f"--pca_components {K} > feature dim {X_base.shape[1]}")

    pca = PCA(n_components=K, random_state=42)
    Xp_base = pca.fit_transform(X_base)

    class_weight = "balanced" if cfg.lr_balanced else None
    lrK = LogisticRegression(max_iter=2000, C=cfg.lr_C,
                             class_weight=class_weight, random_state=42)
    lrK.fit(Xp_base, y)
    acc = float(accuracy_score(y, lrK.predict(Xp_base)))
    cm  = confusion_matrix(y, lrK.predict(Xp_base))

    X2_base = Xp_base[:, :2]
    lr2 = LogisticRegression(max_iter=2000, C=cfg.lr_C,
                             class_weight=class_weight, random_state=42)
    lr2.fit(X2_base, y)

    lr_hi = LogisticRegression(max_iter=2000, C=cfg.lr_C,
                               class_weight=class_weight, random_state=42)
    lr_hi.fit(X_base, y)
    w_pre = lr_hi.coef_.ravel()
    b_pre = float(lr_hi.intercept_.ravel()[0])
    z_base = X_base @ w_pre + b_pre

    overlay = None; dr = {}; z_post_proj = None; post_metrics = None; acc_post = None
    X2_post = None; lr2_post = None

    if cfg.post_hs:
        hs_post = load_hidden_states(cfg.post_hs)
        if hs_post.shape != hs_base.shape:
            raise ValueError(f"post_hs shape {hs_post.shape} != base {hs_base.shape}")
        X_post   = extract_features(hs_post, layers, cfg.band_mode)
        Xp_post  = pca.transform(X_post)
        X2_post  = Xp_post[:, :2]

        lrK_post = LogisticRegression(max_iter=2000, C=cfg.lr_C,
                                      class_weight=class_weight, random_state=42)
        lrK_post.fit(Xp_post, y)
        acc_post = float(accuracy_score(y, lrK_post.predict(Xp_post)))
        cm_post  = confusion_matrix(y, lrK_post.predict(Xp_post))

        lr2_post = LogisticRegression(max_iter=2000, C=cfg.lr_C,
                                      class_weight=class_weight, random_state=42)
        lr2_post.fit(X2_post, y)

        lr_hi_post = LogisticRegression(max_iter=2000, C=cfg.lr_C,
                                        class_weight=class_weight, random_state=42)
        lr_hi_post.fit(X_post, y)
        w_post = lr_hi_post.coef_.ravel()
        dr = {
            "dr_feature_space": 1.0 - cosine(w_pre, w_post),
            "dr_pca2_proxy":    1.0 - cosine(lr2.coef_.ravel(), lr2_post.coef_.ravel()),
        }
        overlay     = (X2_post, y, lr2_post, "Post")
        z_post_proj = X_post @ w_pre + b_pre
        post_metrics = {"pcaK_acc": acc_post, "pcaK_confusion": cm_post.tolist()}

    # individual PNGs
    p_scatter = out / f"pca_scatter_boundary__{tag}.png"
    plot_boundary(X2_base, y, lr2, p_scatter, overlay=overlay,
                  title=f"PCA-2D + LR boundary | {tag}")

    max_k = cfg.max_pc_curve
    if max_k is None:
        max_k = min(50, K)
    else:
        max_k = max(1, min(int(max_k), K))
    p_acc = out / f"acc_vs_pc__{tag}.png"
    ks, accs = plot_acc_vs_pc(Xp_base, y, max_k=max_k, outpath=p_acc,
                              title=f"Acc vs PC count | {tag}")

    p_hist = out / f"truth_axis_hist__{tag}.png"
    plot_truth_axis_hist(z_base, y, p_hist, z_post=z_post_proj,
                         title=f"Truth-axis projection | {tag}")

    report = {
        "tag": tag, "base_hs": cfg.base_hs, "post_hs": cfg.post_hs,
        "layers": layers, "layers_spec": cfg.layers, "band_mode": cfg.band_mode,
        "feature_shape": list(X_base.shape), "pca_components": K,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pcaK_acc": acc, "pcaK_confusion": cm.tolist(),
        "acc_vs_pc": {"max_k": int(max_k), "ks": ks, "acc": accs},
        "dr": dr, "post": post_metrics,
    }
    with open(out / f"report__{tag}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    evr = pca.explained_variance_ratio_
    pdf_lines = [
        f"PCA Probe Report  [{tag}]", "",
        f"base_hs : {cfg.base_hs}", f"post_hs : {cfg.post_hs or '(none)'}",
        f"csv     : {cfg.tf_pairs_csv}", f"split   : {cfg.split}", "",
        f"layers  : {cfg.layers}  ->  {layers}",
        f"band    : {cfg.band_mode}", f"feat    : {X_base.shape}  (N, F)", "",
        f"PCA K   : {K}", f"var PC1 : {evr[0]:.4f}", f"var PC2 : {evr[1]:.4f}",
        f"var 1+2 : {evr[0]+evr[1]:.4f}", "",
        f"LR@PCA(K) acc : {acc:.4f}", "Confusion [[TN,FP],[FN,TP]]:", str(cm),
    ]
    if post_metrics:
        pdf_lines += [
            "", f"POST acc : {acc_post:.4f}",
            "POST confusion:", str(np.array(post_metrics["pcaK_confusion"])), "",
            f"DR feat-space : {dr['dr_feature_space']:.4f}",
            f"DR PCA-2      : {dr['dr_pca2_proxy']:.4f}",
        ]
    pdf_path = out / f"report__{tag}.pdf"
    with PdfPages(pdf_path) as pdf:
        fig0 = fig_text_page(pdf_lines, title=f"Run Summary — {tag}")
        pdf.savefig(fig0); plt.close(fig0)
        for img_path, img_title in [
            (p_scatter, "PCA Scatter + Boundary"),
            (p_acc,     "Accuracy vs PC Count"),
            (p_hist,    "Truth-axis Projection Histogram"),
        ]:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle(img_title, fontsize=14)
            ax = fig.add_subplot(111); ax.axis("off")
            ax.imshow(plt.imread(img_path))
            pdf.savefig(fig); plt.close(fig)

    print(f"[done] {tag}  acc={acc:.4f}  ->  {out}")
    return {
        "tag":    tag,
        "method": getattr(cfg, "_method", None),
        "ck":     getattr(cfg, "_ck",     None),
        "pcaK_acc": acc, "acc_post": acc_post,
        "ks": ks, "accs": accs,
        "dr": dr,
        "X2_base":     X2_base,
        "X2_post":     X2_post,
        "lr2_base":    lr2,
        "lr2_post":    lr2_post,
        "z_base":      z_base,
        "z_post_proj": z_post_proj,
        "y":           y,
        "p_scatter": str(p_scatter),
        "p_acc":     str(p_acc),
        "p_hist":    str(p_hist),
        "out": str(out),
    }


# --------------------------------------------------------------------------- #
#  Combined figures                                                             #
# --------------------------------------------------------------------------- #
def make_combined_figures(results_ok: list, methods: list, cks: list,
                          base_out: Path, layers: str, band_mode: str, pca_k: int):
    if not results_ok:
        return

    base_out = Path(base_out)
    ctag = make_combined_tag(methods, cks, layers, band_mode, pca_k)

    # Sort for consistent ordering: method then checkpoint
    results_ok = sorted(results_ok, key=lambda r: (r.get("method") or "", r.get("ck") or 0))

    n = len(results_ok)
    cmap   = plt.get_cmap("rainbow")
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    ls_cycle = ["-", "--", "-.", ":"]

    def _color_ls(r, idx):
        c  = r.get("ck")
        ls = ls_cycle[(c - 1) % len(ls_cycle)] if isinstance(c, int) else "-"
        return colors[idx], ls

    # shared reference from first result
    r0     = results_ok[0]
    X2b    = r0["X2_base"]
    y0     = r0["y"]

    # meshgrid covering all data
    all_clouds = [r["X2_base"] for r in results_ok] + \
                 [r["X2_post"] for r in results_ok if r.get("X2_post") is not None]
    xx, yy, grid = _build_mesh(all_clouds)

    # base boundary probabilities (reused in several plots)
    zz_base = r0["lr2_base"].predict_proba(grid)[:, 1].reshape(xx.shape)

    # ------------------------------------------------------------------ #
    # 1. Combined scatter — shared PCA space                               #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 7))

    # Base cloud: gray × for False, gray ● for True
    ax.scatter(X2b[y0 == 0, 0], X2b[y0 == 0, 1],
               s=10, alpha=0.25, color="gray", marker="x", label="Base: False")
    ax.scatter(X2b[y0 == 1, 0], X2b[y0 == 1, 1],
               s=10, alpha=0.25, color="gray", marker="o", label="Base: True")

    # Base boundary: gray solid
    ax.contour(xx, yy, zz_base, levels=[0.5], linewidths=2.5, colors=["gray"])

    for idx, r in enumerate(results_ok):
        col, ls = _color_ls(r, idx)
        x2p  = r.get("X2_post")
        lr2p = r.get("lr2_post")
        if x2p is not None:
            y_r = r["y"]
            ax.scatter(x2p[y_r == 0, 0], x2p[y_r == 0, 1],
                       s=8, alpha=0.4, color=col, marker="x")
            ax.scatter(x2p[y_r == 1, 0], x2p[y_r == 1, 1],
                       s=8, alpha=0.4, color=col, marker="o")
        if lr2p is not None:
            zz = lr2p.predict_proba(grid)[:, 1].reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=[0.5], linewidths=1.5,
                       colors=[col], linestyles=[ls])
        acc_str = f"acc={r['pcaK_acc']:.3f}"
        if r.get("acc_post") is not None:
            acc_str += f" / post={r['acc_post']:.3f}"
        ax.plot([], [], color=col, linestyle=ls, linewidth=2,
                label=f"{r['tag']}  ({acc_str})")

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"PCA-2D Scatter + LR Boundaries  |  layers={layers}  {band_mode}  pca={pca_k}")
    ncol = max(1, math.ceil((n + 2) / 12))
    ax.legend(fontsize=7, ncol=ncol, loc="best")
    plt.tight_layout()
    out_scatter = base_out / f"combined_scatter__{ctag}.png"
    fig.savefig(out_scatter, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[combined] scatter           -> {out_scatter}")

    # ------------------------------------------------------------------ #
    # 2. Combined acc-vs-PC overlay                                        #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, r in enumerate(results_ok):
        ks_r, accs_r = r.get("ks"), r.get("accs")
        if not ks_r:
            continue
        col, ls = _color_ls(r, idx)
        ax.plot(ks_r, accs_r, color=col, linestyle=ls, linewidth=1.8,
                label=f"{r['tag']}  ({r['pcaK_acc']:.3f})")
    ax.set_xlabel("Number of PCs"); ax.set_ylabel("Train accuracy (LR)")
    ax.set_title(f"Acc vs PC count  |  layers={layers}  {band_mode}  pca={pca_k}")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.25)
    ncol = max(1, math.ceil(n / 14))
    ax.legend(fontsize=7, ncol=ncol, loc="lower right")
    plt.tight_layout()
    out_acc = base_out / f"combined_acc_vs_pc__{ctag}.png"
    fig.savefig(out_acc, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[combined] acc_vs_pc         -> {out_acc}")

    # ------------------------------------------------------------------ #
    # 3. Combined truth-axis histogram overlay                             #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = 50
    z0_b = r0["z_base"]
    ax.hist(z0_b[y0 == 0], bins=bins, alpha=0.3, density=True,
            color="silver", label="Base: False")
    ax.hist(z0_b[y0 == 1], bins=bins, alpha=0.3, density=True,
            color="gray",   label="Base: True")
    for idx, r in enumerate(results_ok):
        zp = r.get("z_post_proj")
        if zp is None:
            continue
        col = colors[idx]
        y_r = r["y"]
        ax.hist(zp[y_r == 0], bins=bins, density=True, histtype="step",
                linewidth=1.5, linestyle=":", color=col, alpha=0.8,
                label=f"{r['tag']} F")
        ax.hist(zp[y_r == 1], bins=bins, density=True, histtype="step",
                linewidth=1.5, linestyle="-", color=col, alpha=0.8,
                label=f"{r['tag']} T")
    ax.set_xlabel("Score z = w·x + b"); ax.set_ylabel("Density")
    ax.set_title(f"Truth-axis Projection  |  layers={layers}  {band_mode}  pca={pca_k}")
    ncol = max(1, math.ceil((2 * n + 2) / 18))
    ax.legend(fontsize=6, ncol=ncol, loc="best")
    plt.tight_layout()
    out_hist = base_out / f"combined_truth_axis_hist__{ctag}.png"
    fig.savefig(out_hist, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[combined] truth_hist        -> {out_hist}")

    # ------------------------------------------------------------------ #
    # 4. Centers of mass                                                   #
    #    Same boundary lines as scatter, but only centroids shown.        #
    #    Marker size ∝ concentration (1 / mean-distance-from-centroid).   #
    # ------------------------------------------------------------------ #
    # Pre-compute all spreads to build a global size normalization
    _all_spreads = []
    for _mask in [y0 == 0, y0 == 1]:
        _all_spreads.append(_spread(X2b[_mask]))
    for r in results_ok:
        x2p = r.get("X2_post")
        if x2p is None:
            continue
        y_r = r["y"]
        for _mask in [y_r == 0, y_r == 1]:
            _all_spreads.append(_spread(x2p[_mask]))

    _concs   = [1.0 / max(s, 1e-6) for s in _all_spreads]
    _c_min, _c_max = min(_concs), max(_concs)
    _c_rng   = max(_c_max - _c_min, 1e-9)

    def _com_size(pts):
        s = _spread(pts)
        conc = 1.0 / max(s, 1e-6)
        return 80 + 720 * (conc - _c_min) / _c_rng

    fig_com, ax_com = plt.subplots(figsize=(9, 7))

    # Base boundary: gray solid
    ax_com.contour(xx, yy, zz_base, levels=[0.5], linewidths=2.5, colors=["gray"])

    # Base centroids
    for _mask, _marker, _lbl in [(y0 == 0, "x", "Base: False"),
                                  (y0 == 1, "o", "Base: True")]:
        _pts = X2b[_mask]
        _c   = _pts.mean(axis=0)
        ax_com.scatter([_c[0]], [_c[1]], s=_com_size(_pts),
                       color="gray", marker=_marker, linewidths=2,
                       zorder=5, label=_lbl)

    for idx, r in enumerate(results_ok):
        col, ls = _color_ls(r, idx)
        x2p  = r.get("X2_post")
        lr2p = r.get("lr2_post")
        if x2p is None:
            continue
        y_r = r["y"]
        # Boundary
        if lr2p is not None:
            zz = lr2p.predict_proba(grid)[:, 1].reshape(xx.shape)
            ax_com.contour(xx, yy, zz, levels=[0.5], linewidths=1.5,
                           colors=[col], linestyles=[ls])
        # Centroids
        for _mask, _marker in [(y_r == 0, "x"), (y_r == 1, "o")]:
            _pts = x2p[_mask]
            _c   = _pts.mean(axis=0)
            ax_com.scatter([_c[0]], [_c[1]], s=_com_size(_pts),
                           color=col, marker=_marker, linewidths=2, zorder=5)
        # Legend proxy line
        ax_com.plot([], [], color=col, linestyle=ls, linewidth=2, label=r["tag"])

    ax_com.set_xlabel("PC1"); ax_com.set_ylabel("PC2")
    ax_com.set_title(
        f"Centers of Mass  |  layers={layers}  {band_mode}  pca={pca_k}\n"
        f"size ∝ concentration  ·  × = False  ·  ● = True"
    )
    ncol = max(1, math.ceil((n + 2) / 12))
    ax_com.legend(fontsize=7, ncol=ncol, loc="best")
    plt.tight_layout()
    out_com = base_out / f"combined_centers_of_mass__{ctag}.png"
    fig_com.savefig(out_com, dpi=150, bbox_inches="tight")
    plt.close(fig_com)
    print(f"[combined] centers_of_mass   -> {out_com}")

    # ------------------------------------------------------------------ #
    # 5. Trajectory arrows (only when >=2 checkpoints per method)         #
    #    base → ck1 → ck2 → … for a stratified subsample of pairs.       #
    # ------------------------------------------------------------------ #
    method_groups = defaultdict(list)
    for r in results_ok:
        m = r.get("method") or "?"
        if r.get("X2_post") is not None:
            method_groups[m].append(r)

    traj_methods = {
        m: sorted(rs, key=lambda r: r.get("ck") or 0)
        for m, rs in method_groups.items() if len(rs) >= 2
    }

    if traj_methods:
        # Stratified subsample — same indices for every method subplot
        rng_t     = np.random.default_rng(42)
        true_idx  = np.where(y0 == 1)[0]
        false_idx = np.where(y0 == 0)[0]
        n_each    = min(N_TRAJ_PER_CLASS, len(true_idx), len(false_idx))
        sample_idx = np.concatenate([
            rng_t.choice(true_idx,  n_each, replace=False),
            rng_t.choice(false_idx, n_each, replace=False),
        ])

        n_m   = len(traj_methods)
        fig_t, axes_t = plt.subplots(1, n_m, figsize=(7 * n_m, 6), squeeze=False)

        C_TRUE  = "steelblue"
        C_FALSE = "crimson"

        for col_idx, (method_name, method_runs) in enumerate(sorted(traj_methods.items())):
            ax_t  = axes_t[0][col_idx]
            y_m   = method_runs[0]["y"]

            # Meshgrid for this method's data
            traj_clouds = [X2b] + [r["X2_post"] for r in method_runs]
            xx_t, yy_t, grid_t = _build_mesh(traj_clouds)

            # Base boundary: gray solid
            zz_b = r0["lr2_base"].predict_proba(grid_t)[:, 1].reshape(xx_t.shape)
            ax_t.contour(xx_t, yy_t, zz_b, levels=[0.5],
                         linewidths=2, colors=["gray"])

            # Last checkpoint boundary: colored dashed
            last_r = method_runs[-1]
            if last_r.get("lr2_post") is not None:
                zz_last = last_r["lr2_post"].predict_proba(grid_t)[:, 1].reshape(xx_t.shape)
                # use the color of this run in the global palette
                last_idx = results_ok.index(last_r)
                last_col = colors[last_idx]
                ax_t.contour(xx_t, yy_t, zz_last, levels=[0.5],
                             linewidths=1.5, colors=[last_col], linestyles=["--"])

            # Draw trajectories for sampled pairs
            for j in sample_idx:
                c = C_TRUE if y_m[j] == 1 else C_FALSE
                traj = np.array(
                    [X2b[j]] + [r["X2_post"][j] for r in method_runs]
                )
                # Path line
                ax_t.plot(traj[:, 0], traj[:, 1],
                          color=c, alpha=0.2, lw=0.8, zorder=1)
                # Arrows at each step
                for step in range(len(traj) - 1):
                    ax_t.annotate(
                        "", xy=traj[step + 1], xytext=traj[step],
                        arrowprops=dict(arrowstyle="->", color=c,
                                        lw=0.7, alpha=0.45),
                    )
                # Base point: star; end point: square
                ax_t.scatter([traj[0, 0]], [traj[0, 1]],
                             s=35, color=c, marker="*", alpha=0.7, zorder=4)
                ax_t.scatter([traj[-1, 0]], [traj[-1, 1]],
                             s=20, color=c, marker="s", alpha=0.7, zorder=4)

            ck_labels = [r.get("ck") for r in method_runs]
            ax_t.set_title(f"{method_name}  ck {ck_labels[0]}→{ck_labels[-1]}", fontsize=10)
            ax_t.set_xlabel("PC1"); ax_t.set_ylabel("PC2")
            ax_t.grid(True, alpha=0.2)

            legend_elements = [
                Line2D([0], [0], color=C_TRUE,  lw=1.5, label="True"),
                Line2D([0], [0], color=C_FALSE, lw=1.5, label="False"),
                Line2D([0], [0], marker="*", color="gray", lw=0,
                       markersize=8, label="Base (start)"),
                Line2D([0], [0], marker="s", color="gray", lw=0,
                       markersize=6, label="Last ck (end)"),
                Line2D([0], [0], color="gray", lw=2, label="Base boundary"),
            ]
            ax_t.legend(handles=legend_elements, fontsize=8, loc="best")

        fig_t.suptitle(
            f"Trajectories base→ck  |  layers={layers}  {band_mode}  pca={pca_k}\n"
            f"(★=base, ■=last ck; {n_each} True + {n_each} False sampled)",
            fontsize=11,
        )
        plt.tight_layout()
        out_traj = base_out / f"combined_trajectories__{ctag}.png"
        fig_t.savefig(out_traj, dpi=150, bbox_inches="tight")
        plt.close(fig_t)
        print(f"[combined] trajectories      -> {out_traj}")


# --------------------------------------------------------------------------- #
#  Worker wrapper (module-level so it is picklable)                            #
# --------------------------------------------------------------------------- #
def _worker(cfg: argparse.Namespace):
    try:
        return run_one(cfg)
    except Exception as e:
        tag = getattr(cfg, "_tag", "?")
        print(f"[ERROR] {tag}: {e}")
        traceback.print_exc()
        return {"tag": tag, "error": str(e)}


# --------------------------------------------------------------------------- #
#  main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)

    # single-run paths
    ap.add_argument("--base_hs",      default=str(_CHECKPOINT_DIR / "base_hs_train.npy"))
    ap.add_argument("--post_hs",      default=None,
                    help="Explicit post hs path (single-run). Ignored if --methods given.")
    ap.add_argument("--tf_pairs_csv", default=str(_CHECKPOINT_DIR / "wmdp_tf_pairs.csv"))
    ap.add_argument("--split",        default="train")

    # multi-run sweep
    ap.add_argument("--methods", nargs="+", default=None, metavar="M",
                    help='"all" or names e.g. PB_J RMU GradDiff')
    ap.add_argument("--checkpoints", nargs="+", default=None, metavar="CK",
                    help='"all", numbers, or ranges e.g. 1 3-5 8')

    # analysis params
    ap.add_argument("--layers",         default="12-22")
    ap.add_argument("--band_mode",      default="concat", choices=["concat", "mean"])
    ap.add_argument("--pca_components", type=int,   default=40)
    ap.add_argument("--max_pc_curve",   type=int,   default=None)
    ap.add_argument("--lr_C",           type=float, default=1.0)
    ap.add_argument("--lr_balanced",    action="store_true")

    # output / parallelism
    ap.add_argument("--out_dir",  default=str(_SCRIPT_DIR / "pca_output"),
                    help="Base output dir. Each run writes to its own tagged subdir.")
    ap.add_argument("--workers",  type=int, default=1,
                    help="Parallel workers (default 1). Set >1 for multi-run sweeps.")

    args = ap.parse_args()

    configs = []
    sweep_methods = None
    sweep_cks     = None

    if args.methods or args.checkpoints:
        sweep_methods = args.methods if args.methods else ["PB_J"]
        if sweep_methods == ["all"]:
            sweep_methods = ALL_METHODS
        sweep_cks = parse_ck_specs(args.checkpoints) if args.checkpoints else [1]

        for method in sweep_methods:
            for ck in sweep_cks:
                tag = make_tag(method, ck, args.layers, args.band_mode, args.pca_components)
                cfg = argparse.Namespace(**vars(args))
                cfg.post_hs  = str(resolve_post_hs(method, ck))
                cfg.out_dir  = str(Path(args.out_dir) / tag)
                cfg._tag     = tag
                cfg._method  = method
                cfg._ck      = ck
                configs.append(cfg)
    else:
        cfg = argparse.Namespace(**vars(args))
        if args.post_hs:
            p = Path(args.post_hs)
            try:
                method = p.parts[-3].replace("sweep_", "")
                ck     = int(p.parts[-2].replace("ck", ""))
                tag    = make_tag(method, ck, args.layers, args.band_mode, args.pca_components)
                cfg._method, cfg._ck = method, ck
                sweep_methods, sweep_cks = [method], [ck]
            except Exception:
                tag = "run"
                cfg._method = cfg._ck = None
        else:
            tag = f"base_only_layers{args.layers.replace('-','to')}_{args.band_mode}_pca{args.pca_components}"
            cfg._method = cfg._ck = None
        cfg._tag    = tag
        cfg.out_dir = str(Path(args.out_dir) / tag)
        configs.append(cfg)

    print(f"[pca_probe] {len(configs)} run(s), workers={args.workers}")
    for c in configs:
        print(f"  -> {c._tag}")

    if args.workers > 1 and len(configs) > 1:
        n_w = min(args.workers, len(configs))
        print(f"\n[pca_probe] Parallel: {len(configs)} tasks x {n_w} workers\n")
        results = []
        with ProcessPoolExecutor(max_workers=n_w) as pool:
            futures = {pool.submit(_worker, cfg): cfg._tag for cfg in configs}
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        results = [_worker(cfg) for cfg in configs]

    ok  = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]

    if len(configs) > 1 and ok and sweep_methods and sweep_cks:
        print(f"\n[pca_probe] Building combined figures ({len(ok)} successful runs)...")
        make_combined_figures(ok, sweep_methods, sweep_cks,
                              Path(args.out_dir), args.layers,
                              args.band_mode, args.pca_components)

    print(f"\n[pca_probe] Done.  {len(ok)} OK, {len(err)} failed.")
    for r in ok:
        acc_str = f"acc={r['pcaK_acc']:.4f}"
        if r.get("acc_post") is not None:
            acc_str += f"  post={r['acc_post']:.4f}"
        print(f"  OK   {r['tag']}  {acc_str}")
    for r in err:
        print(f"  FAIL {r['tag']}  {r['error']}")


if __name__ == "__main__":
    main()
