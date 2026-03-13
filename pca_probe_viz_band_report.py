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
  combined_scatter__<combined_tag>.png      — grid: rows=methods, cols=checkpoints
  combined_acc_vs_pc__<combined_tag>.png    — overlaid accuracy curves
  combined_truth_axis_hist__<combined_tag>.png — grid of histogram images
"""

import argparse
import json
import math
import zipfile
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")   # non-interactive; must be before other matplotlib imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# --------------------------------------------------------------------------- #
#  Individual plot functions                                                    #
# --------------------------------------------------------------------------- #
def plot_boundary(X2, y, lr2, outpath, overlay=None, title="PCA-2D + LR boundary"):
    fig = plt.figure(figsize=(8.5, 6.5))
    m1, m0 = y == 1, y == 0
    plt.scatter(X2[m0, 0], X2[m0, 1], s=12, alpha=0.6, label="Base: False")
    plt.scatter(X2[m1, 0], X2[m1, 1], s=12, alpha=0.6, label="Base: True")
    x_min, x_max = np.percentile(X2[:, 0], [0.5, 99.5])
    y_min, y_max = np.percentile(X2[:, 1], [0.5, 99.5])
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    xx, yy = np.meshgrid(np.linspace(x_min - pad_x, x_max + pad_x, 300),
                         np.linspace(y_min - pad_y, y_max + pad_y, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = lr2.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, zz, levels=[0.5], linewidths=2)
    if overlay is not None:
        X2b, yb, lr2b, tag = overlay
        mb1, mb0 = yb == 1, yb == 0
        plt.scatter(X2b[mb0, 0], X2b[mb0, 1], s=12, alpha=0.35, marker="x", label=f"{tag}: False")
        plt.scatter(X2b[mb1, 0], X2b[mb1, 1], s=12, alpha=0.35, marker="x", label=f"{tag}: True")
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
        "tag": tag,
        "method": getattr(cfg, "_method", None),
        "ck":     getattr(cfg, "_ck",     None),
        "pcaK_acc": acc, "acc_post": acc_post,
        "ks": ks, "accs": accs,
        "z_base":      z_base,
        "z_post_proj": z_post_proj,
        "y":           y,
        "dr": dr,
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
    """
    Create three combined figures from all completed runs:
      1. combined_scatter  — grid rows=methods, cols=checkpoints (imread)
      2. combined_acc_vs_pc — overlaid accuracy curves
      3. combined_truth_axis_hist — grid rows=methods, cols=checkpoints (imread)
    """
    if not results_ok:
        return

    base_out = Path(base_out)
    ctag = make_combined_tag(methods, cks, layers, band_mode, pca_k)

    n_m = len(methods)
    n_c = len(cks)

    # lookup (method, ck) -> result
    lut = {(r["method"], r["ck"]): r for r in results_ok
           if r.get("method") is not None and r.get("ck") is not None}

    cell_w, cell_h = 4.5, 3.8   # inches per cell

    # ------------------------------------------------------------------ #
    # 1. Scatter grid                                                      #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(n_m, n_c,
                             figsize=(cell_w * n_c, cell_h * n_m),
                             squeeze=False)
    for i, m in enumerate(methods):
        for j, c in enumerate(cks):
            ax = axes[i][j]
            r  = lut.get((m, c))
            p  = Path(r["p_scatter"]) if r else None
            if p and p.exists():
                ax.imshow(plt.imread(p))
                acc_str = f"acc={r['pcaK_acc']:.3f}"
                if r.get("acc_post") is not None:
                    acc_str += f"  post={r['acc_post']:.3f}"
                ax.set_title(f"{m}  ck{c}\n{acc_str}", fontsize=7)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
            ax.axis("off")
        axes[i][0].set_ylabel(m, fontsize=8)
    for j, c in enumerate(cks):
        axes[0][j].set_title(f"ck{c}\n" + axes[0][j].get_title(), fontsize=7)
    fig.suptitle(f"PCA Scatter + Boundary  |  layers={layers}  {band_mode}  pca={pca_k}",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out_scatter = base_out / f"combined_scatter__{ctag}.png"
    fig.savefig(out_scatter, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[combined] scatter      -> {out_scatter}")

    # ------------------------------------------------------------------ #
    # 2. Acc-vs-PC overlay                                                 #
    # ------------------------------------------------------------------ #
    cmap   = plt.get_cmap("tab20")
    colors = [cmap(k / max(len(results_ok) - 1, 1)) for k in range(len(results_ok))]
    # linestyle cycles per checkpoint so methods are distinguished by color
    ls_cycle = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, r in enumerate(results_ok):
        ks_r, accs_r = r.get("ks"), r.get("accs")
        if not ks_r:
            continue
        m, c = r.get("method", "?"), r.get("ck", "?")
        ls   = ls_cycle[(c - 1) % len(ls_cycle)] if isinstance(c, int) else "-"
        label = f"{m}  ck{c}  ({r['pcaK_acc']:.3f})"
        ax.plot(ks_r, accs_r, label=label, color=colors[idx],
                linewidth=1.5, linestyle=ls)

    ax.set_xlabel("Number of PCs"); ax.set_ylabel("Train accuracy (LR)")
    ax.set_title(f"Acc vs PC count  |  layers={layers}  {band_mode}  pca={pca_k}")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.25)
    ncol = max(1, math.ceil(len(results_ok) / 14))
    ax.legend(fontsize=7, ncol=ncol, loc="lower right")
    plt.tight_layout()
    out_acc = base_out / f"combined_acc_vs_pc__{ctag}.png"
    fig.savefig(out_acc, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[combined] acc_vs_pc    -> {out_acc}")

    # ------------------------------------------------------------------ #
    # 3. Truth-axis histogram grid                                         #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(n_m, n_c,
                             figsize=(cell_w * n_c, cell_h * n_m),
                             squeeze=False)
    for i, m in enumerate(methods):
        for j, c in enumerate(cks):
            ax = axes[i][j]
            r  = lut.get((m, c))
            p  = Path(r["p_hist"]) if r else None
            if p and p.exists():
                ax.imshow(plt.imread(p))
                ax.set_title(f"{m}  ck{c}", fontsize=7)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
            ax.axis("off")
    fig.suptitle(f"Truth-axis Histogram  |  layers={layers}  {band_mode}  pca={pca_k}",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out_hist = base_out / f"combined_truth_axis_hist__{ctag}.png"
    fig.savefig(out_hist, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[combined] truth_hist   -> {out_hist}")


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

    # ------------------------------------------------------------------ #
    #  Build per-run config list                                           #
    # ------------------------------------------------------------------ #
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
        # single explicit run
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

    # ------------------------------------------------------------------ #
    #  Execute                                                             #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    #  Combined figures (when running multiple)                            #
    # ------------------------------------------------------------------ #
    if len(configs) > 1 and ok and sweep_methods and sweep_cks:
        print(f"\n[pca_probe] Building combined figures ({len(ok)} successful runs)...")
        make_combined_figures(ok, sweep_methods, sweep_cks,
                              Path(args.out_dir), args.layers,
                              args.band_mode, args.pca_components)

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #
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
