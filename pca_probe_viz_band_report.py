#!/usr/bin/env python3
"""
pca_probe_viz_band_report.py

Extends your band-capable PCA probe visualization script with:

1) Accuracy vs PC-count curve (train-set diagnostic)
2) Truth-axis projection histogram (probe direction in original feature space)
3) Per-run PDF report (config, confusion matrix, all plots, optional DR metrics)
4) Multi-method / multi-checkpoint sweeps with optional parallel execution

Inputs
------
Single-run mode (explicit paths):
  --base_hs        Base hidden states .npy, shape (N, L, D)
  --post_hs        Optional post hidden states (same shape/order)
  --tf_pairs_csv   CSV with label column (and optional split column)

Multi-run mode (auto-resolve sweep paths):
  --methods        One or more method names, or "all"
                   Known: GradDiff RMU RMU-LAT RepNoise ELM RR TAR PB_J
  --checkpoints    Checkpoint numbers or ranges, e.g.  1 2-4 8  (or "all" = 1-8)

Shared options:
  --split          Split to filter (default: train)
  --layers         Single layer "20" or band "12-22"
  --band_mode      concat | mean
  --pca_components PCA K (>=2)
  --max_pc_curve   Max PCs for acc-vs-PC curve (default min(50,K))
  --out_dir        Base output directory (default: pca_output/)
                   In multi-run mode each task writes to its own subdirectory.
  --workers        Parallel worker processes (default 1; set >1 for multi-run)

Outputs (per run, inside out_dir/<tag>/)
-----------------------------------------
  pca_scatter_boundary__<tag>.png
  acc_vs_pc__<tag>.png
  truth_axis_hist__<tag>.png
  report.json
  report.pdf

where <tag> = <method>_ck<N>_layers<L>_<mode>_pca<K>

Notes
-----
• PCA-2D plot is a qualitative slice.
• "Truth axis" is the LR weight vector trained in ORIGINAL feature space.
• matplotlib backend is forced to Agg (non-interactive, safe on SLURM).
"""

import argparse
import json
import zipfile
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")   # must be before any other matplotlib import
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
ALL_CKS     = list(range(1, 9))   # ck1 … ck8


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #
def parse_ck_specs(specs: list) -> list:
    """Parse checkpoint specs like ['1', '2-4', '8', 'all'] -> sorted unique ints."""
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
    layers_safe = layers.replace("-", "to")
    return f"{method}_ck{ck}_layers{layers_safe}_{band_mode}_pca{pca_k}"


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
    raise ValueError(f"Unsupported hidden state format: {path}")


def parse_layers(spec: str):
    spec = str(spec).strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        a, b = int(a), int(b)
        if b < a:
            raise ValueError("layers range must be like 12-22 (start<=end)")
        return list(range(a, b + 1))
    return [int(spec)]


def extract_features(hs: np.ndarray, layers, mode: str) -> np.ndarray:
    if hs.ndim != 3:
        raise ValueError(f"Expected hs shape (N,L,D), got {hs.shape}")
    if len(layers) == 1:
        return hs[:, layers[0], :]
    band = hs[:, layers, :]
    if mode == "mean":
        return band.mean(axis=1)
    if mode == "concat":
        N, bandL, D = band.shape
        return band.reshape(N, bandL * D)
    raise ValueError("Invalid band_mode (use concat or mean)")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(float)
    b = b.reshape(-1).astype(float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


# --------------------------------------------------------------------------- #
#  Plot functions                                                               #
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


def plot_acc_vs_pc(Xp, y, max_k, outpath):
    ks = list(range(1, max_k + 1))
    accs = []
    for k in ks:
        lr = LogisticRegression(max_iter=2000, solver="lbfgs")
        lr.fit(Xp[:, :k], y)
        accs.append(accuracy_score(y, lr.predict(Xp[:, :k])))
    fig = plt.figure(figsize=(8.5, 4.5))
    plt.plot(ks, accs); plt.xlabel("Number of PCs used")
    plt.ylabel("Train accuracy (LR)"); plt.title("Accuracy vs PC count (diagnostic)")
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
                 linewidth=2, linestyle="--", label="Post: False (proj on base axis)")
        plt.hist(z_post[y == 1], bins=bins, alpha=0.25, density=True, histtype="step",
                 linewidth=2, linestyle="--", label="Post: True (proj on base axis)")
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
#  Core analysis — runs one (base_hs, post_hs) pair                            #
# --------------------------------------------------------------------------- #
def run_one(cfg: argparse.Namespace) -> dict:
    """
    Run the full PCA probe analysis for a single configuration.
    Returns a summary dict.  Designed to be called from worker processes.
    """
    tag  = getattr(cfg, "_tag", "run")
    out  = Path(cfg.out_dir)
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
        raise ValueError(f"N mismatch: hs={hs_base.shape[0]} vs labels={len(y)}")
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

    overlay = None
    dr = {}
    z_post_on_base_axis = None
    post_metrics = None

    if cfg.post_hs:
        hs_post = load_hidden_states(cfg.post_hs)
        if hs_post.shape != hs_base.shape:
            raise ValueError(f"post_hs shape {hs_post.shape} != base {hs_base.shape}")
        X_post = extract_features(hs_post, layers, cfg.band_mode)

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
        overlay = (X2_post, y, lr2_post, "Post")
        z_post_on_base_axis = X_post @ w_pre + b_pre
        post_metrics = {"pcaK_acc": acc_post, "pcaK_confusion": cm_post.tolist()}

    # --- output filenames embed the tag for uniqueness ---
    p_scatter = out / f"pca_scatter_boundary__{tag}.png"
    plot_boundary(X2_base, y, lr2, p_scatter, overlay=overlay,
                  title=f"PCA-2D + LR boundary | {tag}")

    max_k = cfg.max_pc_curve
    if max_k is None:
        max_k = min(50, K)
    else:
        max_k = max(1, min(int(max_k), K))
    p_acc = out / f"acc_vs_pc__{tag}.png"
    ks, accs = plot_acc_vs_pc(Xp_base, y, max_k=max_k, outpath=p_acc)

    p_hist = out / f"truth_axis_hist__{tag}.png"
    plot_truth_axis_hist(z_base, y, p_hist, z_post=z_post_on_base_axis,
                         title=f"Truth-axis projection | {tag}")

    report = {
        "tag": tag,
        "base_hs": cfg.base_hs,
        "post_hs": cfg.post_hs,
        "layers": layers, "layers_spec": cfg.layers, "band_mode": cfg.band_mode,
        "feature_shape": list(X_base.shape), "pca_components": K,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pcaK_acc": acc, "pcaK_confusion": cm.tolist(),
        "acc_vs_pc": {"max_k": int(max_k), "ks": ks, "acc": accs},
        "dr": dr, "post": post_metrics,
    }
    with open(out / f"report__{tag}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # PDF
    lines = [
        f"PCA Probe Report  [{tag}]", "",
        f"base_hs : {cfg.base_hs}",
        f"post_hs : {cfg.post_hs or '(none)'}",
        f"csv     : {cfg.tf_pairs_csv}",
        f"split   : {cfg.split}", "",
        f"layers  : {cfg.layers}  ->  {layers}",
        f"band    : {cfg.band_mode}",
        f"feat    : {X_base.shape}  (N, F)", "",
        f"PCA K   : {K}",
    ]
    evr = pca.explained_variance_ratio_
    lines += [
        f"var PC1 : {evr[0]:.4f}",
        f"var PC2 : {evr[1]:.4f}",
        f"var 1+2 : {evr[0]+evr[1]:.4f}", "",
        f"LR@PCA(K) acc : {acc:.4f}",
        "Confusion [[TN,FP],[FN,TP]]:", str(cm),
    ]
    if post_metrics is not None:
        lines += [
            "", f"POST acc : {post_metrics['pcaK_acc']:.4f}",
            "POST confusion:", str(np.array(post_metrics["pcaK_confusion"])), "",
            f"DR feat-space : {dr.get('dr_feature_space'):.4f}",
            f"DR PCA-2      : {dr.get('dr_pca2_proxy'):.4f}",
        ]

    pdf_path = out / f"report__{tag}.pdf"
    with PdfPages(pdf_path) as pdf:
        fig0 = fig_text_page(lines, title=f"Run Summary — {tag}")
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

    print(f"[done] {tag}  ->  {out}")
    return {"tag": tag, "pcaK_acc": acc, "out": str(out)}


# --------------------------------------------------------------------------- #
#  Worker wrapper (top-level so it is picklable)                               #
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
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # --- single-run explicit paths ---
    ap.add_argument("--base_hs",      default=str(_CHECKPOINT_DIR / "base_hs_train.npy"))
    ap.add_argument("--post_hs",      default=None,
                    help="Explicit post hidden-states path (single-run mode).")
    ap.add_argument("--tf_pairs_csv", default=str(_CHECKPOINT_DIR / "wmdp_tf_pairs.csv"))
    ap.add_argument("--split",        default="train")

    # --- multi-run sweep ---
    ap.add_argument("--methods", nargs="+", default=None,
                    metavar="METHOD",
                    help='Method names or "all". e.g. --methods PB_J RMU  or  --methods all')
    ap.add_argument("--checkpoints", nargs="+", default=None,
                    metavar="CK",
                    help='Checkpoint numbers / ranges or "all". e.g. --checkpoints 1 3-5  or  --checkpoints all')

    # --- shared analysis params ---
    ap.add_argument("--layers",         default="12-22", help='e.g. "20" or "12-22"')
    ap.add_argument("--band_mode",      default="concat", choices=["concat", "mean"])
    ap.add_argument("--pca_components", type=int,   default=40)
    ap.add_argument("--max_pc_curve",   type=int,   default=None)
    ap.add_argument("--lr_C",           type=float, default=1.0)
    ap.add_argument("--lr_balanced",    action="store_true")

    # --- output / parallelism ---
    ap.add_argument("--out_dir", default=str(_SCRIPT_DIR / "pca_output"),
                    help="Base output directory. Each run writes to its own subdirectory.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes (default 1). Set >1 with --methods/--checkpoints.")

    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    #  Build list of per-run configs                                       #
    # ------------------------------------------------------------------ #
    configs = []

    if args.methods or args.checkpoints:
        # Multi-run: resolve methods and checkpoint numbers
        methods = args.methods if args.methods else ["PB_J"]
        if methods == ["all"]:
            methods = ALL_METHODS

        cks = parse_ck_specs(args.checkpoints) if args.checkpoints else [1]

        for method in methods:
            for ck in cks:
                post_hs_path = resolve_post_hs(method, ck)
                tag = make_tag(method, ck, args.layers, args.band_mode, args.pca_components)
                cfg = argparse.Namespace(**vars(args))
                cfg.post_hs = str(post_hs_path)
                cfg.out_dir = str(Path(args.out_dir) / tag)
                cfg._tag    = tag
                configs.append(cfg)
    else:
        # Single-run: use explicit --post_hs (or None for base-only)
        cfg = argparse.Namespace(**vars(args))
        if args.post_hs:
            # derive a tag from the post_hs path: .../sweep_RMU/ck3/hs_train.npy -> RMU_ck3_...
            p = Path(args.post_hs)
            try:
                method = p.parts[-3].replace("sweep_", "")
                ck     = int(p.parts[-2].replace("ck", ""))
                tag    = make_tag(method, ck, args.layers, args.band_mode, args.pca_components)
            except Exception:
                tag = "run"
            cfg.out_dir = str(Path(args.out_dir) / tag)
        else:
            tag = f"base_only_layers{args.layers.replace('-','to')}_{args.band_mode}_pca{args.pca_components}"
            cfg.out_dir = str(Path(args.out_dir) / tag)
            tag = "base_only"
        cfg._tag = tag
        configs.append(cfg)

    print(f"[pca_probe] {len(configs)} run(s) queued, workers={args.workers}")
    for c in configs:
        print(f"  -> {c._tag}")

    # ------------------------------------------------------------------ #
    #  Execute (serial or parallel)                                        #
    # ------------------------------------------------------------------ #
    if args.workers > 1 and len(configs) > 1:
        n_workers = min(args.workers, len(configs))
        print(f"\n[pca_probe] Running {len(configs)} tasks with {n_workers} parallel workers...\n")
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, cfg): cfg._tag for cfg in configs}
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        results = [_worker(cfg) for cfg in configs]

    # Summary
    print("\n[pca_probe] All done.")
    ok  = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]
    for r in ok:
        print(f"  OK   {r['tag']}  acc={r.get('pcaK_acc', '?'):.4f}  -> {r['out']}")
    for r in err:
        print(f"  FAIL {r['tag']}  {r['error']}")


if __name__ == "__main__":
    main()
