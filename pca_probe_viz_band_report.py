#!/usr/bin/env python3
"""
pca_probe_viz_band_report.py

Extends your band-capable PCA probe visualization script with:

1) Accuracy vs PC-count curve (train-set diagnostic)
2) Truth-axis projection histogram (probe direction in original feature space)
   - Safe if --post_hs is not provided
3) Per-run PDF report that includes:
   - Task configuration (layers, band_mode, feature shape, PCA components)
   - Confusion matrix, accuracies
   - Plots (PCA scatter+boundary, Acc vs PCs, Truth-axis histogram)
   - Optional post-unlearning overlays + DR metrics

Inputs
------
--base_hs        Base hidden states: .npy or .zip containing one .npy, shape (N, L, D)
--post_hs        Optional post hidden states: same shape/order as base
--tf_pairs_csv   CSV with labels (and optionally split column)
--split          Which split to filter, if split column exists
--layers         Single layer like "20" or band "12-22"
--band_mode      concat | mean
--pca_components PCA K (>=2 recommended)
--max_pc_curve   Max PCs for accuracy curve (default min(50, K))
--out_dir        Output directory

Outputs (in out_dir)
--------------------
- pca_scatter_boundary.png
- acc_vs_pc.png
- truth_axis_hist.png
- report.json
- report.pdf

Notes
-----
• PCA-2D plot is a qualitative slice.
• "Truth axis" is the LR weight vector trained in ORIGINAL feature space (D or band*D).
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.backends.backend_pdf import PdfPages


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
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def plot_boundary(X2: np.ndarray, y: np.ndarray, lr2: LogisticRegression, outpath: Path,
                  overlay=None, title="PCA-2D + LR boundary"):
    fig = plt.figure(figsize=(8.5, 6.5))

    m1 = y == 1
    m0 = ~m1
    plt.scatter(X2[m0, 0], X2[m0, 1], s=12, alpha=0.6, label="Base: False")
    plt.scatter(X2[m1, 0], X2[m1, 1], s=12, alpha=0.6, label="Base: True")

    x_min, x_max = np.percentile(X2[:, 0], [0.5, 99.5])
    y_min, y_max = np.percentile(X2[:, 1], [0.5, 99.5])
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = lr2.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, zz, levels=[0.5], linewidths=2)

    if overlay is not None:
        X2b, yb, lr2b, tag = overlay
        mb1 = yb == 1
        mb0 = ~mb1
        plt.scatter(X2b[mb0, 0], X2b[mb0, 1], s=12, alpha=0.35, marker="x", label=f"{tag}: False")
        plt.scatter(X2b[mb1, 0], X2b[mb1, 1], s=12, alpha=0.35, marker="x", label=f"{tag}: True")
        zz2 = lr2b.predict_proba(grid)[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, zz2, levels=[0.5], linewidths=2, linestyles="--")

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_acc_vs_pc(Xp: np.ndarray, y: np.ndarray, max_k: int, outpath: Path):
    ks = list(range(1, max_k + 1))
    accs = []
    for k in ks:
        lr = LogisticRegression(max_iter=2000, solver="lbfgs")
        lr.fit(Xp[:, :k], y)
        pred = lr.predict(Xp[:, :k])
        accs.append(accuracy_score(y, pred))

    fig = plt.figure(figsize=(8.5, 4.5))
    plt.plot(ks, accs)
    plt.xlabel("Number of PCs used")
    plt.ylabel("Train accuracy (LR)")
    plt.title("Accuracy vs PC count (diagnostic)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return ks, accs


def plot_truth_axis_hist(z: np.ndarray, y: np.ndarray, outpath: Path,
                         z_post: np.ndarray = None, title: str = "Truth-axis projection"):
    fig = plt.figure(figsize=(8.5, 4.8))

    z0 = z[y == 0]
    z1 = z[y == 1]
    bins = 40

    plt.hist(z0, bins=bins, alpha=0.6, density=True, label="Base: False")
    plt.hist(z1, bins=bins, alpha=0.6, density=True, label="Base: True")

    if z_post is not None:
        z0p = z_post[y == 0]
        z1p = z_post[y == 1]
        plt.hist(z0p, bins=bins, alpha=0.25, density=True, histtype="step", linewidth=2, linestyle="--",
                 label="Post: False (proj on base axis)")
        plt.hist(z1p, bins=bins, alpha=0.25, density=True, histtype="step", linewidth=2, linestyle="--",
                 label="Post: True (proj on base axis)")

    plt.title(title)
    plt.xlabel("Score z = w·x + b")
    plt.ylabel("Density")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def fig_text_page(lines, title="Run Summary"):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=14)
    text = "\n".join(lines)
    fig.text(0.05, 0.95, text, va="top", ha="left", family="monospace", fontsize=9)
    plt.axis("off")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_hs", required=True)
    ap.add_argument("--post_hs", default=None)
    ap.add_argument("--tf_pairs_csv", required=True)
    ap.add_argument("--split", default="train")

    ap.add_argument("--layers", default="20", help='e.g. "20" or "12-22"')
    ap.add_argument("--band_mode", default="concat", choices=["concat", "mean"])

    ap.add_argument("--pca_components", type=int, default=20)
    ap.add_argument("--max_pc_curve", type=int, default=None, help="Max PCs for acc-vs-PC curve (default=min(50,K))")

    ap.add_argument("--out_dir", default="pca_out")
    ap.add_argument("--lr_C", type=float, default=1.0)
    ap.add_argument("--lr_balanced", action="store_true")

    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)

    df = pd.read_csv(args.tf_pairs_csv)
    if "split" in df.columns:
        df = df[df["split"] == args.split]
    if df.empty:
        raise ValueError("No rows after split filtering. Check --split and csv columns.")
    y = (df["label"].astype(str).str.lower().str.strip() == "true").astype(int).to_numpy()

    hs_base = load_hidden_states(args.base_hs)
    if hs_base.shape[0] != len(y):
        raise ValueError(f"N mismatch: hs N={hs_base.shape[0]} vs labels N={len(y)}. Ensure same ordering/filtering.")
    X_base = extract_features(hs_base, layers, args.band_mode)

    K = int(args.pca_components)
    if K < 2:
        raise ValueError("--pca_components must be >= 2 for boundary plot")
    if K > X_base.shape[1]:
        raise ValueError(f"--pca_components {K} > feature dim {X_base.shape[1]}")
    pca = PCA(n_components=K, random_state=42)
    Xp_base = pca.fit_transform(X_base)

    class_weight = "balanced" if args.lr_balanced else None
    lrK = LogisticRegression(max_iter=2000, C=args.lr_C, class_weight=class_weight, random_state=42)
    lrK.fit(Xp_base, y)
    pred = lrK.predict(Xp_base)
    acc = float(accuracy_score(y, pred))
    cm = confusion_matrix(y, pred)

    X2_base = Xp_base[:, :2]
    lr2 = LogisticRegression(max_iter=2000, C=args.lr_C, class_weight=class_weight, random_state=42)
    lr2.fit(X2_base, y)

    lr_hi = LogisticRegression(max_iter=2000, C=args.lr_C, class_weight=class_weight, random_state=42)
    lr_hi.fit(X_base, y)
    w_pre = lr_hi.coef_.ravel()
    b_pre = float(lr_hi.intercept_.ravel()[0])
    z_base = X_base @ w_pre + b_pre

    overlay = None
    dr = {}
    z_post_on_base_axis = None
    post_metrics = None

    if args.post_hs:
        hs_post = load_hidden_states(args.post_hs)
        if hs_post.shape != hs_base.shape:
            raise ValueError(f"post_hs shape {hs_post.shape} must match base {hs_base.shape}")
        X_post = extract_features(hs_post, layers, args.band_mode)

        Xp_post = pca.transform(X_post)
        X2_post = Xp_post[:, :2]

        lrK_post = LogisticRegression(max_iter=2000, C=args.lr_C, class_weight=class_weight, random_state=42)
        lrK_post.fit(Xp_post, y)
        pred_post = lrK_post.predict(Xp_post)
        acc_post = float(accuracy_score(y, pred_post))
        cm_post = confusion_matrix(y, pred_post)

        lr2_post = LogisticRegression(max_iter=2000, C=args.lr_C, class_weight=class_weight, random_state=42)
        lr2_post.fit(X2_post, y)

        lr_hi_post = LogisticRegression(max_iter=2000, C=args.lr_C, class_weight=class_weight, random_state=42)
        lr_hi_post.fit(X_post, y)
        w_post = lr_hi_post.coef_.ravel()
        dr4096 = 1.0 - cosine(w_pre, w_post)
        dr2 = 1.0 - cosine(lr2.coef_.ravel(), lr2_post.coef_.ravel())
        dr = {"dr_feature_space": float(dr4096), "dr_pca2_proxy": float(dr2)}

        overlay = (X2_post, y, lr2_post, "Post")
        z_post_on_base_axis = X_post @ w_pre + b_pre
        post_metrics = {"pcaK_acc": acc_post, "pcaK_confusion": cm_post.tolist()}

    p_scatter = out / "pca_scatter_boundary.png"
    plot_boundary(X2_base, y, lr2, p_scatter, overlay=overlay,
                  title=f"PCA-2D + LR boundary | layers={args.layers} mode={args.band_mode}")

    max_k = args.max_pc_curve
    if max_k is None:
        max_k = min(50, K)
    else:
        max_k = int(max_k)
        max_k = max(1, min(max_k, K))
    p_acc = out / "acc_vs_pc.png"
    ks, accs = plot_acc_vs_pc(Xp_base, y, max_k=max_k, outpath=p_acc)

    p_hist = out / "truth_axis_hist.png"
    plot_truth_axis_hist(z_base, y, p_hist, z_post=z_post_on_base_axis,
                         title=f"Truth-axis projection (feature space) | layers={args.layers} mode={args.band_mode}")

    report = {
        "layers": layers,
        "layers_spec": args.layers,
        "band_mode": args.band_mode,
        "feature_shape": list(X_base.shape),
        "pca_components": K,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pcaK_acc": acc,
        "pcaK_confusion": cm.tolist(),
        "acc_vs_pc": {"max_k": int(max_k), "ks": ks, "acc": accs},
        "dr": dr,
        "post": post_metrics,
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pdf_path = out / "report.pdf"
    lines = []
    lines.append("PCA Probe Visualization Report")
    lines.append("")
    lines.append(f"base_hs: {args.base_hs}")
    lines.append(f"post_hs: {args.post_hs if args.post_hs else '(none)'}")
    lines.append(f"tf_pairs_csv: {args.tf_pairs_csv}")
    lines.append(f"split: {args.split} (applied only if csv has 'split' column)")
    lines.append("")
    lines.append(f"layers_spec: {args.layers}   parsed_layers: {layers}")
    lines.append(f"band_mode: {args.band_mode}")
    lines.append(f"feature_shape: {X_base.shape}  (N, F)")
    lines.append("")
    lines.append(f"pca_components(K): {K}")
    evr = pca.explained_variance_ratio_
    evr2 = float(evr[0] + evr[1]) if len(evr) >= 2 else float(evr.sum())
    lines.append(f"explained_var(PC1): {evr[0]:.4f}")
    lines.append(f"explained_var(PC2): {evr[1]:.4f}")
    lines.append(f"explained_var(PC1+PC2): {evr2:.4f}")
    lines.append("")
    lines.append(f"LR@PCA(K) acc: {acc:.4f}")
    lines.append("Confusion matrix [[TN,FP],[FN,TP]]:")
    lines.append(str(cm))
    if post_metrics is not None:
        lines.append("")
        lines.append(f"POST LR@basePCA(K) acc: {post_metrics['pcaK_acc']:.4f}")
        lines.append("POST confusion matrix [[TN,FP],[FN,TP]]:")
        lines.append(str(np.array(post_metrics["pcaK_confusion"])))
        lines.append("")
        lines.append(f"DR(feature space cosine): {dr.get('dr_feature_space', None)}")
        lines.append(f"DR(PCA2 proxy cosine):    {dr.get('dr_pca2_proxy', None)}")

    with PdfPages(pdf_path) as pdf:
        fig0 = fig_text_page(lines, title="Run Summary")
        pdf.savefig(fig0)
        plt.close(fig0)

        for img_path, title in [
            (p_scatter, "PCA Scatter + Boundary"),
            (p_acc, "Accuracy vs PC Count"),
            (p_hist, "Truth-axis Projection Histogram"),
        ]:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle(title, fontsize=14)
            ax = fig.add_subplot(111)
            ax.axis("off")
            img = plt.imread(img_path)
            ax.imshow(img)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[done] Wrote outputs to: {out}")
    print(f"  - {p_scatter}")
    print(f"  - {p_acc}")
    print(f"  - {p_hist}")
    print(f"  - {out / 'report.json'}")
    print(f"  - {pdf_path}")


if __name__ == "__main__":
    main()
