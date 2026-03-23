#!/usr/bin/env python3
"""
plot_DR_scale.py
================
Visualises the geometry of how unlearning transforms a linear probe's
weight vector, decomposed into two orthogonal components:

  DR  (Directional Rotation) = 1 - cos(w_pre, w_post)
        Measures angular displacement of the decision boundary.
        Range [0, 2]; 0 = no rotation, 1 = orthogonal, 2 = reversed.

  SR  (Scale Ratio)          = log2( ||w_post|| / ||w_pre|| )
        Measures multiplicative change in weight-vector norm.
        0 = no scale change, negative = shrinkage, positive = amplification.
        Using log2 so that halving and doubling are symmetric (±1).

  RE  (Representational Erasure) = from summary tables (color axis)
        (APP_base - APP_post) / (APP_base - 0.5)

Why these three?
  A linear probe weight vector w can change as:
      w_post ≈ s · R · w_pre
  where R is a rotation (captured by DR) and s is a positive scalar (captured
  by SR = log s).  Together they give a 2-D picture of what each unlearning
  method does geometrically to the discriminative direction, with RE as a
  third axis encoded in colour.

Output
------
  {out_dir}/DR_scale_plot_{metric}_{band}_{clf}_{dr_mode}.png
  {out_dir}/DR_scale_table_{metric}_{band}_{clf}_{dr_mode}.csv

Usage
-----
  python plot_DR_scale.py \\
    --geometry_input_dir checkpoints \\
    --shared_df data/wmdp_tf_pairs.csv \\
    --data_dir data \\
    --re_band mb --re_clf lr \\
    --geometry_layers 1-6 \\
    --out_dir dr_scale_outputs \\
    --methods GradDiff,RMU,RMU-LAT,RepNoise,ELM,RR,TAR,PB_J

SLURM (parallel per-method geometry):
  python plot_DR_scale.py --generate_slurm [same flags] > run_scale.sh
  sbatch run_scale.sh
  # after array completes:
  python plot_DR_scale.py --slurm_collect [same flags]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Method styling ────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "Base":     "#4c78a8",
    "GradDiff": "#f58518",
    "RMU":      "#54a24b",
    "RMU-LAT":  "#e45756",
    "RepNoise": "#b279a2",
    "ELM":      "#72b7b2",
    "RR":       "#ff9da6",
    "TAR":      "#c8a459",
    "PB&J":     "#d67195",
}
_FALLBACK_COLOR = "#888888"

SAFE_TO_DISPLAY = {"PB_J": "PB&J", "RMU_LAT": "RMU-LAT"}

def _display_name(m: str) -> str:
    return SAFE_TO_DISPLAY.get(m, m)

def _safe_name(m: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", m)

DEFAULT_METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]

OFFSETS: Dict[str, Tuple[int, int]] = {
    "GradDiff": ( 8,  5),
    "RMU":      ( 8,  5),
    "RMU-LAT":  ( 8, -12),
    "RepNoise": ( 8,  5),
    "ELM":      ( 8, -12),
    "RR":       ( 8,  5),
    "TAR":      ( 8,  5),
    "PB&J":     ( 8,  5),
}

# ── Geometry helpers ──────────────────────────────────────────────────────────

def load_hidden_states(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected (N,L,D), got {arr.shape} from {path}")
    return arr


def parse_layer_spec(spec: str) -> List[int]:
    spec = str(spec).strip()
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(spec)]


class BandCompressor:
    def __init__(self, layers: List[int], mode: str = "concat", pca_dim: int = 0):
        self.layers   = list(layers)
        self.mode     = mode
        self.pca_dim  = int(pca_dim)
        self.pcas: List[Optional[PCA]] = []

    def fit(self, hs: np.ndarray) -> "BandCompressor":
        if self.mode == "mean":
            self.pcas = []
            return self
        band = hs[:, self.layers, :]
        if self.pca_dim > 0:
            self.pcas = []
            for i in range(band.shape[1]):
                k = min(self.pca_dim, band.shape[0], band.shape[2])
                p = PCA(n_components=k, random_state=42)
                p.fit(band[:, i, :])
                self.pcas.append(p)
        else:
            self.pcas = [None] * band.shape[1]
        return self

    def transform(self, hs: np.ndarray) -> np.ndarray:
        band = hs[:, self.layers, :]
        if self.mode == "mean":
            return band.mean(axis=1)
        if self.pca_dim > 0:
            return np.concatenate(
                [self.pcas[i].transform(band[:, i, :]) for i in range(band.shape[1])],
                axis=1)
        return band.reshape(hs.shape[0], -1)


def build_lr(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    LogisticRegression(max_iter=4000, C=C, random_state=42)),
    ])


def weight_vector(pipe: Pipeline) -> np.ndarray:
    return np.asarray(pipe.named_steps["clf"].coef_).reshape(-1)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return None if (na == 0 or nb == 0) else float(np.dot(a, b) / (na * nb))


def discover_hs_methods(d: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    pat = re.compile(r"^(?P<m>.+)_hs_(?P<s>train|val|test)\.npy$")
    for p in d.iterdir():
        mo = pat.match(p.name)
        if mo:
            out.setdefault(mo.group("m"), {})[f"hs_{mo.group('s')}"] = str(p)
    return out


def fit_probe(hs: np.ndarray, y: np.ndarray,
              layers: List[int], mode: str, pca_dim: int, C: float
              ) -> Tuple[BandCompressor, Pipeline, np.ndarray]:
    comp = BandCompressor(layers, mode, pca_dim).fit(hs)
    X    = comp.transform(hs)
    pipe = build_lr(C)
    pipe.fit(X, y)
    return comp, pipe, weight_vector(pipe)


# ── Geometry metrics ──────────────────────────────────────────────────────────

def compute_dr_sr(w_pre: np.ndarray, w_post: np.ndarray) -> Tuple[float, float]:
    """Return (DR, SR) for a pair of weight vectors.

    DR = 1 - cos(w_pre, w_post)  in [0, 2]
    SR = log2(||w_post|| / ||w_pre||)  — signed log scale ratio
    """
    cs  = cosine_sim(w_pre, w_post)
    DR  = np.nan if cs is None else float(1.0 - cs)
    na, nb = np.linalg.norm(w_pre), np.linalg.norm(w_post)
    SR  = np.nan if na == 0 else float(np.log2(nb / na))
    return DR, SR


def compute_geometry_all_methods(
    geometry_input_dir: Path,
    shared_df_path: Path,
    base_method: str,
    methods: Iterable[str],
    layers_spec: str,
    mode: str,
    pca_dim: int,
    C: float,
    split_col: str,
    label_col: str,
    train_split: str,
) -> Dict[str, Dict[str, float]]:
    """Returns {method: {DR: float, SR: float, w_norm_pre: float, w_norm_post: float}}."""
    files    = discover_hs_methods(geometry_input_dir)
    shared   = pd.read_csv(shared_df_path)
    train_df = shared[shared[split_col].astype(str) == str(train_split)].reset_index(drop=True)
    y_train  = (train_df[label_col].astype(str).str.lower().str.strip() == "true").astype(int).to_numpy()
    layers   = parse_layer_spec(layers_spec)

    if base_method not in files or "hs_train" not in files[base_method]:
        raise FileNotFoundError(f"Base hs_train not found in {geometry_input_dir}")
    base_hs = load_hidden_states(files[base_method]["hs_train"])
    _, _, w_pre = fit_probe(base_hs, y_train, layers, mode, pca_dim, C)

    out: Dict[str, Dict[str, float]] = {}
    for method in methods:
        if method == base_method:
            continue
        mf = files.get(method)
        if not mf or "hs_train" not in mf:
            raise FileNotFoundError(f"hs_train for '{method}' not found in {geometry_input_dir}")
        N = min(len(load_hidden_states(mf["hs_train"])), len(y_train))
        hs_m   = load_hidden_states(mf["hs_train"])[:N]
        _, _, w_post = fit_probe(hs_m, y_train[:N], layers, mode, pca_dim, C)
        DR, SR = compute_dr_sr(w_pre, w_post)
        out[method] = {
            "DR": DR, "SR": SR,
            "w_norm_pre": float(np.linalg.norm(w_pre)),
            "w_norm_post": float(np.linalg.norm(w_post)),
        }
    return out


def compute_geometry_one_method(
    method: str,
    geometry_input_dir: Path,
    shared_df_path: Path,
    base_method: str,
    layers_spec: str,
    mode: str,
    pca_dim: int,
    C: float,
    split_col: str,
    label_col: str,
    train_split: str,
) -> Dict[str, float]:
    files    = discover_hs_methods(geometry_input_dir)
    shared   = pd.read_csv(shared_df_path)
    train_df = shared[shared[split_col].astype(str) == str(train_split)].reset_index(drop=True)
    y_train  = (train_df[label_col].astype(str).str.lower().str.strip() == "true").astype(int).to_numpy()
    layers   = parse_layer_spec(layers_spec)

    base_hs = load_hidden_states(files[base_method]["hs_train"])
    _, _, w_pre = fit_probe(base_hs, y_train, layers, mode, pca_dim, C)

    hs_m = load_hidden_states(files[method]["hs_train"])
    N    = min(len(hs_m), len(y_train))
    _, _, w_post = fit_probe(hs_m[:N], y_train[:N], layers, mode, pca_dim, C)
    DR, SR = compute_dr_sr(w_pre, w_post)
    return {
        "DR": DR, "SR": SR,
        "w_norm_pre": float(np.linalg.norm(w_pre)),
        "w_norm_post": float(np.linalg.norm(w_post)),
    }


# ── RE from summary tables ────────────────────────────────────────────────────

def load_re_from_summary(
    data_dir: Path,
    methods: List[str],
    re_band: str,
    re_clf: str,
    metric: str,
) -> Dict[str, float]:
    t2   = pd.read_csv(data_dir / "summary_table2_base_probes.csv",   index_col=0)
    t3   = pd.read_csv(data_dir / "summary_table3_method_probes.csv", index_col=0)
    col  = f"{re_band}_{re_clf}_{metric}"
    APP_base = float(t2.loc["Base", col])
    out: Dict[str, float] = {}
    for m in methods:
        dm = _display_name(m)
        APP_post = float(t3.loc[dm, col]) if dm in t3.index else np.nan
        denom = APP_base - 0.5
        out[dm] = np.nan if abs(denom) < 1e-12 else (APP_base - APP_post) / denom
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_dr_scale(
    df: pd.DataFrame,
    out_png: Path,
    metric: str,
    title: str,
) -> None:
    """Scatter: x = DR, y = SR (log2 scale ratio), color = RE per method."""

    re_vals = df["RE"].dropna()
    if len(re_vals) == 0:
        print("[warn] No RE values to plot.")
        return

    # Diverging colormap centred at RE = 0
    re_min, re_max = float(re_vals.min()), float(re_vals.max())
    if re_min < 0 < re_max:
        norm = TwoSlopeNorm(vmin=re_min, vcenter=0.0, vmax=re_max)
    else:
        norm = Normalize(vmin=re_min, vmax=re_max)
    cmap = cm.RdYlGn  # green = high RE (strong erasure), red = low RE

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.set_axisbelow(True)
    ax.grid(color="grey", alpha=0.18, lw=0.6)

    # Reference lines
    ax.axhline(0, color="grey",       lw=0.9, ls="--", alpha=0.45)
    ax.axvline(0, color="grey",       lw=0.9, ls="--", alpha=0.45)
    ax.axvline(1, color="darkorange", lw=0.8, ls=":",  alpha=0.55, label="DR = 1")

    # Scatter
    sc = ax.scatter(
        df["DR"], df["SR"],
        c=df["RE"], cmap=cmap, norm=norm,
        s=300, zorder=5, edgecolors="k", linewidths=0.9,
    )

    # Per-method color label in the legend
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=METHOD_COLORS.get(dm, _FALLBACK_COLOR),
               markeredgecolor="k", markeredgewidth=0.7,
               markersize=9, label=dm)
        for dm in df["method"].tolist()
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left",
              framealpha=0.88, edgecolor="#cccccc")

    # Method labels
    for _, row in df.iterrows():
        ox, oy = OFFSETS.get(str(row["method"]), (8, 5))
        ax.annotate(str(row["method"]),
                    xy=(row["DR"], row["SR"]),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=9, fontweight="semibold")

    # Colorbar for RE
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="3%", pad=0.08)
    cbar    = plt.colorbar(sc, cax=cax)
    cbar.set_label(
        f"RE — Representational Erasure ({metric.upper()})\n"
        r"$\frac{A_{PP}^{\rm base} - A_{PP}^{\rm post}}{A_{PP}^{\rm base} - 0.5}$",
        fontsize=9)

    # Auto-scale axes
    vx = df["DR"].dropna()
    vy = df["SR"].dropna()
    if len(vx):
        xm = max((vx.max() - vx.min()) * 0.22 + 0.05, 0.12)
        ax.set_xlim(vx.min() - xm, vx.max() + xm)
    if len(vy):
        ym = max((vy.max() - vy.min()) * 0.22 + 0.05, 0.15)
        ax.set_ylim(vy.min() - ym, vy.max() + ym)

    ax.set_xlabel(
        "DR — Directional Rotation\n"
        r"$1 - \cos(\mathbf{w}_{\rm pre},\,\mathbf{w}_{\rm post})$",
        fontsize=10)
    ax.set_ylabel(
        r"SR — Scale Ratio  $\log_2\!\left(\|\mathbf{w}_{\rm post}\| / \|\mathbf{w}_{\rm pre}\|\right)$",
        fontsize=10)

    fig.suptitle(title, fontsize=13, y=0.99)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved plot -> {out_png}")


# ── SLURM helpers ─────────────────────────────────────────────────────────────

def generate_slurm_script(
    methods: List[str],
    passthrough_args: str,
    partial_dir: str,
    partition: str,
    mem: str,
    cpus: int,
    time_limit: str,
    script_path: str,
) -> str:
    n = len(methods)
    methods_json = json.dumps(methods)
    return f"""#!/bin/bash
#SBATCH --job-name=dr_scale_geom
#SBATCH --partition={partition}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time_limit}
#SBATCH --array=0-{n - 1}
#SBATCH --output=logs/dr_scale_%A_%a.out

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

METHODS='{methods_json}'
METHOD=$(python3 -c "import sys,json; ms=json.loads(sys.argv[1]); print(ms[int(sys.argv[2])])" "$METHODS" "$SLURM_ARRAY_TASK_ID")

python {script_path} \\
  {passthrough_args} \\
  --slurm_task_method "$METHOD" \\
  --slurm_partial_dir "{partial_dir}"
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Geometry
    ap.add_argument("--geometry_input_dir",         required=True)
    ap.add_argument("--shared_df",                  required=True)
    ap.add_argument("--base_method",                default="base")
    ap.add_argument("--geometry_layers",            default="0-31")
    ap.add_argument("--geometry_mode", choices=["mean", "concat"], default="concat")
    ap.add_argument("--geometry_pca_dim",           type=int, default=0)
    ap.add_argument("--geometry_C",                 type=float, default=1.0)
    ap.add_argument("--split_col",                  default="split")
    ap.add_argument("--label_col",                  default="label")
    ap.add_argument("--train_split",                default="train")
    # RE from summary tables
    ap.add_argument("--data_dir",                   default=".")
    ap.add_argument("--re_band",                    default="mb")
    ap.add_argument("--re_clf",                     default="lr")
    ap.add_argument("--metrics",                    default="acc,auc")
    # Methods
    ap.add_argument("--methods",  default=",".join(DEFAULT_METHODS))
    ap.add_argument("--out_dir",  default="dr_scale_outputs")
    # SLURM
    ap.add_argument("--generate_slurm",    action="store_true")
    ap.add_argument("--slurm_partition",   default="public")
    ap.add_argument("--slurm_mem",         default="32G")
    ap.add_argument("--slurm_cpus",        type=int, default=4)
    ap.add_argument("--slurm_time",        default="00:30:00")
    ap.add_argument("--slurm_partial_dir", default="dr_scale_partial")
    ap.add_argument("--slurm_task_method", default=None)
    ap.add_argument("--slurm_collect",     action="store_true")

    args    = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    geom_kw = dict(
        geometry_input_dir = Path(args.geometry_input_dir),
        shared_df_path     = Path(args.shared_df),
        base_method        = args.base_method,
        layers_spec        = args.geometry_layers,
        mode               = args.geometry_mode,
        pca_dim            = args.geometry_pca_dim,
        C                  = args.geometry_C,
        split_col          = args.split_col,
        label_col          = args.label_col,
        train_split        = args.train_split,
    )

    # ── SLURM: generate script ────────────────────────────────────────────────
    if args.generate_slurm:
        passthrough = [
            f"--geometry_input_dir {args.geometry_input_dir}",
            f"--shared_df {args.shared_df}",
            f"--base_method {args.base_method}",
            f"--geometry_layers {args.geometry_layers}",
            f"--geometry_mode {args.geometry_mode}",
            f"--geometry_C {args.geometry_C}",
            f"--data_dir {args.data_dir}",
            f"--re_band {args.re_band}",
            f"--re_clf {args.re_clf}",
            f"--metrics {args.metrics}",
            f"--methods {args.methods}",
            f"--out_dir {args.out_dir}",
        ]
        print(generate_slurm_script(
            methods=methods,
            passthrough_args=" \\\n  ".join(passthrough),
            partial_dir=args.slurm_partial_dir,
            partition=args.slurm_partition,
            mem=args.slurm_mem,
            cpus=args.slurm_cpus,
            time_limit=args.slurm_time,
            script_path=str(Path(__file__).resolve()),
        ))
        return

    # ── SLURM: single-method task ─────────────────────────────────────────────
    if args.slurm_task_method is not None:
        method      = args.slurm_task_method
        partial_dir = Path(args.slurm_partial_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)
        out_json    = partial_dir / f"{_safe_name(method)}_geometry.json"
        if out_json.exists():
            print(f"[skip] {out_json} already exists.")
            return
        result = compute_geometry_one_method(method=method, **geom_kw)
        with open(out_json, "w") as fh:
            json.dump({method: result}, fh)
        print(f"Saved geometry for {method}: DR={result['DR']:.4f}  SR={result['SR']:.4f}  -> {out_json}")
        return

    # ── Collect geometry ──────────────────────────────────────────────────────
    if args.slurm_collect:
        partial_dir = Path(args.slurm_partial_dir)
        raw_geom: Dict[str, Dict[str, float]] = {}
        for p in partial_dir.glob("*_geometry.json"):
            with open(p) as fh:
                raw_geom.update(json.load(fh))
        print(f"Loaded geometry for {len(raw_geom)} methods from {partial_dir}")
    else:
        raw_geom = compute_geometry_all_methods(methods=methods, **geom_kw)

    # Remap safe names to display names
    geom: Dict[str, Dict[str, float]] = {_display_name(k): v for k, v in raw_geom.items()}

    # ── Per-metric loop ───────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    for metric in metrics:
        re_map = load_re_from_summary(data_dir, methods, args.re_band, args.re_clf, metric)

        rows = []
        for m in methods:
            dm = _display_name(m)
            if dm not in geom:
                print(f"  [skip] {dm} — no geometry data")
                continue
            rows.append({
                "method":     dm,
                "DR":         geom[dm]["DR"],
                "SR":         geom[dm]["SR"],
                "RE":         re_map.get(dm, np.nan),
                "w_norm_pre":  geom[dm]["w_norm_pre"],
                "w_norm_post": geom[dm]["w_norm_post"],
                "metric":     metric,
            })
        if not rows:
            print(f"  [skip] metric={metric} — no data")
            continue

        df = pd.DataFrame(rows)

        suffix   = f"{metric}_{args.re_band}_{args.re_clf}_{args.geometry_layers.replace('-', '_')}"
        csv_path = out_dir / f"DR_scale_table_{suffix}.csv"
        png_path = out_dir / f"DR_scale_plot_{suffix}.png"
        df.to_csv(csv_path, index=False)
        print(f"  {metric.upper()} table -> {csv_path}")

        plot_dr_scale(
            df, png_path,
            metric=metric,
            title=(f"Probe Weight Geometry After Unlearning\n"
                   f"layers={args.geometry_layers}  band={args.re_band}  metric={metric.upper()}"),
        )


if __name__ == "__main__":
    main()
