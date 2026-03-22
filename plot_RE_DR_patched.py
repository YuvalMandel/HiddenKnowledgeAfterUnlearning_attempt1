#!/usr/bin/env python3
"""
Patch/extension of plot_RE_DR.py.

What this script adds
---------------------
1) RE from AUC in addition to accuracy.
   The original script already *almost* supported this via --metric auc because the
   summary tables contain *_auc columns. This patched version makes that explicit,
   supports plotting multiple metrics in one run, and writes the computed table.

2) Optional geometry-based DR using cosine displacement of probe weight vectors:

       DR_geom = 1 - cos(w_pre, w_post)

   where:
     - w_pre  = weight vector of a linear probe trained on Base hidden states
     - w_post = weight vector of a linear probe trained on Method hidden states

   This requires hidden-state .npy files (and a shared labels/splits CSV) and cannot
   be recovered from the summary tables alone.

Important distinction
---------------------
- Summary-mode DR (default) reproduces the *functional/proxy* DR from transfer accuracies/AUC.
- Geometry-mode DR uses actual probe weights learned from hidden states.

Inputs for geometry mode
------------------------
Expected file naming inside --geometry-input-dir:
  <method>_hs_train.npy
  <method>_hs_val.npy      (not required here but tolerated)
  <method>_hs_test.npy     (not used here but tolerated)

Required methods include the Base method and all post-unlearning methods.

Shared labels CSV (e.g. wmdp_tf_pairs.csv) must contain:
  - split column (default: split)
  - binary label column with values True/False (default: label)

By default this script trains on train split only and computes w_pre/w_post from the
chosen band compression and linear probe.

Output
------
For each metric requested, produces:
  - CSV of computed RE/DR values
  - PNG scatter (two panels if attack scores are available)

If geometry DR is requested and files are available, DR comes from 1-cos(weights).
Otherwise DR falls back to the original summary/proxy definition.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
#  Geometry helpers (minimal, self-contained subset of batch_confidence...)   #
# --------------------------------------------------------------------------- #

def load_hidden_states(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected hidden states shape (N,L,D), got {arr.shape} from {path}")
    return arr


def parse_layer_spec(spec: str) -> List[int]:
    spec = str(spec).strip()
    if "," in spec:
        vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
        if not vals:
            raise ValueError(f"Invalid layer spec: {spec}")
        return vals
    if "-" in spec:
        a, b = spec.split("-", 1)
        a, b = int(a), int(b)
        if b < a:
            raise ValueError(f"Invalid layer range: {spec}")
        return list(range(a, b + 1))
    return [int(spec)]


class BandCompressor:
    """Compress a band of hidden states into a fixed vector.

    mode='mean'  -> average across selected layers
    mode='concat' -> concatenate selected layers, optionally after per-layer PCA
    """
    def __init__(self, layers: List[int], mode: str = "concat", pca_dim_per_layer: int = 0):
        self.layers = list(layers)
        self.mode = mode
        self.pca_dim_per_layer = int(pca_dim_per_layer)
        self.layer_pcas: List[Optional[PCA]] = []

    def fit(self, hs_train: np.ndarray) -> "BandCompressor":
        _, L, _ = hs_train.shape
        for ell in self.layers:
            if ell < 0 or ell >= L:
                raise ValueError(f"Layer index {ell} out of range for tensor with L={L}")
        if self.mode == "mean":
            self.layer_pcas = []
            return self
        if self.mode != "concat":
            raise ValueError(f"Unsupported mode: {self.mode}")
        band = hs_train[:, self.layers, :]
        if self.pca_dim_per_layer > 0:
            self.layer_pcas = []
            for i in range(band.shape[1]):
                Xi = band[:, i, :]
                k = min(self.pca_dim_per_layer, Xi.shape[0], Xi.shape[1])
                pca = PCA(n_components=k, random_state=42)
                pca.fit(Xi)
                self.layer_pcas.append(pca)
        else:
            self.layer_pcas = [None] * band.shape[1]
        return self

    def transform(self, hs: np.ndarray) -> np.ndarray:
        band = hs[:, self.layers, :]
        if self.mode == "mean":
            return band.mean(axis=1)
        if self.mode != "concat":
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.pca_dim_per_layer > 0:
            return np.concatenate([self.layer_pcas[i].transform(band[:, i, :]) for i in range(band.shape[1])], axis=1)
        return band.reshape(hs.shape[0], -1)


def build_linear_classifier(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, C=C, random_state=42)),
    ])


def get_lr_weight(pipeline: Pipeline) -> np.ndarray:
    return np.asarray(pipeline.named_steps["clf"].coef_).reshape(-1)


def cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def dr_from_weights(w_pre: np.ndarray, w_post: np.ndarray) -> Optional[float]:
    c = cosine(w_pre, w_post)
    if c is None:
        return None
    return float(1.0 - c)


def discover_hs_methods(input_dir: Path) -> Dict[str, Dict[str, str]]:
    """Discover methods from <method>_hs_<split>.npy files."""
    methods: Dict[str, Dict[str, str]] = {}
    hs_pat = re.compile(r"^(?P<method>.+)_hs_(?P<split>train|val|test)\.npy$")
    for p in input_dir.iterdir():
        m = hs_pat.match(p.name)
        if m:
            methods.setdefault(m.group("method"), {})[f"hs_{m.group('split')}"] = str(p)
    return methods


def compute_geometry_dr_from_probe_weights(
    geometry_input_dir: Path,
    shared_df_path: Path,
    base_method: str,
    methods: Iterable[str],
    band: str,
    clf: str,
    split_col: str,
    label_col: str,
    train_split: str,
    layers_spec: str,
    mode: str,
    pca_dim_per_layer: int,
    C: float,
) -> Dict[str, float]:
    """Return geometry-based DR = 1 - cos(w_pre, w_post) for each method.

    Notes:
    - Uses only linear probes, because cosine-DR needs explicit weight vectors.
    - Fits a band compressor separately on Base train states and on Method train states,
      mirroring the pre/post representation frames.
    """
    if clf != "lr":
        raise ValueError("Geometry-based DR requires a linear probe (clf=lr) to expose weight vectors.")

    methods_files = discover_hs_methods(geometry_input_dir)
    if base_method not in methods_files or "hs_train" not in methods_files[base_method]:
        raise FileNotFoundError(f"Base hidden states not found for method '{base_method}' in {geometry_input_dir}")

    shared_df = pd.read_csv(shared_df_path)
    if split_col not in shared_df.columns or label_col not in shared_df.columns:
        raise ValueError(f"Shared labels file must contain columns '{split_col}' and '{label_col}'")

    train_df = shared_df[shared_df[split_col].astype(str) == str(train_split)].reset_index(drop=True)
    y_train = (train_df[label_col].astype(str).str.lower().str.strip() == "true").astype(int).to_numpy()

    base_hs_train = load_hidden_states(methods_files[base_method]["hs_train"])
    if len(base_hs_train) != len(train_df):
        raise ValueError(f"Base hs_train length {len(base_hs_train)} != train labels length {len(train_df)}")

    layers = parse_layer_spec(layers_spec)
    base_comp = BandCompressor(layers, mode=mode, pca_dim_per_layer=pca_dim_per_layer).fit(base_hs_train)
    Xb_train = base_comp.transform(base_hs_train)
    pre_probe = build_linear_classifier(C=C)
    pre_probe.fit(Xb_train, y_train)
    w_pre = get_lr_weight(pre_probe)

    out: Dict[str, float] = {}
    for method in methods:
        if method == base_method:
            continue
        files = methods_files.get(method)
        if not files or "hs_train" not in files:
            raise FileNotFoundError(f"Hidden states for method '{method}' not found in {geometry_input_dir}")
        hs_train = load_hidden_states(files["hs_train"])
        if len(hs_train) != len(train_df):
            raise ValueError(f"{method} hs_train length {len(hs_train)} != train labels length {len(train_df)}")
        post_comp = BandCompressor(layers, mode=mode, pca_dim_per_layer=pca_dim_per_layer).fit(hs_train)
        Xm_train = post_comp.transform(hs_train)
        post_probe = build_linear_classifier(C=C)
        post_probe.fit(Xm_train, y_train)
        w_post = get_lr_weight(post_probe)
        dr = dr_from_weights(w_pre, w_post)
        out[method] = np.nan if dr is None else float(dr)
    return out

# --------------------------------------------------------------------------- #
#  Summary-table RE/DR logic                                                  #
# --------------------------------------------------------------------------- #

GAT_NAME_MAP = {
    "Grad Diff":  "GradDiff",
    "RMU":        "RMU",
    "RMU + LAT":  "RMU-LAT",
    "RepNoise":   "RepNoise",
    "ELM":        "ELM",
    "RR":         "RR",
    "TAR":        "TAR",
    "PB&J":       "PB&J",
}
DEFAULT_METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]

# Map safe filenames → display names used as index in summary tables
SAFE_TO_DISPLAY = {"PB_J": "PB&J", "RMU_LAT": "RMU-LAT"}

def _display_name(m: str) -> str:
    """Resolve safe filename variant to the name used in summary table index."""
    return SAFE_TO_DISPLAY.get(m, m)
OFFSETS = {
    "GradDiff": ( 8,  5),
    "RMU":      ( 8,  5),
    "RMU-LAT":  ( 8, -12),
    "RepNoise": ( 8,  5),
    "ELM":      ( 8, -12),
    "RR":       ( 8,  5),
    "TAR":      ( 8,  5),
    "PB&J":     ( 8,  5),
}


def load_attack_scores(data_dir: Path) -> Optional[pd.DataFrame]:
    tgat_path = data_dir / "LLM-GAT_summary_table.csv"
    if not tgat_path.exists():
        return None
    tgat = pd.read_csv(tgat_path, index_col=0)
    df = pd.DataFrame({"method": list(GAT_NAME_MAP.values())})
    df["attack_tamp"] = df["method"].map({our: float(tgat.loc[gat, "WMDP, Best Tamp. Attack"]) for gat, our in GAT_NAME_MAP.items()})
    df["attack_input"] = df["method"].map({our: float(tgat.loc[gat, "WMDP, Best Input Attack"]) for gat, our in GAT_NAME_MAP.items()})
    return df


def compute_summary_re_dr(
    data_dir: Path,
    methods: List[str],
    re_band: str,
    re_clf: str,
    dr_band: str,
    dr_clf: str,
    metric: str,
) -> pd.DataFrame:
    """Compute RE and summary/proxy DR from the summary tables.

    RE = (APP_base - APP_post) / (APP_base - 0.5)
    DR_fwd = (APP_post - ApP) / (APP_post - 0.5)
    DR_bwd = (APP_base - APp) / (APP_base - 0.5)
    DR = (DR_fwd + DR_bwd) / 2

    where A can be either accuracy or AUC.
    """
    t2 = pd.read_csv(data_dir / "summary_table2_base_probes.csv", index_col=0)
    t3 = pd.read_csv(data_dir / "summary_table3_method_probes.csv", index_col=0)
    t5 = pd.read_csv(data_dir / "summary_table5_cross_probes.csv", index_col=0)

    RE_COL  = f"{re_band}_{re_clf}_{metric}"
    DR_COL  = f"{dr_band}_{dr_clf}_{metric}"
    APp_COL = DR_COL
    APP_base = float(t2.loc["Base", RE_COL])

    rows = []
    for m in methods:
        dm = _display_name(m)
        APP_post = float(t3.loc[dm, RE_COL])
        ApP = float(t2.loc[dm, DR_COL])
        APp = float(t5.loc[dm, APp_COL])

        denom_re = APP_base - 0.5
        RE = np.nan if abs(denom_re) < 1e-12 else (APP_base - APP_post) / denom_re

        denom_fwd = APP_post - 0.5
        DR_fwd = np.nan if abs(denom_fwd) < 1e-12 else (APP_post - ApP) / denom_fwd
        denom_bwd = APP_base - 0.5
        DR_bwd = np.nan if abs(denom_bwd) < 1e-12 else (APP_base - APp) / denom_bwd
        DR = np.nanmean([DR_fwd, DR_bwd])

        rows.append({
            "method": dm,
            "metric": metric,
            "RE": RE,
            "DR": DR,
            "DR_fwd": DR_fwd,
            "DR_bwd": DR_bwd,
            "APP_base": APP_base,
            "APP_post": APP_post,
            "ApP": ApP,
            "APp": APp,
            "re_col": RE_COL,
            "dr_col": DR_COL,
            "dr_mode": "summary_proxy",
        })
    return pd.DataFrame(rows)


def add_geometry_dr(
    df: pd.DataFrame,
    geometry_dr: Dict[str, float],
) -> pd.DataFrame:
    out = df.copy()
    out["DR_summary"] = out["DR"]
    out["DR"] = out["method"].map(geometry_dr)
    out["dr_mode"] = "geometry_1_minus_cos"
    return out


def make_axis_labels(metric: str, dr_mode: str) -> Tuple[str, str, str]:
    metric_disp = metric.upper()
    y = (
        f"RE -- Representational Erasure ({metric_disp})\n"
        r"$\frac{A_{PP}^{\rm base} - A_{PP}^{\rm post}}{A_{PP}^{\rm base} - 0.5}$"
    )
    if dr_mode == "geometry_1_minus_cos":
        x = (
            "DR -- Geometry-based Directional Rotation\n"
            r"$1 - \cos(\mathbf{w}_{pre},\mathbf{w}_{post})$"
        )
        subtitle = "DR from cosine displacement of linear-probe weight vectors"
    else:
        x = (
            f"DR -- Directional Rotation proxy ({metric_disp})\n"
            r"$\frac{1}{2}\!\left["
            r"\frac{A_{PP}^{\rm post}{-}A_{pP}}{A_{PP}^{\rm post}{-}0.5}"
            r"+\frac{A_{PP}^{\rm base}{-}A_{Pp}}{A_{PP}^{\rm base}{-}0.5}"
            r"\right]$"
        )
        subtitle = "DR from probe-transfer summary tables"
    return x, y, subtitle


def draw_panel(ax, df: pd.DataFrame, attack_col: str, cbar_label: str, shared_norm: Optional[Normalize], metric: str):
    xlab, ylab, subtitle = make_axis_labels(metric, str(df["dr_mode"].iloc[0]))

    if attack_col in df.columns and df[attack_col].notna().any():
        cmap = plt.get_cmap("RdYlGn_r")
        sc = ax.scatter(
            df["DR"], df["RE"],
            c=df[attack_col], cmap=cmap, norm=shared_norm,
            s=220, zorder=5, edgecolors="k", linewidths=0.7,
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=10)
    else:
        sc = ax.scatter(df["DR"], df["RE"], s=220, zorder=5, edgecolors="k", linewidths=0.7)

    for _, row in df.iterrows():
        ox, oy = OFFSETS.get(row["method"], (8, 5))
        ax.annotate(str(row["method"]), xy=(row["DR"], row["RE"]),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=9, fontweight="semibold")

    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(subtitle, fontsize=10)
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1, color="steelblue",  lw=0.7, ls=":", alpha=0.4, label="RE=1")
    ax.axvline(1, color="darkorange", lw=0.7, ls=":", alpha=0.4, label="DR=1")
    ax.legend(fontsize=8, loc="lower right")
    valid_x = df["DR"].dropna()
    valid_y = df["RE"].dropna()
    xm = (valid_x.max() - valid_x.min()) * 0.15 + 0.05 if len(valid_x) else 0.2
    ym = (valid_y.max() - valid_y.min()) * 0.15 + 0.05 if len(valid_y) else 0.2
    if len(valid_x):
        ax.set_xlim(valid_x.min() - xm, valid_x.max() + xm)
    if len(valid_y):
        ax.set_ylim(valid_y.min() - ym, valid_y.max() + ym)


def plot_metric(df: pd.DataFrame, out_png: Path, title_prefix: str = "Hidden Knowledge After Unlearning"):
    attack_cols_present = [c for c in ["attack_tamp", "attack_input"] if c in df.columns and df[c].notna().any()]
    metric = str(df["metric"].iloc[0])

    if len(attack_cols_present) >= 2:
        all_vals = pd.concat([df[attack_cols_present[0]], df[attack_cols_present[1]]])
        shared_norm = Normalize(vmin=all_vals.min(), vmax=all_vals.max())
        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        fig.suptitle(f"{title_prefix}\nMetric={metric.upper()}", fontsize=13, y=1.01)
        draw_panel(axes[0], df, attack_cols_present[0], "Best Tampering Attack (WMDP, LLM-GAT)", shared_norm, metric)
        axes[0].set_title("Color = Best Tampering Attack", fontsize=11)
        draw_panel(axes[1], df, attack_cols_present[1], "Best Input Attack (WMDP, LLM-GAT)", shared_norm, metric)
        axes[1].set_title("Color = Best Input Attack", fontsize=11)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 7))
        fig.suptitle(f"{title_prefix}\nMetric={metric.upper()}", fontsize=13, y=0.98)
        shared_norm = None
        draw_panel(ax, df, attack_cols_present[0] if attack_cols_present else "", "", shared_norm, metric)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=".", help="Directory containing the summary tables")
    ap.add_argument("--re_band",  default="mb", help="Probe band for RE (default: mb)")
    ap.add_argument("--re_clf",   default="lr", help="Classifier for RE (default: lr)")
    ap.add_argument("--dr_band",  default="mb", help="Probe band for DR (default: mb)")
    ap.add_argument("--dr_clf",   default="lr", help="Classifier for DR (default: lr)")
    ap.add_argument("--metrics",  default="acc,auc", help="Comma-separated metrics to plot, e.g. acc,auc")
    ap.add_argument("--methods",  default=",".join(DEFAULT_METHODS), help="Comma-separated method list")
    ap.add_argument("--out_dir",  default="re_dr_outputs", help="Output directory")
    ap.add_argument("--no_attack_colors", action="store_true", help="Ignore LLM-GAT attack table even if present")

    # Geometry DR options
    ap.add_argument("--dr_mode", choices=["summary", "geometry"], default="summary",
                    help="Use summary/proxy DR or geometry-based DR from 1-cos(probe weights)")
    ap.add_argument("--geometry_input_dir", default=None,
                    help="Directory containing <method>_hs_train.npy etc. Required for --dr_mode geometry")
    ap.add_argument("--shared_df", default=None,
                    help="CSV with split/label columns, e.g. wmdp_tf_pairs.csv. Required for --dr_mode geometry")
    ap.add_argument("--base_method", default="base", help="Method name for the base hidden states in geometry mode")
    ap.add_argument("--split_col", default="split", help="Split column in shared_df")
    ap.add_argument("--label_col", default="label", help="Binary label column in shared_df with True/False values")
    ap.add_argument("--train_split", default="train", help="Which split to use to fit the linear probes in geometry mode")
    ap.add_argument("--geometry_layers", default="0-31", help="Layer spec for geometry DR, e.g. 0-31 or 10,11,12")
    ap.add_argument("--geometry_mode", choices=["mean", "concat"], default="concat", help="Band compression mode")
    ap.add_argument("--geometry_pca_dim_per_layer", type=int, default=0, help="Optional per-layer PCA dim")
    ap.add_argument("--geometry_C", type=float, default=1.0, help="LogReg C for geometry probes")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    attacks = None if args.no_attack_colors else load_attack_scores(data_dir)

    geometry_dr_by_metric: Dict[str, Dict[str, float]] = {}
    if args.dr_mode == "geometry":
        if not args.geometry_input_dir or not args.shared_df:
            raise SystemExit("--dr_mode geometry requires both --geometry_input_dir and --shared_df")
        if args.dr_clf != "lr":
            raise SystemExit("Geometry DR requires --dr_clf lr because it uses linear probe weights.")
        geometry_dr = compute_geometry_dr_from_probe_weights(
            geometry_input_dir=Path(args.geometry_input_dir),
            shared_df_path=Path(args.shared_df),
            base_method=args.base_method,
            methods=methods,
            band=args.dr_band,
            clf=args.dr_clf,
            split_col=args.split_col,
            label_col=args.label_col,
            train_split=args.train_split,
            layers_spec=args.geometry_layers,
            mode=args.geometry_mode,
            pca_dim_per_layer=args.geometry_pca_dim_per_layer,
            C=args.geometry_C,
        )
        # same geometry DR used regardless of summary metric acc/auc
        for metric in metrics:
            geometry_dr_by_metric[metric] = geometry_dr

    manifest_rows = []
    for metric in metrics:
        df = compute_summary_re_dr(
            data_dir=data_dir,
            methods=methods,
            re_band=args.re_band,
            re_clf=args.re_clf,
            dr_band=args.dr_band,
            dr_clf=args.dr_clf,
            metric=metric,
        )
        if attacks is not None:
            df = df.merge(attacks, on="method", how="left")
        if args.dr_mode == "geometry":
            df = add_geometry_dr(df, geometry_dr_by_metric[metric])

        csv_path = out_dir / f"RE_DR_table_{metric}_{args.re_band}_{args.re_clf}_RE_{args.dr_band}_{args.dr_clf}_{args.dr_mode}.csv"
        png_path = out_dir / f"RE_DR_plot_{metric}_{args.re_band}_{args.re_clf}_RE_{args.dr_band}_{args.dr_clf}_{args.dr_mode}.png"
        df.to_csv(csv_path, index=False)
        plot_metric(df, png_path)
        manifest_rows.append({"metric": metric, "csv": str(csv_path), "png": str(png_path), "dr_mode": args.dr_mode})
        print(f"Saved {metric.upper()} table -> {csv_path}")
        print(f"Saved {metric.upper()} plot  -> {png_path}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Saved manifest -> {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
