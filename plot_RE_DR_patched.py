#!/usr/bin/env python3
"""
plot_RE_DR_patched.py  (v2)

Computes and plots Representational Erasure (RE) vs Directional Rotation (DR)
for unlearning methods.

New in v2
---------
- Per-method colored/shaped scatter markers (matching dual-confidence line plots).
- Attack colors are opt-in via --attack_colors (default: method-color scatter).
- Figure size matches line plots (12 x 7).
- Checkpoint trajectory: overlay RE/DR path across up to 8 training checkpoints
  of a chosen method (--ckpt_method / --ckpt_dir / --ckpt_steps).
- SLURM job-array support for parallel geometry DR computation:
    --generate_slurm   print a ready-to-submit job-array script and exit
    --slurm_task_method METHOD   compute & save geometry DR for one method
    --slurm_collect    load all partial results and produce final plots

Geometry checkpoint file naming convention
------------------------------------------
  {safe_name}_hs_train.npy
  {safe_name}_ckpt{step}_hs_train.npy   (checkpoint, step = int or string label)

SLURM workflow
--------------
  python plot_RE_DR_patched.py --generate_slurm [all other flags] > run_re_dr.sh
  sbatch run_re_dr.sh
  # After all array jobs finish:
  python plot_RE_DR_patched.py --slurm_collect [all other flags]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Method styling (matches dual-confidence line plots) ──────────────────────
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
METHOD_MARKERS = {
    "Base": "o", "GradDiff": "s", "RMU": "^", "RMU-LAT": "D",
    "RepNoise": "P", "ELM": "X", "RR": "*", "TAR": "h", "PB&J": "v",
}
_FALLBACK_COLOR  = "#888888"
_FALLBACK_MARKER = "o"

# ── Name normalisation ────────────────────────────────────────────────────────
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
        vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
        if not vals:
            raise ValueError(f"Invalid layer spec: {spec}")
        return vals
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(spec)]


class BandCompressor:
    """Compress a band of hidden states to a fixed vector per example."""

    def __init__(self, layers: List[int], mode: str = "concat", pca_dim_per_layer: int = 0):
        self.layers = list(layers)
        self.mode   = mode
        self.pca_dim_per_layer = int(pca_dim_per_layer)
        self.layer_pcas: List[Optional[PCA]] = []

    def fit(self, hs: np.ndarray) -> "BandCompressor":
        if self.mode == "mean":
            self.layer_pcas = []
            return self
        if self.mode != "concat":
            raise ValueError(f"Unsupported mode: {self.mode}")
        band = hs[:, self.layers, :]
        if self.pca_dim_per_layer > 0:
            self.layer_pcas = []
            for i in range(band.shape[1]):
                k = min(self.pca_dim_per_layer, band.shape[0], band.shape[2])
                pca = PCA(n_components=k, random_state=42)
                pca.fit(band[:, i, :])
                self.layer_pcas.append(pca)
        else:
            self.layer_pcas = [None] * band.shape[1]
        return self

    def transform(self, hs: np.ndarray) -> np.ndarray:
        band = hs[:, self.layers, :]
        if self.mode == "mean":
            return band.mean(axis=1)
        if self.pca_dim_per_layer > 0:
            return np.concatenate(
                [self.layer_pcas[i].transform(band[:, i, :]) for i in range(band.shape[1])],
                axis=1)
        return band.reshape(hs.shape[0], -1)


def build_lr_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    LogisticRegression(max_iter=4000, C=C, random_state=42)),
    ])


def get_lr_weight(pipe: Pipeline) -> np.ndarray:
    return np.asarray(pipe.named_steps["clf"].coef_).reshape(-1)


def cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return None if (na == 0 or nb == 0) else float(np.dot(a, b) / (na * nb))


def dr_from_weights(w_pre: np.ndarray, w_post: np.ndarray) -> Optional[float]:
    c = cosine(w_pre, w_post)
    return None if c is None else float(1.0 - c)


def discover_hs_methods(input_dir: Path) -> Dict[str, Dict[str, str]]:
    methods: Dict[str, Dict[str, str]] = {}
    pat = re.compile(r"^(?P<method>.+)_hs_(?P<split>train|val|test)\.npy$")
    for p in input_dir.iterdir():
        m = pat.match(p.name)
        if m:
            methods.setdefault(m.group("method"), {})[f"hs_{m.group('split')}"] = str(p)
    return methods


def _fit_base_probe(
    base_hs_train: np.ndarray,
    y_train: np.ndarray,
    layers: List[int],
    mode: str,
    pca_dim: int,
    C: float,
) -> Tuple["BandCompressor", Pipeline, np.ndarray]:
    """Returns (compressor, fitted_pipeline, weight_vector)."""
    comp = BandCompressor(layers, mode=mode, pca_dim_per_layer=pca_dim).fit(base_hs_train)
    X    = comp.transform(base_hs_train)
    pipe = build_lr_pipeline(C=C)
    pipe.fit(X, y_train)
    return comp, pipe, get_lr_weight(pipe)


def compute_geometry_dr_for_one_method(
    method_hs_train: np.ndarray,
    y_train: np.ndarray,
    w_pre: np.ndarray,
    layers: List[int],
    mode: str,
    pca_dim: int,
    C: float,
) -> float:
    N = min(len(method_hs_train), len(y_train))
    comp = BandCompressor(layers, mode=mode, pca_dim_per_layer=pca_dim).fit(method_hs_train[:N])
    X    = comp.transform(method_hs_train[:N])
    pipe = build_lr_pipeline(C=C)
    pipe.fit(X, y_train[:N])
    dr   = dr_from_weights(w_pre, get_lr_weight(pipe))
    return np.nan if dr is None else float(dr)


def compute_geometry_dr_from_probe_weights(
    geometry_input_dir: Path,
    shared_df_path: Path,
    base_method: str,
    methods: Iterable[str],
    layers_spec: str,
    mode: str,
    pca_dim_per_layer: int,
    C: float,
    split_col: str,
    label_col: str,
    train_split: str,
) -> Dict[str, float]:
    methods_files = discover_hs_methods(geometry_input_dir)
    if base_method not in methods_files or "hs_train" not in methods_files[base_method]:
        raise FileNotFoundError(f"Base hidden states not found for '{base_method}' in {geometry_input_dir}")

    shared_df = pd.read_csv(shared_df_path)
    train_df  = shared_df[shared_df[split_col].astype(str) == str(train_split)].reset_index(drop=True)
    y_train   = (train_df[label_col].astype(str).str.lower().str.strip() == "true").astype(int).to_numpy()
    layers    = parse_layer_spec(layers_spec)

    base_hs_train = load_hidden_states(methods_files[base_method]["hs_train"])
    _, _, w_pre   = _fit_base_probe(base_hs_train, y_train, layers, mode, pca_dim_per_layer, C)

    out: Dict[str, float] = {}
    for method in methods:
        if method == base_method:
            continue
        files = methods_files.get(method)
        if not files or "hs_train" not in files:
            raise FileNotFoundError(f"Hidden states for method '{method}' not found in {geometry_input_dir}")
        hs_train = load_hidden_states(files["hs_train"])
        out[method] = compute_geometry_dr_for_one_method(
            hs_train, y_train, w_pre, layers, mode, pca_dim_per_layer, C)
    return out


# ── Checkpoint trajectory from sweep CSVs ─────────────────────────────────────
#
# Checkpoint data lives in:  data/sweep_{sn}/{sn}_sweep.csv
# Columns: checkpoint (1-8), mp_{clf}_{band}_{metric}, bp_{clf}_{band}_{metric}
#          optionally xp_{clf}_{band}_{metric}
#
#   mp_ = APP_post  (method probe → method model)
#   bp_ = ApP       (base  probe → method model)
#   xp_ = APp       (method probe → base  model)  — enables full symmetric DR

N_CHECKPOINTS = 8


def _safe_name_sweep(m: str) -> str:
    """Match hidden_knowledge_after_unlearning.safe_name() for sweep dir names."""
    return m.replace("&", "_").replace("/", "_").replace(" ", "_")


def load_sweep_trajectory(
    data_dir: Path,
    method: str,
    re_band: str,
    re_clf: str,
    dr_band: str,
    dr_clf: str,
    metric: str,
    checkpoints: Optional[List[int]],
    app_base: float,
) -> Optional[pd.DataFrame]:
    """Load RE and DR trajectory from the precomputed sweep CSV.

    Returns DataFrame with columns: step (int), RE, DR (or DR_fwd if xp_ absent).
    Returns None if the sweep CSV is missing.
    """
    sn   = _safe_name_sweep(method)
    path = data_dir / f"sweep_{sn}" / f"{sn}_sweep.csv"
    if not path.exists():
        print(f"  [warn] Sweep CSV not found: {path}")
        return None

    df = pd.read_csv(path)

    mp_col = f"mp_{re_clf}_{re_band}_{metric}"
    bp_col = f"bp_{re_clf}_{re_band}_{metric}"
    xp_col = f"xp_{dr_clf}_{dr_band}_{metric}"

    for col in (mp_col, bp_col):
        if col not in df.columns:
            print(f"  [warn] Column '{col}' missing in {path}")
            return None

    if checkpoints is not None:
        df = df[df["checkpoint"].isin(checkpoints)]

    rows = []
    for _, row in df.iterrows():
        ck       = int(row["checkpoint"])
        app_post = float(row[mp_col])
        apP      = float(row[bp_col])

        denom_re = app_base - 0.5
        RE = np.nan if abs(denom_re) < 1e-12 else (app_base - app_post) / denom_re

        denom_fwd = app_post - 0.5
        DR_fwd = np.nan if abs(denom_fwd) < 1e-12 else (app_post - apP) / denom_fwd

        if xp_col in df.columns and not pd.isna(row.get(xp_col, np.nan)):
            aPp    = float(row[xp_col])
            DR_bwd = (app_base - aPp) / denom_re if abs(denom_re) > 1e-12 else np.nan
            DR     = float(np.nanmean([DR_fwd, DR_bwd]))
        else:
            DR = DR_fwd  # fall back to forward-only

        rows.append({"step": ck, "RE": RE, "DR": DR})

    if not rows:
        return None
    result = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    # Prepend base (ck0): RE=0, DR=0 by definition
    base_row = pd.DataFrame([{"step": 0, "RE": 0.0, "DR": 0.0}])
    return pd.concat([base_row, result], ignore_index=True)


def compute_geometry_dr_for_checkpoints(
    method: str,
    ck_nums: List[int],
    geometry_input_dir: Path,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_pre: np.ndarray,
    layers: List[int],
    mode: str,
    pca_dim: int,
    C: float,
) -> Dict[int, float]:
    """Geometry DR for each checkpoint, using the same w_pre and layer spec as the main scatter.

    Looks for sweep_{sn}/ck{N}/hs_train.npy first (consistent with main scatter),
    falls back to hs_test.npy with matching test labels if train is absent.
    ck8's hs_train should be identical to {sn}_hs_train.npy so it lands on the method dot.
    """
    sn  = _safe_name_sweep(method)
    out: Dict[int, float] = {}
    for ck in ck_nums:
        ck_dir     = geometry_input_dir / f"sweep_{sn}" / f"ck{ck}"
        train_path = ck_dir / "hs_train.npy"
        test_path  = ck_dir / "hs_test.npy"

        if train_path.exists():
            hs   = load_hidden_states(str(train_path))
            y_ck = y_train
        elif test_path.exists():
            hs   = load_hidden_states(str(test_path))
            y_ck = y_test
        else:
            print(f"  [warn] No hidden states for {method} ck{ck} — skipping")
            continue

        N  = min(len(hs), len(y_ck))
        dr = compute_geometry_dr_for_one_method(
            hs[:N], y_ck[:N], w_pre, layers, mode, pca_dim, C)
        out[ck] = dr
    return out


# ── Summary-table RE/DR ───────────────────────────────────────────────────────

GAT_NAME_MAP = {
    "Grad Diff": "GradDiff", "RMU": "RMU", "RMU + LAT": "RMU-LAT",
    "RepNoise": "RepNoise",  "ELM": "ELM", "RR": "RR",
    "TAR": "TAR",            "PB&J": "PB&J",
}


def load_attack_scores(data_dir: Path) -> Optional[pd.DataFrame]:
    p = data_dir / "LLM-GAT_summary_table.csv"
    if not p.exists():
        return None
    tgat = pd.read_csv(p, index_col=0)
    df   = pd.DataFrame({"method": list(GAT_NAME_MAP.values())})
    df["attack_tamp"]  = df["method"].map(
        {our: float(tgat.loc[gat, "WMDP, Best Tamp. Attack"])
         for gat, our in GAT_NAME_MAP.items()})
    df["attack_input"] = df["method"].map(
        {our: float(tgat.loc[gat, "WMDP, Best Input Attack"])
         for gat, our in GAT_NAME_MAP.items()})
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
    t2 = pd.read_csv(data_dir / "summary_table2_base_probes.csv",   index_col=0)
    t3 = pd.read_csv(data_dir / "summary_table3_method_probes.csv", index_col=0)
    t5 = pd.read_csv(data_dir / "summary_table5_cross_probes.csv",  index_col=0)

    RE_COL   = f"{re_band}_{re_clf}_{metric}"
    DR_COL   = f"{dr_band}_{dr_clf}_{metric}"
    APP_base = float(t2.loc["Base", RE_COL])

    rows = []
    for m in methods:
        dm       = _display_name(m)
        APP_post = float(t3.loc[dm, RE_COL])
        ApP      = float(t2.loc[dm, DR_COL])
        APp      = float(t5.loc[dm, DR_COL])
        denom_re = APP_base - 0.5
        RE       = np.nan if abs(denom_re) < 1e-12 else (APP_base - APP_post) / denom_re
        denom_fwd = APP_post - 0.5
        DR_fwd   = np.nan if abs(denom_fwd) < 1e-12 else (APP_post - ApP) / denom_fwd
        denom_bwd = APP_base - 0.5
        DR_bwd   = np.nan if abs(denom_bwd) < 1e-12 else (APP_base - APp) / denom_bwd
        DR       = float(np.nanmean([DR_fwd, DR_bwd]))
        rows.append({"method": dm, "metric": metric, "RE": RE, "DR": DR,
                     "DR_fwd": DR_fwd, "DR_bwd": DR_bwd,
                     "APP_base": APP_base, "APP_post": APP_post,
                     "ApP": ApP, "APp": APp,
                     "re_col": RE_COL, "dr_col": DR_COL,
                     "dr_mode": "summary_proxy"})
    return pd.DataFrame(rows)


def add_geometry_dr(df: pd.DataFrame, geometry_dr: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["DR_summary"] = out["DR"]
    out["DR"]         = out["method"].map(geometry_dr)
    out["dr_mode"]    = "geometry_1_minus_cos"
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def _dr_axis_label(dr_mode: str) -> str:
    if dr_mode == "geometry_1_minus_cos":
        return ("DR — Directional Rotation (geometry)\n"
                r"$1 - \cos(\mathbf{w}_{\rm pre},\,\mathbf{w}_{\rm post})$")
    return ("DR — Directional Rotation proxy\n"
            r"$\frac{1}{2}\!\left["
            r"\frac{A_{PP}^{\rm post}{-}A_{pP}}{A_{PP}^{\rm post}{-}0.5}"
            r"+\frac{A_{PP}^{\rm base}{-}A_{Pp}}{A_{PP}^{\rm base}{-}0.5}"
            r"\right]$")


def _re_axis_label(metric: str) -> str:
    return (f"RE — Representational Erasure ({metric.upper()})\n"
            r"$\frac{A_{PP}^{\rm base} - A_{PP}^{\rm post}}{A_{PP}^{\rm base} - 0.5}$")


def draw_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    attack_col: Optional[str],
    shared_norm: Optional[Normalize],
    ckpt_trajectories: Optional[Dict[str, Optional[pd.DataFrame]]] = None,
) -> None:
    dr_mode = str(df["dr_mode"].iloc[0])

    # Grid and reference lines
    ax.set_facecolor("#fafafa")
    ax.set_axisbelow(True)
    ax.grid(color="grey", alpha=0.18, lw=0.6)
    ax.axhline(0, color="grey",       lw=0.9, ls="--", alpha=0.45, zorder=2)
    ax.axvline(0, color="grey",       lw=0.9, ls="--", alpha=0.45, zorder=2)
    ax.axhline(1, color="steelblue",  lw=0.8, ls=":",  alpha=0.55, zorder=2)
    ax.axvline(1, color="darkorange", lw=0.8, ls=":",  alpha=0.55, zorder=2)

    # ── Checkpoint trajectories (behind main scatter) ────────────────────────
    traj_items = list((ckpt_trajectories or {}).items())
    vx_extra: List[pd.Series] = []
    vy_extra: List[pd.Series] = []
    for cm, cdf in traj_items:
        if cdf is None or cdf.empty:
            continue
        dm_cm      = _display_name(cm)
        ckpt_color = METHOD_COLORS.get(dm_cm, _FALLBACK_COLOR)
        ax.plot(cdf["DR"].to_numpy(), cdf["RE"].to_numpy(),
                "-o", color=ckpt_color, alpha=0.45, lw=1.8, ms=5, zorder=3)
        for _, row in cdf.iterrows():
            lbl = "base" if int(row["step"]) == 0 else f"ck{row['step']}"
            ax.annotate(lbl, xy=(row["DR"], row["RE"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=ckpt_color, alpha=0.75)
        vx_extra.append(cdf["DR"].dropna())
        vy_extra.append(cdf["RE"].dropna())

    # ── Main scatter ──────────────────────────────────────────────────────────
    if attack_col and attack_col in df.columns and df[attack_col].notna().any():
        cmap = plt.get_cmap("RdYlGn_r")
        sc   = ax.scatter(df["DR"], df["RE"],
                          c=df[attack_col], cmap=cmap, norm=shared_norm,
                          s=280, zorder=5, edgecolors="k", linewidths=0.8)
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
        cbar.set_label(attack_col.replace("_", " ").title(), fontsize=9)
    else:
        for _, row in df.iterrows():
            dm = str(row["method"])
            ax.scatter(row["DR"], row["RE"],
                       color=METHOD_COLORS.get(dm, _FALLBACK_COLOR),
                       marker="o",
                       s=280, zorder=5, edgecolors="k", linewidths=0.8)

    # ── Method labels ─────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        ox, oy = OFFSETS.get(str(row["method"]), (8, 5))
        ax.annotate(str(row["method"]),
                    xy=(row["DR"], row["RE"]),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=9, fontweight="semibold")

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax.set_xlabel(_dr_axis_label(dr_mode), fontsize=10)
    ax.set_ylabel(_re_axis_label(metric),  fontsize=10)

    # ── Auto-scale (including trajectory range) ───────────────────────────────
    vx = pd.concat([df["DR"].dropna()] + vx_extra)
    vy = pd.concat([df["RE"].dropna()] + vy_extra)
    if len(vx):
        xm = max((vx.max() - vx.min()) * 0.20 + 0.05, 0.12)
        ax.set_xlim(vx.min() - xm, vx.max() + xm)
    if len(vy):
        ym = max((vy.max() - vy.min()) * 0.20 + 0.05, 0.12)
        ax.set_ylim(vy.min() - ym, vy.max() + ym)

    # ── Legend (method colors + trajectory entries) ────────────────────────────
    if not (attack_col and attack_col in df.columns and df[attack_col].notna().any()):
        handles = [
            Line2D([0], [0],
                   marker="o", color="w",
                   markerfacecolor=METHOD_COLORS.get(m, _FALLBACK_COLOR),
                   markeredgecolor="k", markeredgewidth=0.7,
                   markersize=9, label=m)
            for m in df["method"].tolist()
        ]
        for cm, cdf in traj_items:
            if cdf is None or cdf.empty:
                continue
            dm_cm      = _display_name(cm)
            ckpt_color = METHOD_COLORS.get(dm_cm, _FALLBACK_COLOR)
            handles.append(Line2D([0], [0], color=ckpt_color, lw=2.0,
                                  alpha=0.6, label=f"{dm_cm} checkpoints"))
        ax.legend(handles=handles, fontsize=8.5, loc="best",
                  framealpha=0.88, edgecolor="#cccccc")


def plot_metric(
    df: pd.DataFrame,
    out_png: Path,
    title_prefix: str,
    attack_col: Optional[str],
    shared_norm: Optional[Normalize],
    ckpt_trajectories: Optional[Dict[str, Optional[pd.DataFrame]]] = None,
) -> None:
    metric = str(df["metric"].iloc[0])
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{title_prefix} — {metric.upper()}", fontsize=13, y=0.98)
    draw_scatter(ax, df, metric, attack_col, shared_norm, ckpt_trajectories)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


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
#SBATCH --job-name=re_dr_geom
#SBATCH --partition={partition}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time_limit}
#SBATCH --array=0-{n - 1}
#SBATCH --output=logs/re_dr_%A_%a.out

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

METHODS='{methods_json}'
METHOD=$(python3 -c "import sys,json; ms=json.loads(sys.argv[1]); print(ms[int(sys.argv[2])])" "$METHODS" "$SLURM_ARRAY_TASK_ID")

python {script_path} \\
  {passthrough_args} \\
  --slurm_task_method "$METHOD" \\
  --slurm_partial_dir "{partial_dir}"
"""


# ── CLI / main ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # Summary tables
    ap.add_argument("--data_dir",  default=".",    help="Directory containing summary CSV tables")
    ap.add_argument("--re_band",   default="mb",   help="Probe band for RE column (default: mb)")
    ap.add_argument("--re_clf",    default="lr",   help="Classifier for RE (default: lr)")
    ap.add_argument("--dr_band",   default="mb",   help="Probe band for summary DR (default: mb)")
    ap.add_argument("--dr_clf",    default="lr",   help="Classifier for summary DR (default: lr)")
    ap.add_argument("--metrics",   default="acc,auc")
    ap.add_argument("--methods",   default=",".join(DEFAULT_METHODS))
    ap.add_argument("--out_dir",   default="re_dr_outputs")

    # Attack colors (opt-in)
    ap.add_argument("--attack_colors", action="store_true",
                    help="Color points by LLM-GAT attack scores (default: per-method colors)")

    # Geometry DR
    ap.add_argument("--dr_mode",   choices=["summary", "geometry"], default="summary")
    ap.add_argument("--geometry_input_dir",           default=None)
    ap.add_argument("--shared_df",                    default=None)
    ap.add_argument("--base_method",                  default="base")
    ap.add_argument("--split_col",                    default="split")
    ap.add_argument("--label_col",                    default="label")
    ap.add_argument("--train_split",                  default="train")
    ap.add_argument("--geometry_layers",              default="0-31")
    ap.add_argument("--geometry_mode", choices=["mean", "concat"], default="concat")
    ap.add_argument("--geometry_pca_dim_per_layer",   type=int, default=0)
    ap.add_argument("--geometry_C",                   type=float, default=1.0)

    # Checkpoint trajectory (reads from precomputed sweep CSVs in data_dir)
    ap.add_argument("--ckpt_method", default=None,
                    help="Method to overlay sweep trajectory for, e.g. GradDiff  (or 'all')")
    ap.add_argument("--ckpt_checkpoints", default=None,
                    help="Comma-separated checkpoint numbers 1-8; all if omitted")

    # SLURM
    ap.add_argument("--generate_slurm",    action="store_true",
                    help="Print a SLURM job-array script to stdout and exit")
    ap.add_argument("--slurm_partition",   default="public")
    ap.add_argument("--slurm_mem",         default="32G")
    ap.add_argument("--slurm_cpus",        type=int, default=4)
    ap.add_argument("--slurm_time",        default="00:30:00")
    ap.add_argument("--slurm_partial_dir", default="re_dr_partial",
                    help="Dir for per-method geometry DR JSON files")
    ap.add_argument("--slurm_task_method", default=None,
                    help="Compute geometry DR for this one method and save; used by job array")
    ap.add_argument("--slurm_collect",     action="store_true",
                    help="Collect partial results from --slurm_partial_dir and produce plots")

    args    = ap.parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods  = [m.strip() for m in args.methods.split(",") if m.strip()]
    metrics  = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # ── SLURM: generate script and exit ──────────────────────────────────────
    if args.generate_slurm:
        passthrough = [
            f"--data_dir {args.data_dir}",
            f"--re_band {args.re_band}", f"--re_clf {args.re_clf}",
            f"--dr_band {args.dr_band}", f"--dr_clf {args.dr_clf}",
            f"--metrics {args.metrics}", f"--methods {args.methods}",
            f"--out_dir {args.out_dir}",
            f"--dr_mode {args.dr_mode}",
            f"--base_method {args.base_method}",
            f"--geometry_layers {args.geometry_layers}",
            f"--geometry_mode {args.geometry_mode}",
            f"--geometry_C {args.geometry_C}",
        ]
        if args.geometry_input_dir: passthrough.append(f"--geometry_input_dir {args.geometry_input_dir}")
        if args.shared_df:          passthrough.append(f"--shared_df {args.shared_df}")
        if args.attack_colors:      passthrough.append("--attack_colors")
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

    # ── SLURM: single-method geometry task ────────────────────────────────────
    if args.slurm_task_method is not None:
        method      = args.slurm_task_method
        partial_dir = Path(args.slurm_partial_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)
        out_json    = partial_dir / f"{_safe_name(method)}_geometry_dr.json"
        if out_json.exists():
            print(f"[skip] {out_json} already exists.")
            return
        if not args.geometry_input_dir or not args.shared_df:
            sys.exit("--geometry_input_dir and --shared_df required for SLURM task mode")

        shared_df_t   = pd.read_csv(args.shared_df)
        train_df_t    = shared_df_t[shared_df_t[args.split_col].astype(str) == args.train_split].reset_index(drop=True)
        y_train_t     = (train_df_t[args.label_col].astype(str).str.lower().str.strip() == "true").astype(int).to_numpy()
        layers_t      = parse_layer_spec(args.geometry_layers)
        methods_files = discover_hs_methods(Path(args.geometry_input_dir))
        base_hs       = load_hidden_states(methods_files[args.base_method]["hs_train"])
        _, _, w_pre   = _fit_base_probe(base_hs, y_train_t, layers_t,
                                        args.geometry_mode, args.geometry_pca_dim_per_layer, args.geometry_C)
        files = methods_files.get(method)
        if not files or "hs_train" not in files:
            sys.exit(f"Hidden states for '{method}' not found in {args.geometry_input_dir}")
        hs_m = load_hidden_states(files["hs_train"])
        dr   = compute_geometry_dr_for_one_method(
            hs_m, y_train_t, w_pre, layers_t,
            args.geometry_mode, args.geometry_pca_dim_per_layer, args.geometry_C)
        with open(out_json, "w") as fh:
            json.dump({method: float(dr)}, fh)
        print(f"Saved geometry DR for {method}: {dr:.4f}  ->  {out_json}")
        return

    # ── Geometry DR ───────────────────────────────────────────────────────────
    geometry_dr_by_metric: Dict[str, Dict[str, float]] = {}
    if args.dr_mode == "geometry":
        if args.slurm_collect:
            partial_dir  = Path(args.slurm_partial_dir)
            geometry_dr: Dict[str, float] = {}
            for p in partial_dir.glob("*_geometry_dr.json"):
                with open(p) as fh:
                    geometry_dr.update(json.load(fh))
            print(f"Loaded geometry DR for {len(geometry_dr)} methods from {partial_dir}")
        else:
            if not args.geometry_input_dir or not args.shared_df:
                sys.exit("--dr_mode geometry requires --geometry_input_dir and --shared_df")
            geometry_dr = compute_geometry_dr_from_probe_weights(
                geometry_input_dir = Path(args.geometry_input_dir),
                shared_df_path     = Path(args.shared_df),
                base_method        = args.base_method,
                methods            = methods,
                layers_spec        = args.geometry_layers,
                mode               = args.geometry_mode,
                pca_dim_per_layer  = args.geometry_pca_dim_per_layer,
                C                  = args.geometry_C,
                split_col          = args.split_col,
                label_col          = args.label_col,
                train_split        = args.train_split,
            )
        geometry_dr = {_display_name(k): v for k, v in geometry_dr.items()}
        for metric in metrics:
            geometry_dr_by_metric[metric] = geometry_dr

    # ── Checkpoint trajectory from sweep CSVs (RE) + optional geometry DR ─────
    ckpt_dfs: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}  # metric -> method -> df
    ckpt_methods: List[str] = []
    if args.ckpt_method:
        ckpt_methods = (
            ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]
            if args.ckpt_method == "all"
            else [m.strip() for m in args.ckpt_method.split(",")]
        )
        ckpt_checkpoints: Optional[List[int]] = (
            None if not args.ckpt_checkpoints
            else [int(c) for c in args.ckpt_checkpoints.split(",")]
        )
        ck_nums = ckpt_checkpoints if ckpt_checkpoints else list(range(1, N_CHECKPOINTS + 1))

        # Pre-compute geometry ingredients once (shared across all ckpt methods & metrics)
        ckpt_w_pre: Optional[np.ndarray] = None
        ckpt_y_train: Optional[np.ndarray] = None
        ckpt_y_test:  Optional[np.ndarray] = None
        ckpt_layers: Optional[List[int]] = None
        if args.dr_mode == "geometry" and args.geometry_input_dir and args.shared_df:
            shared_ck   = pd.read_csv(args.shared_df)
            train_ck    = shared_ck[shared_ck[args.split_col].astype(str) == args.train_split].reset_index(drop=True)
            test_ck     = shared_ck[shared_ck[args.split_col].astype(str) == "test"].reset_index(drop=True)
            ckpt_y_train = (train_ck[args.label_col].astype(str).str.lower().str.strip() == "true").astype(int).to_numpy()
            ckpt_y_test  = (test_ck[args.label_col].astype(str).str.lower().str.strip()  == "true").astype(int).to_numpy()
            ckpt_layers  = parse_layer_spec(args.geometry_layers)
            mf_ck        = discover_hs_methods(Path(args.geometry_input_dir))
            base_hs_ck   = load_hidden_states(mf_ck[args.base_method]["hs_train"])
            N_b          = min(len(base_hs_ck), len(ckpt_y_train))
            _, _, ckpt_w_pre = _fit_base_probe(
                base_hs_ck[:N_b], ckpt_y_train[:N_b],
                ckpt_layers, args.geometry_mode, args.geometry_pca_dim_per_layer, args.geometry_C)

        # APP_base for RE normalisation
        t2_for_ck = pd.read_csv(data_dir / "summary_table2_base_probes.csv", index_col=0)
        for metric in metrics:
            re_col   = f"{args.re_band}_{args.re_clf}_{metric}"
            app_base = float(t2_for_ck.loc["Base", re_col]) if re_col in t2_for_ck.columns else 0.5
            ckpt_dfs[metric] = {}
            for cm in ckpt_methods:
                cdf = load_sweep_trajectory(
                    data_dir, cm,
                    args.re_band, args.re_clf,
                    args.dr_band, args.dr_clf,
                    metric, ckpt_checkpoints, app_base,
                )
                # Replace DR with geometry DR when in geometry mode
                if (cdf is not None and args.dr_mode == "geometry"
                        and ckpt_w_pre is not None and ckpt_y_train is not None):
                    geom_dr = compute_geometry_dr_for_checkpoints(
                        method=cm,
                        ck_nums=ck_nums,
                        geometry_input_dir=Path(args.geometry_input_dir),
                        y_train=ckpt_y_train,
                        y_test=ckpt_y_test,
                        w_pre=ckpt_w_pre,
                        layers=ckpt_layers,
                        mode=args.geometry_mode,
                        pca_dim=args.geometry_pca_dim_per_layer,
                        C=args.geometry_C,
                    )
                    # ck0 stays DR=0; replace ck1-ck8 with geometry values
                    cdf["DR"] = cdf["step"].apply(
                        lambda s: 0.0 if int(s) == 0 else geom_dr.get(int(s), np.nan))
                ckpt_dfs[metric][cm] = cdf

    # ── Attack colors ─────────────────────────────────────────────────────────
    attacks = load_attack_scores(data_dir) if args.attack_colors else None

    # ── Per-metric loop ───────────────────────────────────────────────────────
    manifest_rows = []
    for metric in metrics:
        df = compute_summary_re_dr(
            data_dir=data_dir, methods=methods,
            re_band=args.re_band, re_clf=args.re_clf,
            dr_band=args.dr_band, dr_clf=args.dr_clf,
            metric=metric,
        )
        if attacks is not None:
            df = df.merge(attacks, on="method", how="left")
        if args.dr_mode == "geometry":
            df = add_geometry_dr(df, geometry_dr_by_metric[metric])

        attack_col: Optional[str] = None
        shared_norm: Optional[Normalize] = None
        if args.attack_colors and attacks is not None and "attack_tamp" in df.columns:
            attack_col  = "attack_tamp"
            all_vals    = pd.concat([df["attack_tamp"], df["attack_input"]]).dropna()
            shared_norm = Normalize(vmin=all_vals.min(), vmax=all_vals.max())

        suffix   = f"{metric}_{args.re_band}_{args.re_clf}_RE_{args.dr_band}_{args.dr_clf}_{args.dr_mode}"
        csv_path = out_dir / f"RE_DR_table_{suffix}.csv"
        png_path = out_dir / f"RE_DR_plot_{suffix}.png"
        df.to_csv(csv_path, index=False)

        # Collect trajectory DataFrames for this metric
        metric_ckpt: Dict[str, Optional[pd.DataFrame]] = ckpt_dfs.get(metric, {})
        for cm, cdf in metric_ckpt.items():
            if cdf is not None and not cdf.empty:
                cdf.to_csv(out_dir / f"RE_DR_ckpt_{_safe_name(cm)}_{metric}.csv", index=False)

        plot_metric(df, png_path,
                    title_prefix="Hidden Knowledge After Unlearning",
                    attack_col=attack_col,
                    shared_norm=shared_norm,
                    ckpt_trajectories=metric_ckpt)

        manifest_rows.append({"metric": metric, "csv": str(csv_path),
                               "png": str(png_path), "dr_mode": args.dr_mode})
        print(f"  {metric.upper()} table -> {csv_path}")
        print(f"  {metric.upper()} plot  -> {png_path}")

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"Manifest -> {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
