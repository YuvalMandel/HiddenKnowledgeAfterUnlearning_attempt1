#!/usr/bin/env python3
"""
Task G: Continuous regression — base features predict trajectory targets.

Base features (8) -> predict continuous trajectory outcomes per question×method:
  1. k_int_traj_auc    — mean K_int across ck1-ck8, full-layer probe
  2. k_int_ck8         — K_int at ck8, full-layer probe
  3. ext_drop          — K_ext_base - K_ext_ck8
  4. int_ext_gap_ck8   — K_int_ck8 - K_ext_ck8

Models: Ridge, Lasso, ElasticNet, RandomForest, HistGradientBoosting
CV: RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

Outputs (plots/trajectory_regression/):
  trajectory_regression_results.csv
  trajectory_regression_results.md
  predicted_vs_actual_{target}.pdf / .png
  feature_importance_{target}.pdf / .png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_predict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "plots"))

from hk_utils import (
    METHODS, load_correct_idx, load_split_indices, load_scores_df,
    compute_prefilter_mask, get_parquet_full_single, get_method_ck8_labels,
    method_fname, compute_k_ext, EXT_DIR,
)

POOLED_FEATURES = REPO / "plots" / "base_feature_prediction" / "pooled_features.csv"
TRAJ_BY_Q       = REPO / "plots" / "ckpt_layer_trajectory_metrics" / "trajectory_metrics_by_method_subset.csv"
OUT_DIR         = REPO / "plots" / "trajectory_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "min_pairwise_sigmoid_margin",
    "max_distractor_confidence",
    "correct_option_confidence",
    "earliest_layer_kint1",
    "mean_kint_layers",
    "std_kint_layers",
    "min_probe_probability_margin",
    "rank_alignment",
]
FEATURE_SHORT = [
    "min_ext_margin",
    "max_distractor",
    "correct_conf",
    "earliest_kint1",
    "mean_kint",
    "std_kint",
    "min_int_margin",
    "rank_align",
]

CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

MODELS = {
    "Ridge":   Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Lasso":   Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.01, max_iter=5000))]),
    "ElasticNet": Pipeline([("sc", StandardScaler()),
                            ("m", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))]),
    "RF":      Pipeline([("sc", StandardScaler()),
                         ("m", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=4))]),
    "HGB":     Pipeline([("sc", StandardScaler()),
                         ("m", HistGradientBoostingRegressor(max_iter=200, random_state=42))]),
}

TARGETS = {
    "k_int_traj_auc":  "K_int trajectory (full-layer, ck1-ck8)",
    "k_int_ck8":       "K_int @ ck8 (full-layer)",
    "ext_drop":        "External drop (K_ext_base - K_ext_ck8)",
    "int_ext_gap_ck8": "K_int - K_ext gap @ ck8",
}


def load_targets():
    traj = pd.read_csv(TRAJ_BY_Q)
    full = traj[traj["layer_band"] == "full"][["method", "question_idx", "auc", "ck8"]].rename(
        columns={"auc": "k_int_traj_auc", "ck8": "k_int_ck8"})

    correct_idx = load_correct_idx()
    te, orig_to_pos = load_split_indices()
    df = load_scores_df()
    parquet_full = get_parquet_full_single(df)
    filt_mask = compute_prefilter_mask(te, orig_to_pos, df)
    filt_te = te[filt_mask]
    filt_o2p = {int(qi): i for i, qi in enumerate(filt_te)}

    base_ext = parquet_full[parquet_full["model_id"] == "base"][["question_idx", "k_external"]].rename(
        columns={"k_external": "k_ext_base"})

    ext_rows = []
    for method in METHODS:
        labels, _, _ = get_method_ck8_labels(
            method, correct_idx, filt_te, filt_o2p, parquet_full)
        if labels is None:
            continue
        fname = method_fname(method)
        ext_path = EXT_DIR / f"{fname}_ck8_bio_ext.npy"
        if not ext_path.exists():
            continue
        mk_ext = compute_k_ext(np.load(ext_path), correct_idx, filt_te)
        for qi, kext in zip(filt_te, mk_ext):
            ext_rows.append({"method": method, "question_idx": int(qi), "k_ext_ck8": float(kext)})

    ext = pd.DataFrame(ext_rows).merge(base_ext, on="question_idx", how="left")
    ext["ext_drop"] = ext["k_ext_base"] - ext["k_ext_ck8"]

    targets = full.merge(ext[["method", "question_idx", "ext_drop", "k_ext_ck8"]],
                         on=["method", "question_idx"], how="left")
    targets["int_ext_gap_ck8"] = targets["k_int_ck8"] - targets["k_ext_ck8"]
    return targets


def build_dataset():
    feats = pd.read_csv(POOLED_FEATURES)
    targs = load_targets()
    data  = feats.merge(targs, on=["method", "question_idx"], how="inner")
    data  = data.dropna(subset=FEATURE_COLS + list(TARGETS.keys()))
    return data


def run_cv_regression(X, y, model_name, model):
    from sklearn.model_selection import cross_val_score, KFold
    r2_scores = cross_val_score(model, X, y, cv=CV, scoring="r2", n_jobs=1)
    y_pred    = cross_val_predict(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
    rho, _    = spearmanr(y, y_pred)
    return {
        "r2_mean":  float(r2_scores.mean()),
        "r2_std":   float(r2_scores.std()),
        "spearman": float(rho),
        "y_pred":   y_pred,
    }


def get_feature_importances(X, y, model_name, model):
    model.fit(X, y)
    m = model.named_steps["m"]
    if hasattr(m, "coef_"):
        sc   = model.named_steps["sc"]
        coef = m.coef_ * sc.scale_
        return np.abs(coef)
    elif hasattr(m, "feature_importances_"):
        return m.feature_importances_
    return np.zeros(len(FEATURE_COLS))


def plot_pred_vs_actual(y_true, y_pred_dict, target_key, target_label):
    best_model = max(y_pred_dict, key=lambda k: spearmanr(y_true, y_pred_dict[k]).statistic)
    y_pred = y_pred_dict[best_model]
    rho, _ = spearmanr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, alpha=0.2, s=8, color="#1f77b4", rasterized=True)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel(f"True: {target_label}", fontsize=9)
    ax.set_ylabel("Predicted", fontsize=9)
    ax.set_title(f"{best_model}  |  Spearman rho = {rho:.3f}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    stem = OUT_DIR / f"predicted_vs_actual_{target_key}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_feature_importance(importances_dict, target_key, target_label):
    arr = np.array(list(importances_dict.values()))
    normed = arr / (arr.sum(axis=1, keepdims=True) + 1e-12)
    mean_imp = normed.mean(axis=0)
    order = np.argsort(mean_imp)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = ["#d62728" if mean_imp[i] == mean_imp.max() else "#1f77b4" for i in order]
    ax.barh(np.arange(len(order)), mean_imp[order], color=colors, alpha=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([FEATURE_SHORT[i] for i in order], fontsize=9)
    ax.set_title(f"Feature importance: {target_label}", fontsize=9)
    ax.set_xlabel("Mean normalized importance", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    stem = OUT_DIR / f"feature_importance_{target_key}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def main():
    print("Building dataset...")
    data = build_dataset()
    X = data[FEATURE_COLS].to_numpy(dtype=float)
    print(f"  {len(data)} rows, {len(FEATURE_COLS)} features, {len(TARGETS)} targets")

    result_rows = []
    for t_key, t_label in TARGETS.items():
        print(f"\nTarget: {t_key}")
        y = data[t_key].to_numpy(dtype=float)
        y_preds = {}
        imps = {}
        for mname, model in MODELS.items():
            res = run_cv_regression(X, y, mname, model)
            print(f"  {mname:12s}  R2={res['r2_mean']:.3f}+-{res['r2_std']:.3f}  "
                  f"rho={res['spearman']:.3f}")
            result_rows.append({
                "target": t_key, "model": mname,
                "r2_mean": round(res["r2_mean"], 4),
                "r2_std":  round(res["r2_std"],  4),
                "spearman_rho": round(res["spearman"], 4),
            })
            y_preds[mname] = res["y_pred"]
            imps[mname] = get_feature_importances(X, y, mname, model)

        plot_pred_vs_actual(y, y_preds, t_key, t_label)
        plot_feature_importance(imps, t_key, t_label)

    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_DIR / "trajectory_regression_results.csv", index=False)

    lines = [
        "# Trajectory Regression Results",
        "",
        "Base features (8) predicting continuous trajectory targets (full-layer probe).",
        "CV: RepeatedKFold(n_splits=5, n_repeats=10).",
        "",
        "| Target | Model | R2 mean | R2 std | Spearman rho |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in results.iterrows():
        lines.append(f"| {r['target']} | {r['model']} | {r['r2_mean']:.4f} | "
                     f"{r['r2_std']:.4f} | {r['spearman_rho']:.4f} |")
    (OUT_DIR / "trajectory_regression_results.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(f"\nSaved all outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
