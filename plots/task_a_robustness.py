#!/usr/bin/env python3
"""
Task A: Robustness validation for base-feature classifiers.

Sweeps 20 random seeds × 9 C values for the two binary tasks:
  - suppressed_vs_forgotten  (suppressed=1, forgotten=0)
  - suppressed_vs_retained   (suppressed=1, retained=0)

Input:  plots/base_feature_prediction/pooled_features.csv
Output: plots/task_a_robustness/
  robustness_results.csv          — full seed×C grid
  robustness_summary.csv/md       — mean ± std per C, per task
  robustness_violin_{task}.pdf/png — AUC distribution across seeds per C
  robustness_seed_stability.pdf/png — per-seed mean AUC (C=1.0 default)
"""
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

REPO = Path(__file__).resolve().parent.parent
IN_CSV  = REPO / "plots" / "base_feature_prediction" / "pooled_features.csv"
OUT_DIR = REPO / "plots" / "task_a_robustness"
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

BINARY_TASKS = {
    "suppressed_vs_forgotten": {"suppressed": 1, "forgotten": 0},
    "suppressed_vs_retained":  {"suppressed": 1, "retained":  0},
}

SEEDS  = list(range(20))
C_VALS = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
N_SPLITS = 5


def make_pipe(C, seed):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(
            C=C, solver="lbfgs", class_weight="balanced",
            max_iter=5000, random_state=seed,
        )),
    ])


def run_one(X, y, C, seed):
    n_min = int(np.bincount(y).min())
    n_splits = max(2, min(N_SPLITS, n_min))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipe = make_pipe(C, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, y_prob))


def main():
    data = pd.read_csv(IN_CSV)
    data = data.dropna(subset=FEATURE_COLS)

    rows = []
    for task_name, mapping in BINARY_TASKS.items():
        sub = data[data["subset"].isin(mapping.keys())].copy()
        sub["y"] = sub["subset"].map(mapping).astype(int)
        X = sub[FEATURE_COLS].to_numpy(dtype=float)
        y = sub["y"].to_numpy()
        n_pos  = int(y.sum())
        n_neg  = int((1 - y).sum())
        n_total = len(y)
        print(f"\nTask: {task_name}  n={n_total} (pos={n_pos}, neg={n_neg})")

        for seed in SEEDS:
            for C in C_VALS:
                auc = run_one(X, y, C, seed)
                rows.append({"task": task_name, "seed": seed, "C": C, "roc_auc": auc})
                print(f"  seed={seed:2d}  C={C:7.3f}  AUC={auc:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "robustness_results.csv", index=False)

    # ── summary: mean ± std per task × C ─────────────────────────────────────
    summary = (
        results.groupby(["task", "C"])["roc_auc"]
        .agg(mean="mean", std="std", min="min", max="max")
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "robustness_summary.csv", index=False)

    lines = [
        "# Task A: Robustness Summary",
        "",
        "ROC-AUC across 20 seeds × 9 C values.  StratifiedKFold(5).",
        "",
    ]
    for task_name in BINARY_TASKS:
        lines += [f"## {task_name}", ""]
        lines += ["| C | Mean AUC | Std | Min | Max |",
                  "|---|---:|---:|---:|---:|"]
        sub = summary[summary["task"] == task_name]
        for _, r in sub.iterrows():
            lines.append(f"| {r['C']:.3f} | {r['mean']:.4f} | {r['std']:.4f} | {r['min']:.4f} | {r['max']:.4f} |")
        lines.append("")
    (OUT_DIR / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # ── violin plot: AUC distribution per C, per task ────────────────────────
    for task_name in BINARY_TASKS:
        sub = results[results["task"] == task_name]
        groups = [sub[sub["C"] == c]["roc_auc"].to_numpy() for c in C_VALS]
        labels = [str(c) for c in C_VALS]

        fig, ax = plt.subplots(figsize=(9, 4))
        parts = ax.violinplot(groups, positions=range(len(C_VALS)),
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#1f77b4")
            pc.set_alpha(0.6)
        # overlay individual seed points
        for i, g in enumerate(groups):
            ax.scatter(np.full(len(g), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(g)),
                       g, s=12, color="#d62728", alpha=0.6, zorder=3)
        ax.set_xticks(range(len(C_VALS)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("C (regularization)", fontsize=10)
        ax.set_ylabel("ROC-AUC (20 seeds)", fontsize=10)
        ax.set_title(f"Task A — {task_name.replace('_', ' ')}", fontsize=11)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        stem = OUT_DIR / f"robustness_violin_{task_name}"
        fig.savefig(f"{stem}.png", dpi=180, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved violin: {stem}")

    # ── seed stability: per-seed mean AUC at default C=1.0 ───────────────────
    default_c = 1.0
    seed_sub = results[results["C"] == default_c]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, task_name in zip(axes, BINARY_TASKS):
        ts = seed_sub[seed_sub["task"] == task_name].sort_values("seed")
        ax.bar(ts["seed"], ts["roc_auc"], color="#1f77b4", alpha=0.8)
        ax.axhline(ts["roc_auc"].mean(), color="#d62728", linestyle="--",
                   linewidth=1.2, label=f"mean={ts['roc_auc'].mean():.3f}")
        ax.set_xlabel("Seed", fontsize=9)
        ax.set_ylabel("ROC-AUC", fontsize=9)
        ax.set_title(task_name.replace("_", " ") + f"  (C={default_c})", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Task A — Seed Stability", fontsize=11)
    fig.tight_layout()
    stem = OUT_DIR / "robustness_seed_stability"
    fig.savefig(f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved seed stability: {stem}")

    print(f"\nAll outputs saved to {OUT_DIR}")
    print("\nSummary (C=1.0):")
    print(summary[summary["C"] == 1.0].to_string(index=False))


if __name__ == "__main__":
    main()
