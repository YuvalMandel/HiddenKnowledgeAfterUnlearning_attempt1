#!/usr/bin/env python3
"""
Task B: Feature collinearity diagnosis for the 8 base-model features.

Produces:
  1. Pearson correlation heatmap of the 8 features (pooled across methods)
  2. VIF (variance inflation factor) table
  3. Per-method LR coefficient heatmap (suppressed_vs_forgotten task)

Input:  plots/base_feature_prediction/pooled_features.csv
        plots/base_feature_prediction/feature_coefficients.csv
Output: plots/task_b_collinearity/
  correlation_heatmap.pdf/png
  vif_table.csv/md
  coefficient_heatmap_suppressed_vs_forgotten.pdf/png
  coefficient_heatmap_suppressed_vs_retained.pdf/png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

REPO    = Path(__file__).resolve().parent.parent
IN_CSV  = REPO / "plots" / "base_feature_prediction" / "pooled_features.csv"
COE_CSV = REPO / "plots" / "base_feature_prediction" / "feature_coefficients.csv"
OUT_DIR = REPO / "plots" / "task_b_collinearity"
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

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]


# ── VIF ───────────────────────────────────────────────────────────────────────

def compute_vif(X: np.ndarray, feature_names: list) -> pd.DataFrame:
    """VIF_i = 1 / (1 - R²_i), where R²_i is from regressing feature i on all others."""
    rows = []
    for i, name in enumerate(feature_names):
        y  = X[:, i]
        Xo = np.delete(X, i, axis=1)
        r2 = LinearRegression().fit(Xo, y).score(Xo, y)
        r2 = min(r2, 1 - 1e-10)  # avoid division by zero
        vif = 1.0 / (1.0 - r2)
        rows.append({"feature": name, "R2": round(r2, 4), "VIF": round(vif, 2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_corr_heatmap(corr: np.ndarray, labels: list, out_stem: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Feature Pearson Correlation (pooled, n per method × 307 questions)", fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr[i, j]
            color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_stem}")


def plot_coef_heatmap(coef_df: pd.DataFrame, task: str, out_stem: Path):
    # pivot: rows = methods, cols = features; value = coefficient
    # For binary tasks the positive class is 'suppressed'
    sub = coef_df[
        (coef_df["task"] == task) &
        (coef_df["class_or_positive"].isin(["1", "suppressed", 1]))
    ].copy()
    if sub.empty:
        # fallback: take whichever class row is available
        sub = coef_df[coef_df["task"] == task].copy()

    pivot = sub.pivot_table(index="method", columns="feature", values="coef", aggfunc="mean")
    # reorder columns and rows
    feat_order = [f for f in FEATURE_COLS if f in pivot.columns]
    meth_order = [m for m in METHODS if m in pivot.index]
    pivot = pivot.reindex(index=meth_order, columns=feat_order)

    short_cols = [FEATURE_SHORT[FEATURE_COLS.index(f)] for f in feat_order]
    vmax = float(np.nanmax(np.abs(pivot.to_numpy())))

    fig, ax = plt.subplots(figsize=(9, 4))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(pivot.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(short_cols)))
    ax.set_xticklabels(short_cols, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(meth_order)))
    ax.set_yticklabels(meth_order, fontsize=9)
    ax.set_title(f"LR coefficients — {task.replace('_', ' ')} (standardized features)", fontsize=10)
    for i in range(len(meth_order)):
        for j in range(len(feat_order)):
            v = pivot.iloc[i, j]
            if np.isfinite(v):
                color = "white" if abs(v) > 0.6 * vmax else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Coefficient")
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_stem}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    data = pd.read_csv(IN_CSV).dropna(subset=FEATURE_COLS)
    X = data[FEATURE_COLS].to_numpy(dtype=float)
    valid = np.isfinite(X).all(axis=1)
    X = X[valid]
    print(f"Loaded {len(X)} rows (after dropping NaN).")

    # ── 1. Correlation heatmap ─────────────────────────────────────────────
    corr = np.corrcoef(X, rowvar=False)
    plot_corr_heatmap(corr, FEATURE_SHORT,
                      OUT_DIR / "correlation_heatmap")

    # Save raw correlation matrix
    corr_df = pd.DataFrame(corr, index=FEATURE_COLS, columns=FEATURE_COLS)
    corr_df.to_csv(OUT_DIR / "correlation_matrix.csv")

    # ── 2. VIF table ───────────────────────────────────────────────────────
    vif_df = compute_vif(X, FEATURE_COLS)
    vif_df.to_csv(OUT_DIR / "vif_table.csv", index=False)

    lines = [
        "# Task B: VIF Table",
        "",
        "Variance Inflation Factor for each of the 8 base-model features.",
        "VIF > 5 indicates moderate collinearity; VIF > 10 is severe.",
        "",
        "| Feature | R² (regressed on others) | VIF |",
        "|:--------|-------------------------:|----:|",
    ]
    for _, r in vif_df.iterrows():
        flag = " ⚠" if r["VIF"] > 5 else ""
        lines.append(f"| {r['feature']} | {r['R2']:.4f} | {r['VIF']:.2f}{flag} |")
    (OUT_DIR / "vif_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nVIF Table:")
    print(vif_df.to_string(index=False))

    # ── 3. Coefficient heatmaps ────────────────────────────────────────────
    if not COE_CSV.exists():
        print(f"\nCoefficients file not found: {COE_CSV}. Skipping heatmaps.")
    else:
        coef_df = pd.read_csv(COE_CSV)
        for task in ["suppressed_vs_forgotten", "suppressed_vs_retained"]:
            if task not in coef_df["task"].values:
                print(f"Task '{task}' not found in coefficients CSV. Skipping.")
                continue
            plot_coef_heatmap(
                coef_df, task,
                OUT_DIR / f"coefficient_heatmap_{task}",
            )

    # ── 4. Pairwise correlation summary (flag high-correlation pairs) ──────
    pairs = []
    n = len(FEATURE_COLS)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "feature_a": FEATURE_COLS[i],
                "feature_b": FEATURE_COLS[j],
                "pearson_r": round(float(corr[i, j]), 4),
                "abs_r":     round(abs(float(corr[i, j])), 4),
            })
    pairs_df = pd.DataFrame(pairs).sort_values("abs_r", ascending=False)
    pairs_df.to_csv(OUT_DIR / "pairwise_correlations.csv", index=False)

    print("\nTop 10 highest-correlation pairs:")
    print(pairs_df.head(10).to_string(index=False))

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
