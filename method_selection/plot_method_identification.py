#!/usr/bin/env python3
"""
plot_method_identification.py — Figure 1 equivalent from research_decision_summary.pdf.

Reads per-band summary CSVs produced by raw_hiddenstate_method_classification_safe.py
and produces a grouped bar chart: accuracy by band (feature space) and classifier.

The plot replicates the finding that method identification from raw hidden states is
nearly trivial, especially in late layers (RF ~0.98, logistic ~0.998).

Usage:
  python method_selection/plot_method_identification.py \
      --results_dir method_selection_out/method_id \
      --out_dir plots
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BANDS = ["early", "mid", "late"]
BAND_LABELS = {"early": "Early\n(layers 0–10)", "mid": "Mid\n(layers 11–21)", "late": "Late\n(layers 22–32)"}

MODEL_COLORS = {
    "logistic": "#2166ac",
    "rf":       "#d6604d",
    "nn":       "#4dac26",
}
MODEL_LABELS = {
    "logistic": "Logistic Regression",
    "rf":       "Random Forest",
    "nn":       "1-NN",
}
CHANCE = 1.0 / 8.0


def load_summaries(results_dir: Path) -> pd.DataFrame:
    frames = []
    for band in BANDS:
        band_dir = results_dir / band
        csv_path = band_dir / "method_classification_summary.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found — skipping band={band}")
            continue
        df = pd.read_csv(csv_path)
        df["band"] = band
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No method_classification_summary.csv found under {results_dir}/{{early,mid,late}}/. "
            "Run slurm_method_id.sh first."
        )
    return pd.concat(frames, ignore_index=True)


def make_bar_plot(df: pd.DataFrame, out_path: str) -> None:
    models = [m for m in ["logistic", "rf", "nn"] if m in df["model"].values]
    n_bands = len(BANDS)
    n_models = len(models)
    x = np.arange(n_bands)
    width = 0.72 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("band")
        heights, errs = [], []
        for band in BANDS:
            if band in sub.index:
                heights.append(float(sub.loc[band, "accuracy_mean"]))
                ci_lo = float(sub.loc[band, "accuracy_ci95_low"])
                ci_hi = float(sub.loc[band, "accuracy_ci95_high"])
                errs.append(float(sub.loc[band, "accuracy_mean"]) - ci_lo)
            else:
                heights.append(float("nan"))
                errs.append(0.0)

        bars = ax.bar(
            x + offsets[i], heights, width,
            yerr=errs, capsize=4,
            color=MODEL_COLORS.get(model, "#999999"),
            label=MODEL_LABELS.get(model, model),
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )

        for bar, h in zip(bars, heights):
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.axhline(CHANCE, color="black", linestyle="--", linewidth=1.2,
               label=f"Chance (1/8 = {CHANCE:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([BAND_LABELS.get(b, b) for b in BANDS], fontsize=11)
    ax.set_ylabel("Method Identification Accuracy", fontsize=12)
    ax.set_xlabel("Feature Space (Hidden-State Band)", fontsize=12)
    ax.set_title(
        "Method Identification from Raw Hidden States\n"
        "(train: 100 val rows/method, test: full test set)",
        fontsize=12,
    )
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot method identification accuracy by hidden-state band."
    )
    ap.add_argument("--results_dir", default="method_selection_out/method_id",
                    help="Root dir containing early/ mid/ late/ subdirs with summary CSVs")
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_summaries(Path(args.results_dir))

    print(f"\nLoaded {len(df)} rows across bands: {sorted(df['band'].unique())}")
    print(df[["band", "model", "accuracy_mean", "accuracy_ci95_low", "accuracy_ci95_high"]].to_string(index=False))

    out_path = os.path.join(args.out_dir, "method_identification_by_band.png")
    make_bar_plot(df, out_path)
    print(f"\nDone. Plot saved to {out_path}")


if __name__ == "__main__":
    main()
