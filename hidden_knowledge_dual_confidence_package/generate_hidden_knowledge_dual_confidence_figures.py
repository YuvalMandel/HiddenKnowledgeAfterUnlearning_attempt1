#!/usr/bin/env python3
"""
Generate hidden-knowledge figures under two confidence systems:

System A: raw_base_threshold
    - confidence score: abs(raw_margin)
    - binary confident split: Base median abs(raw_margin)
    - low/mid/high buckets: Base terciles of abs(raw_margin)

System B: percentile_to_base
    - confidence score: percentile rank of abs(raw_margin) within each method
    - binary confident split: percentile >= 0.5
    - low/mid/high buckets: percentile terciles
    - also computes Base-mapped confidence values for reference

Why two systems?
    raw_base_threshold preserves absolute margin effects relative to the untouched Base model.
    percentile_to_base preserves each method's internal ranking / sample distribution while
    still providing common thresholds across methods.

Figures produced for EACH system
--------------------------------
Figure 1
    Quadrant prevalence with gold-label decomposition.
    Quadrants are defined using binary confidence x surfaced correctness.

Figure 2b (two versions)
    Probe accuracy by quadrant:
      - early_rf
      - mid_linear

Figure 3b (two versions)
    Apples-to-apples comparison on confidence buckets (low/mid/high), not quadrants:
      x = LLM accuracy vs gold within bucket
      y = probe accuracy vs gold within bucket
      - early_rf version
      - mid_linear version

Figure 4c (two versions)
    Hidden-knowledge gap by confidence bucket:
      gap = (probe_acc - Base_probe_acc) - (llm_acc - Base_llm_acc)
      - early_rf version
      - mid_linear version

Inputs expected
---------------
The script auto-discovers all *_test_predictions.csv files in the input directory.
Expected files (one per method, Base + 8 unlearning methods):
    Base_test_predictions.csv
    GradDiff_test_predictions.csv
    RMU_test_predictions.csv
    RMU-LAT_test_predictions.csv
    RepNoise_test_predictions.csv
    ELM_test_predictions.csv
    RR_test_predictions.csv
    TAR_test_predictions.csv
    PB_J_test_predictions.csv

Required columns:
    method_name
    gold_label
    raw_margin
    abs_raw_margin
    early_rf_label_prob
    mid_linear_label_score

Outputs
-------
For each system:
    quadrant_metrics.csv
    bucket_metrics.csv
    sample_assignments.csv
    quadrant_auc_feasibility_diagnostic.csv
    figure1_quadrant_prevalence_gold_decomposition.png
    figure2b_probe_accuracy_by_quadrant_early_rf.png
    figure2b_probe_accuracy_by_quadrant_mid_linear.png
    figure3b_llm_accuracy_vs_probe_accuracy_early_rf.png
    figure3b_llm_accuracy_vs_probe_accuracy_mid_linear.png
    figure4c_hidden_knowledge_gap_early_rf.png
    figure4c_hidden_knowledge_gap_mid_linear.png
    figure4c_gap_metrics_early_rf.csv
    figure4c_gap_metrics_mid_linear.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ALL_METHODS = ["Base", "GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]
QUAD_ORDER = ["Q1 confident-correct", "Q2 unconfident-correct", "Q3 confident-wrong", "Q4 unconfident-wrong"]
BUCKET_ORDER = ["low", "mid", "high"]
METHOD_MARKERS = {
    "Base": "o", "GradDiff": "s", "RMU": "^", "RMU-LAT": "D",
    "RepNoise": "P", "ELM": "X", "RR": "*", "TAR": "h", "PB&J": "v",
}
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
QUAD_COLORS = {"Q1 confident-correct":"#2ca02c","Q2 unconfident-correct":"#ff7f0e","Q3 confident-wrong":"#d62728","Q4 unconfident-wrong":"#1f77b4"}
BUCKET_COLORS = {"low":"#1f77b4","mid":"#ff7f0e","high":"#2ca02c"}


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    """Return methods present in df in display order."""
    present = set(df["method"].unique())
    return [m for m in ALL_METHODS if m in present]


def discover_prediction_files(input_dir: Path) -> list[Path]:
    """Discover all *_test_predictions.csv files in input_dir."""
    files = sorted(input_dir.glob("*_test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No *_test_predictions.csv files found in {input_dir}")
    return files

def load_predictions(input_dir: Path) -> pd.DataFrame:
    frames = []
    for path in discover_prediction_files(input_dir):
        df = pd.read_csv(path)
        required = {"method_name","gold_label","raw_margin","abs_raw_margin","early_rf_label_prob","mid_linear_label_score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        df = df.copy()
        method = str(df["method_name"].iloc[0])
        method = "Base" if method.lower() == "base" else method
        df["method"] = method
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["gold_label"] = out["gold_label"].astype(int)
    out["raw_margin"] = out["raw_margin"].astype(float)
    out["abs_raw_margin"] = out["abs_raw_margin"].astype(float)
    out["surface_pred"] = (out["raw_margin"] > 0).astype(int)
    out["surface_correct"] = (out["surface_pred"] == out["gold_label"]).astype(int)
    out["probe_pred_early_rf"] = (out["early_rf_label_prob"].astype(float) >= 0.5).astype(int)
    out["probe_correct_early_rf"] = (out["probe_pred_early_rf"] == out["gold_label"]).astype(int)
    out["probe_pred_mid_linear"] = (out["mid_linear_label_score"].astype(float) >= 0.0).astype(int)
    out["probe_correct_mid_linear"] = (out["probe_pred_mid_linear"] == out["gold_label"]).astype(int)
    out["sample_id"] = np.arange(len(out))
    return out

def ecdf_percentiles(vals: pd.Series) -> pd.Series:
    return vals.rank(method="average", pct=True).astype(float)

def assign_system(df: pd.DataFrame, system_name: str) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    if system_name == "raw_base_threshold":
        base_abs = d.loc[d["method"]=="Base","abs_raw_margin"]
        threshold = float(base_abs.median())
        q1, q2 = base_abs.quantile([1/3, 2/3]).tolist()
        d["confidence_score"] = d["abs_raw_margin"]
        d["confidence_score_display"] = d["confidence_score"]
        d["is_confident"] = (d["confidence_score"] >= threshold).astype(int)
        def bucket(v: float) -> str:
            if v < q1: return "low"
            if v < q2: return "mid"
            return "high"
        d["confidence_bucket"] = d["confidence_score"].map(bucket)
        thresholds = {"binary_base_median":threshold, "bucket_base_tercile_1":float(q1), "bucket_base_tercile_2":float(q2)}
    elif system_name == "percentile_to_base":
        d["confidence_percentile_within_method"] = d.groupby("method")["abs_raw_margin"].transform(ecdf_percentiles)
        base_abs_sorted = np.sort(d.loc[d["method"]=="Base","abs_raw_margin"].to_numpy())
        u = d["confidence_percentile_within_method"].clip(0,1).to_numpy()
        d["confidence_score"] = d["confidence_percentile_within_method"]
        d["confidence_score_display"] = np.quantile(base_abs_sorted, u, method="linear")
        d["is_confident"] = (d["confidence_percentile_within_method"] >= 0.5).astype(int)
        d["confidence_bucket"] = pd.cut(
            d["confidence_percentile_within_method"],
            bins=[0, 1/3, 2/3, 1.0000001],
            labels=["low","mid","high"],
            include_lowest=True,
            right=False,
        ).astype(str)
        thresholds = {"binary_percentile_threshold":0.5, "bucket_percentile_1":1/3, "bucket_percentile_2":2/3}
    else:
        raise ValueError(system_name)
    d["quadrant"] = np.select(
        [
            (d["is_confident"]==1) & (d["surface_correct"]==1),
            (d["is_confident"]==0) & (d["surface_correct"]==1),
            (d["is_confident"]==1) & (d["surface_correct"]==0),
        ],
        ["Q1 confident-correct","Q2 unconfident-correct","Q3 confident-wrong"],
        default="Q4 unconfident-wrong",
    )
    return d, thresholds

def compute_quadrant_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in _ordered_methods(df):
        dm = df[df["method"]==method]
        n_total = len(dm)
        for q in QUAD_ORDER:
            g = dm[dm["quadrant"]==q]
            rows.append({
                "method":method,
                "quadrant":q,
                "n":len(g),
                "fraction_of_samples":len(g)/n_total if n_total else np.nan,
                "gold_false_count":int((g["gold_label"]==0).sum()),
                "gold_true_count":int((g["gold_label"]==1).sum()),
                "gold_false_fraction_of_total":((g["gold_label"]==0).sum()/n_total if n_total else np.nan),
                "gold_true_fraction_of_total":((g["gold_label"]==1).sum()/n_total if n_total else np.nan),
                "early_rf_probe_accuracy":g["probe_correct_early_rf"].mean() if len(g) else np.nan,
                "mid_linear_probe_accuracy":g["probe_correct_mid_linear"].mean() if len(g) else np.nan,
            })
    return pd.DataFrame(rows)

def compute_bucket_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in _ordered_methods(df):
        dm = df[df["method"]==method]
        for b in BUCKET_ORDER:
            g = dm[dm["confidence_bucket"]==b]
            rows.append({
                "method":method,
                "confidence_bucket":b,
                "n":len(g),
                "fraction_of_samples":len(g)/len(dm) if len(dm) else np.nan,
                "llm_accuracy":g["surface_correct"].mean() if len(g) else np.nan,
                "early_rf_probe_accuracy":g["probe_correct_early_rf"].mean() if len(g) else np.nan,
                "mid_linear_probe_accuracy":g["probe_correct_mid_linear"].mean() if len(g) else np.nan,
                "mean_abs_margin":g["abs_raw_margin"].mean() if len(g) else np.nan,
                "mean_confidence_score":g["confidence_score"].mean() if len(g) else np.nan,
                "mean_confidence_score_display":g["confidence_score_display"].mean() if len(g) else np.nan,
            })
    return pd.DataFrame(rows)

def plot_fig1(quadrant_metrics: pd.DataFrame, out_path: Path, title: str) -> None:
    methods = _ordered_methods(quadrant_metrics)
    n = len(methods)
    x = np.arange(len(QUAD_ORDER))
    width = min(0.18, 0.8 / n)
    offsets = np.linspace(-(n-1)/2 * width, (n-1)/2 * width, n)
    figw = max(14, 2 * n + 6)
    fig, ax = plt.subplots(figsize=(figw, 7))
    for offset, method in zip(offsets, methods):
        sub = quadrant_metrics[quadrant_metrics["method"]==method].set_index("quadrant").loc[QUAD_ORDER]
        gf = sub["gold_false_fraction_of_total"].to_numpy()
        gt = sub["gold_true_fraction_of_total"].to_numpy()
        c = METHOD_COLORS.get(method, "#888888")
        ax.bar(x+offset, gf, width=width, color=c, alpha=0.35)
        ax.bar(x+offset, gt, width=width, bottom=gf, color=c, alpha=0.85)
        totals = gf + gt
        for xi, yi in zip(x+offset, totals):
            ax.text(xi, yi + 0.006, f"{yi:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(QUAD_ORDER)
    ax.set_ylabel("Fraction of all samples")
    ax.set_xlabel("Quadrant")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    method_handles = [Patch(facecolor=METHOD_COLORS.get(m, "#888888"), edgecolor="none", alpha=0.85, label=m) for m in methods]
    segment_handles = [Patch(facecolor="gray", alpha=0.35, label="gold = F segment"),
                       Patch(facecolor="gray", alpha=0.85, label="gold = T segment")]
    leg1 = ax.legend(handles=method_handles, title="Method", loc="upper left", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=segment_handles, title="Stack meaning", loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def plot_fig2(quadrant_metrics: pd.DataFrame, out_path: Path, probe_col: str, title: str) -> None:
    methods = _ordered_methods(quadrant_metrics)
    x = np.arange(len(QUAD_ORDER))
    fig, ax = plt.subplots(figsize=(12,7))
    for method in methods:
        sub = quadrant_metrics[quadrant_metrics["method"]==method].set_index("quadrant").loc[QUAD_ORDER]
        ax.plot(x, sub[probe_col].to_numpy(),
                marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, "#888888"),
                linewidth=2, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(QUAD_ORDER)
    ax.set_ylim(0,1)
    ax.set_ylabel("Probe accuracy vs gold")
    ax.set_xlabel("Quadrant")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def plot_fig3(bucket_metrics: pd.DataFrame, out_path: Path, probe_col: str, title: str) -> None:
    methods = _ordered_methods(bucket_metrics)
    fig, ax = plt.subplots(figsize=(12,7))
    for method in methods:
        sub = bucket_metrics[bucket_metrics["method"]==method]
        for _, row in sub.iterrows():
            bucket = row["confidence_bucket"]
            ax.scatter(
                row["llm_accuracy"],
                row[probe_col],
                marker=METHOD_MARKERS.get(method, "o"),
                s=150,
                color=BUCKET_COLORS[bucket],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.9,
            )
            ax.text(row["llm_accuracy"]+0.005, row[probe_col]+0.005, f"{method}-{bucket}", fontsize=8)
    method_handles = [Line2D([0],[0], marker=METHOD_MARKERS.get(m,"o"), linestyle="", markerfacecolor="black", markeredgecolor="black", markersize=10, label=m) for m in methods]
    bucket_handles = [Line2D([0],[0], marker="o", linestyle="", markerfacecolor=BUCKET_COLORS[b], markeredgecolor="black", markersize=10, label=b) for b in BUCKET_ORDER]
    leg1 = ax.legend(handles=method_handles, title="Method", loc="lower right", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=bucket_handles, title="Confidence bucket", loc="upper left", frameon=True)
    ax.axline((0,0), (1,1), color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(-0.02,1.02)
    ax.set_ylim(-0.02,1.02)
    ax.set_xlabel("LLM accuracy vs gold (same bucket)")
    ax.set_ylabel("Probe accuracy vs gold (same bucket)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def plot_fig4(bucket_metrics: pd.DataFrame, out_path: Path, probe_col: str, title: str) -> pd.DataFrame:
    methods = _ordered_methods(bucket_metrics)
    n = len(methods)
    gap = bucket_metrics.copy()
    base = gap[gap["method"]=="Base"][["confidence_bucket","llm_accuracy",probe_col]].rename(
        columns={"llm_accuracy":"base_llm_accuracy", probe_col:"base_probe_accuracy"}
    )
    gap = gap.merge(base, on="confidence_bucket", how="left")
    gap["delta_llm_accuracy"] = gap["llm_accuracy"] - gap["base_llm_accuracy"]
    gap["delta_probe_accuracy"] = gap[probe_col] - gap["base_probe_accuracy"]
    gap["hidden_knowledge_gap"] = gap["delta_probe_accuracy"] - gap["delta_llm_accuracy"]

    x = np.arange(len(BUCKET_ORDER))
    width = min(0.18, 0.8 / n)
    offsets = np.linspace(-(n-1)/2 * width, (n-1)/2 * width, n)
    figw = max(12, 2 * n + 4)
    fig, ax = plt.subplots(figsize=(figw, 7))
    for offset, method in zip(offsets, methods):
        sub = gap[gap["method"]==method].set_index("confidence_bucket").loc[BUCKET_ORDER]
        vals = sub["hidden_knowledge_gap"].to_numpy()
        ax.bar(x+offset, vals, width=width, color=METHOD_COLORS.get(method, "#888888"), label=method)
        for xi, yi in zip(x+offset, vals):
            ax.text(xi, yi + (0.008 if yi>=0 else -0.008), f"{yi:.2f}",
                    ha="center", va="bottom" if yi>=0 else "top", fontsize=7, rotation=90)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_ORDER)
    ax.set_xlabel("Confidence bucket")
    ax.set_ylabel("(Δ probe accuracy) - (Δ LLM accuracy)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return gap

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hidden-knowledge figures for raw and percentile confidence systems.")
    parser.add_argument("--input-dir", required=True, help="Directory containing the four *_test_predictions.csv files.")
    parser.add_argument("--output-dir", required=True, help="Directory where figures/tables will be written.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_predictions(input_dir)

    systems = {
        "raw_base_threshold": assign_system(raw, "raw_base_threshold"),
        "percentile_to_base": assign_system(raw, "percentile_to_base"),
    }

    summary_lines = []
    for system_name, (df_sys, thresholds) in systems.items():
        sdir = output_dir / system_name
        sdir.mkdir(exist_ok=True)

        quadrant_metrics = compute_quadrant_metrics(df_sys)
        bucket_metrics = compute_bucket_metrics(df_sys)

        quadrant_metrics.to_csv(sdir / "quadrant_metrics.csv", index=False)
        bucket_metrics.to_csv(sdir / "bucket_metrics.csv", index=False)
        df_sys.to_csv(sdir / "sample_assignments.csv", index=False)

        plot_fig1(quadrant_metrics, sdir / "figure1_quadrant_prevalence_gold_decomposition.png",
                  f"Figure 1 — Quadrant prevalence with gold decomposition ({system_name})")
        plot_fig2(quadrant_metrics, sdir / "figure2b_probe_accuracy_by_quadrant_early_rf.png",
                  "early_rf_probe_accuracy",
                  f"Figure 2b — early_rf probe accuracy by quadrant ({system_name})")
        plot_fig2(quadrant_metrics, sdir / "figure2b_probe_accuracy_by_quadrant_mid_linear.png",
                  "mid_linear_probe_accuracy",
                  f"Figure 2b — mid_linear probe accuracy by quadrant ({system_name})")
        plot_fig3(bucket_metrics, sdir / "figure3b_llm_accuracy_vs_probe_accuracy_early_rf.png",
                  "early_rf_probe_accuracy",
                  f"Figure 3b — LLM accuracy vs early_rf probe accuracy ({system_name})")
        plot_fig3(bucket_metrics, sdir / "figure3b_llm_accuracy_vs_probe_accuracy_mid_linear.png",
                  "mid_linear_probe_accuracy",
                  f"Figure 3b — LLM accuracy vs mid_linear probe accuracy ({system_name})")
        gap_early = plot_fig4(bucket_metrics, sdir / "figure4c_hidden_knowledge_gap_early_rf.png",
                              "early_rf_probe_accuracy",
                              f"Figure 4c — hidden-knowledge gap using early_rf accuracy ({system_name})")
        gap_mid = plot_fig4(bucket_metrics, sdir / "figure4c_hidden_knowledge_gap_mid_linear.png",
                            "mid_linear_probe_accuracy",
                            f"Figure 4c — hidden-knowledge gap using mid_linear accuracy ({system_name})")
        gap_early.to_csv(sdir / "figure4c_gap_metrics_early_rf.csv", index=False)
        gap_mid.to_csv(sdir / "figure4c_gap_metrics_mid_linear.csv", index=False)

        auc_diag = []
        for method in _ordered_methods(df_sys):
            dm = df_sys[df_sys["method"]==method]
            for q in QUAD_ORDER:
                g = dm[dm["quadrant"]==q]
                auc_diag.append({
                    "method":method,
                    "quadrant":q,
                    "n":len(g),
                    "n_gold0":int((g["gold_label"]==0).sum()),
                    "n_gold1":int((g["gold_label"]==1).sum()),
                    "auc_defined":int(g["gold_label"].nunique()==2),
                })
        pd.DataFrame(auc_diag).to_csv(sdir / "quadrant_auc_feasibility_diagnostic.csv", index=False)

        summary_lines.append(f"{system_name}: {thresholds}")

    (output_dir / "README.txt").write_text(
        "Hidden-knowledge figure package with two confidence systems.\n\n"
        + "\n".join(summary_lines)
        + "\n\nSee the top-level Python script docstring for the full methodological details.\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
