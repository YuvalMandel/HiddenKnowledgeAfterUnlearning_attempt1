from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = ROOT / "out_threat_model_judge_v1" / "labels" / "wmdp_bio_threat_model_labels_inferred_v1_audited.parquet"
DEFAULT_PAIR = ROOT / "plots" / "pair_level_suppressed_forgotten" / "pair_level_dataset.csv"
DEFAULT_APPENDIX = ROOT / "LLM as as Judge" / "wmdp_bio_inferred_categories_first_pass.csv"
DEFAULT_OUTPUT = ROOT / "out_threat_model_judge_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--pair-dataset", default=str(DEFAULT_PAIR))
    parser.add_argument("--appendix-categories", default=str(DEFAULT_APPENDIX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def load_question_level(pair_path: Path) -> pd.DataFrame:
    pair = pd.read_csv(pair_path)
    subset = (
        pair.loc[pair["checkpoint"].eq("ck8"), ["method", "question_idx", "subset_ck8"]]
        .drop_duplicates()
        .copy()
    )
    subset = subset[subset["subset_ck8"].isin(["suppressed", "forgotten"])].copy()
    subset["suppressed"] = subset["subset_ck8"].eq("suppressed").astype(int)
    return subset


def load_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        labels = pd.read_parquet(path)
    else:
        labels = pd.read_csv(path)
    return labels.copy()


def merge_inputs(question_level: pd.DataFrame, labels: pd.DataFrame, appendix_path: Path) -> pd.DataFrame:
    label_columns = [
        "question_id",
        "question_hash",
        "risk_stage_analysis_final",
        "risk_stage_final",
        "risk_mechanism_final",
        "proxy_relation_final",
        "needs_review",
    ]
    merged = question_level.merge(
        labels[label_columns],
        left_on="question_idx",
        right_on="question_id",
        how="left",
    )
    if appendix_path.exists():
        appendix = pd.read_csv(appendix_path)[["question_id", "category_first_pass"]].copy()
        merged = merged.merge(appendix, on="question_id", how="left")
    return merged


def bootstrap_shift(df: pd.DataFrame, value_col: str, group_cols: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    grouped = [((), df)] if not group_cols else list(df.groupby(group_cols, dropna=False))
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        supp = group_df[group_df["subset_ck8"].eq("suppressed")].copy()
        forg = group_df[group_df["subset_ck8"].eq("forgotten")].copy()
        if supp.empty or forg.empty:
            continue
        n = min(len(supp), len(forg))
        labels = sorted(group_df[value_col].fillna("missing").unique())
        draws = {label: [] for label in labels}
        for _ in range(n_boot):
            sample_s = supp.sample(n=n, replace=True, random_state=int(rng.integers(0, 1_000_000_000)))
            sample_f = forg.sample(n=n, replace=True, random_state=int(rng.integers(0, 1_000_000_000)))
            for label in labels:
                prop_s = sample_s[value_col].fillna("missing").eq(label).mean()
                prop_f = sample_f[value_col].fillna("missing").eq(label).mean()
                draws[label].append(prop_s - prop_f)
        for label, values in draws.items():
            arr = np.asarray(values, dtype=float)
            record = {
                "label": label,
                "mean_shift_pp": float(arr.mean() * 100),
                "ci_low_pp": float(np.quantile(arr, 0.025) * 100),
                "ci_high_pp": float(np.quantile(arr, 0.975) * 100),
                "pr_shift_gt_0": float((arr > 0).mean()),
                "pr_shift_lt_0": float((arr < 0).mean()),
                "n_rows": int(len(group_df)),
                "n_unique_questions": int(group_df["question_idx"].nunique()),
            }
            for key_name, key_value in zip(group_cols, group_key):
                record[key_name] = key_value
            records.append(record)
    ordered = group_cols + ["label", "mean_shift_pp", "ci_low_pp", "ci_high_pp", "pr_shift_gt_0", "pr_shift_lt_0", "n_rows", "n_unique_questions"]
    return pd.DataFrame(records)[ordered] if records else pd.DataFrame(columns=ordered)


def intrinsic_effect(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if group_cols:
        group_iter = df.groupby(group_cols, dropna=False)
    else:
        group_iter = [((), df)]
    for group_key, group_df in group_iter:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base_rate = group_df["suppressed"].mean()
        per_label = group_df.groupby(value_col)["suppressed"].mean()
        counts = group_df.groupby(value_col).size()
        unique_questions = group_df.groupby(value_col)["question_idx"].nunique()
        for label, label_rate in per_label.items():
            record = {
                "label": label,
                "centered_suppression_effect_pp": float((label_rate - base_rate) * 100),
                "p_suppressed_given_label": float(label_rate),
                "p_suppressed_baseline": float(base_rate),
                "n_rows": int(counts[label]),
                "n_unique_questions": int(unique_questions[label]),
            }
            for key_name, key_value in zip(group_cols, group_key):
                record[key_name] = key_value
            records.append(record)
    ordered = group_cols + ["label", "centered_suppression_effect_pp", "p_suppressed_given_label", "p_suppressed_baseline", "n_rows", "n_unique_questions"]
    return pd.DataFrame(records)[ordered] if records else pd.DataFrame(columns=ordered)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= start) & (y_prob < end if end < 1 else y_prob <= end)
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += abs(acc - conf) * mask.mean()
    return float(ece)


def run_predictive_models(df: pd.DataFrame, output_csv: Path, figure_path: Path) -> pd.DataFrame:
    specs = [
        ("Model A", ["method"]),
        ("Model B", ["method", "category_first_pass"]),
        ("Model C", ["method", "risk_stage_analysis_final"]),
        ("Model D", ["method", "risk_mechanism_final"]),
        ("Model E", ["method", "proxy_relation_final"]),
        ("Model F", ["method", "category_first_pass", "risk_stage_analysis_final", "risk_mechanism_final", "proxy_relation_final"]),
    ]
    work = df.copy()
    work["category_first_pass"] = work["category_first_pass"].fillna("missing")
    work["risk_stage_analysis_final"] = work["risk_stage_analysis_final"].fillna("missing")
    work["risk_mechanism_final"] = work["risk_mechanism_final"].fillna("missing")
    work["proxy_relation_final"] = work["proxy_relation_final"].fillna("missing")
    y = work["suppressed"].to_numpy()
    groups = work["question_idx"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    rows = []
    base_metrics: dict[str, float] | None = None
    for model_name, features in specs:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    features,
                )
            ]
        )
        clf = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )

        oof_prob = np.zeros(len(work), dtype=float)
        for train_idx, test_idx in cv.split(work[features], y, groups):
            clf.fit(work.iloc[train_idx][features], y[train_idx])
            oof_prob[test_idx] = clf.predict_proba(work.iloc[test_idx][features])[:, 1]
        oof_pred = (oof_prob >= 0.5).astype(int)
        frac_pos, mean_pred = calibration_curve(y, oof_prob, n_bins=10, strategy="uniform")
        metrics = {
            "model": model_name,
            "features": " + ".join(features),
            "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
            "macro_f1": float(f1_score(y, oof_pred, average="macro")),
            "roc_auc": float(roc_auc_score(y, oof_prob)),
            "pr_auc": float(average_precision_score(y, oof_prob)),
            "log_loss": float(log_loss(y, oof_prob, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y, oof_prob)),
            "expected_calibration_error": expected_calibration_error(y, oof_prob),
            "calibration_bins_used": int(len(frac_pos)),
        }
        if base_metrics is None:
            base_metrics = metrics
        metrics["incremental_balanced_accuracy_vs_A"] = float(metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"])
        metrics["incremental_macro_f1_vs_A"] = float(metrics["macro_f1"] - base_metrics["macro_f1"])
        metrics["incremental_roc_auc_vs_A"] = float(metrics["roc_auc"] - base_metrics["roc_auc"])
        rows.append(metrics)

    results = pd.DataFrame(rows)
    results.to_csv(output_csv, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(results))
    vals = results["balanced_accuracy"].to_numpy()
    ax.bar(x, vals, color="#2f6690")
    ax.set_xticks(x)
    ax.set_xticklabels(results["model"], rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Predictive comparison for inferred threat-model labels")
    for idx, row in results.iterrows():
        ax.text(idx, row["balanced_accuracy"] + 0.005, f"{row['balanced_accuracy']:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, min(1.0, max(vals) + 0.08))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return results


def plot_overall_shift(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        return
    work = df.sort_values("mean_shift_pp")
    fig, ax = plt.subplots(figsize=(9.5, max(3.8, 0.45 * len(work))))
    y = np.arange(len(work))
    colors = np.where(work["mean_shift_pp"] >= 0, "#2E8B57", "#B22222")
    ax.barh(y, work["mean_shift_pp"], color=colors, alpha=0.9)
    ax.errorbar(
        work["mean_shift_pp"],
        y,
        xerr=[
            work["mean_shift_pp"] - work["ci_low_pp"],
            work["ci_high_pp"] - work["mean_shift_pp"],
        ],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )
    ax.axvline(0, color="black", linewidth=1)
    labels = [f"{label} (n={n})" for label, n in zip(work["label"], work["n_rows"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Suppressed - Forgotten (percentage points)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_method_heatmap(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        return
    mat = df.pivot(index="method", columns="label", values="mean_shift_pp").fillna(0)
    nmat = df.pivot(index="method", columns="label", values="n_rows").fillna(0)
    fig, ax = plt.subplots(figsize=(max(7.2, 0.9 * mat.shape[1]), 4.8))
    vmax = max(float(np.abs(mat.to_numpy()).max()), 1e-6)
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([f"{col}\n(n={int(nmat[col].sum())})" for col in mat.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            small_n = int(nmat.iloc[i, j]) < 10
            label = f"{val:+.1f}" + ("*" if small_n else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Percentage-point difference")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_centered_heatmap(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        return
    mat = df.pivot(index="method", columns="label", values="centered_suppression_effect_pp").fillna(0)
    nmat = df.pivot(index="method", columns="label", values="n_rows").fillna(0)
    fig, ax = plt.subplots(figsize=(max(7.2, 0.9 * mat.shape[1]), 4.8))
    vmax = max(float(np.abs(mat.to_numpy()).max()), 1e-6)
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([f"{col}\n(n={int(nmat[col].sum())})" for col in mat.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            small_n = int(nmat.iloc[i, j]) < 10
            label = f"{val:+.1f}" + ("*" if small_n else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Method-centered effect (pp)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_axis_outputs(
    merged: pd.DataFrame,
    axis_name: str,
    value_col: str,
    analysis_dir: Path,
    figures_dir: Path,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = bootstrap_shift(merged, value_col=value_col, group_cols=[], n_boot=bootstraps, seed=seed)
    by_method = bootstrap_shift(merged, value_col=value_col, group_cols=["method"], n_boot=bootstraps, seed=seed)
    overall_path = analysis_dir / f"balanced_{axis_name}_shift_overall.csv"
    by_method_path = analysis_dir / f"balanced_{axis_name}_shift_by_method.csv"
    overall.to_csv(overall_path, index=False)
    by_method.to_csv(by_method_path, index=False)

    plot_overall_shift(overall, f"Inferred {axis_name.replace('_', ' ')} shift in suppressed vs forgotten", figures_dir / f"balanced_{axis_name}_shift_overall.png")
    plot_method_heatmap(by_method, f"Per-method inferred {axis_name.replace('_', ' ')} shift", figures_dir / f"balanced_{axis_name}_shift_by_method_heatmap.png")
    return overall, by_method


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    analysis_dir = output_dir / "analysis"
    figures_dir = output_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    question_level = load_question_level(Path(args.pair_dataset))
    labels = load_labels(Path(args.labels))
    merged = merge_inputs(question_level, labels, Path(args.appendix_categories))

    stage_overall, stage_by_method = save_axis_outputs(
        merged,
        axis_name="threat_stage",
        value_col="risk_stage_analysis_final",
        analysis_dir=analysis_dir,
        figures_dir=figures_dir,
        bootstraps=args.bootstrap_samples,
        seed=args.random_seed,
    )
    mech_overall, mech_by_method = save_axis_outputs(
        merged,
        axis_name="risk_mechanism",
        value_col="risk_mechanism_final",
        analysis_dir=analysis_dir,
        figures_dir=figures_dir,
        bootstraps=args.bootstrap_samples,
        seed=args.random_seed + 1,
    )
    proxy_overall, proxy_by_method = save_axis_outputs(
        merged,
        axis_name="proxy_relation",
        value_col="proxy_relation_final",
        analysis_dir=analysis_dir,
        figures_dir=figures_dir,
        bootstraps=args.bootstrap_samples,
        seed=args.random_seed + 2,
    )

    stage_intrinsic_overall = intrinsic_effect(merged, "risk_stage_analysis_final", [])
    stage_intrinsic_by_method = intrinsic_effect(merged, "risk_stage_analysis_final", ["method"])
    stage_intrinsic_overall.to_csv(analysis_dir / "intrinsic_threat_stage_effect_overall.csv", index=False)
    stage_intrinsic_by_method.to_csv(analysis_dir / "intrinsic_threat_stage_effect_by_method.csv", index=False)
    plot_centered_heatmap(
        stage_intrinsic_by_method,
        "Method-centered inferred threat stage effect",
        figures_dir / "method_centered_threat_stage_effect_heatmap.png",
    )

    mech_intrinsic_by_method = intrinsic_effect(merged, "risk_mechanism_final", ["method"])
    proxy_intrinsic_by_method = intrinsic_effect(merged, "proxy_relation_final", ["method"])
    plot_centered_heatmap(
        mech_intrinsic_by_method,
        "Method-centered inferred risk mechanism effect",
        figures_dir / "method_centered_risk_mechanism_effect_heatmap.png",
    )
    plot_centered_heatmap(
        proxy_intrinsic_by_method,
        "Method-centered inferred proxy relationship effect",
        figures_dir / "method_centered_proxy_relation_effect_heatmap.png",
    )

    predictive = run_predictive_models(
        merged,
        output_csv=analysis_dir / "predictive_model_comparison.csv",
        figure_path=figures_dir / "predictive_model_comparison.png",
    )

    print(analysis_dir / "balanced_threat_stage_shift_overall.csv")
    print(analysis_dir / "balanced_threat_stage_shift_by_method.csv")
    print(analysis_dir / "balanced_risk_mechanism_shift_overall.csv")
    print(analysis_dir / "balanced_risk_mechanism_shift_by_method.csv")
    print(analysis_dir / "balanced_proxy_relation_shift_overall.csv")
    print(analysis_dir / "balanced_proxy_relation_shift_by_method.csv")
    print(analysis_dir / "intrinsic_threat_stage_effect_overall.csv")
    print(analysis_dir / "intrinsic_threat_stage_effect_by_method.csv")
    print(analysis_dir / "predictive_model_comparison.csv")
    print(figures_dir / "balanced_threat_stage_shift_overall.png")
    print(figures_dir / "balanced_threat_stage_shift_by_method_heatmap.png")
    print(figures_dir / "method_centered_threat_stage_effect_heatmap.png")
    print(figures_dir / "balanced_risk_mechanism_shift_overall.png")
    print(figures_dir / "method_centered_risk_mechanism_effect_heatmap.png")
    print(figures_dir / "balanced_proxy_relation_shift_overall.png")
    print(figures_dir / "method_centered_proxy_relation_effect_heatmap.png")
    print(figures_dir / "predictive_model_comparison.png")
    print(f"rows={len(merged)} unique_questions={merged['question_idx'].nunique()} models={predictive['model'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
