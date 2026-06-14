from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTCOMES_INPUT = ROOT / "plots" / "pair_level_suppressed_forgotten" / "pair_level_dataset.csv"
DEFAULT_LABELS_INPUT = ROOT / "multilabel" / "wmdp_bio_threat_chain_multilabel_labels_chatgpt_assistant_v1.csv"
MULTILABEL_COLS = [
    "is_ideation",
    "is_design",
    "is_build",
    "is_test",
    "is_learn",
    "is_release",
    "is_test_learn",
    "is_background_knowledge",
    "is_cannot_determine",
]
OPERATIONAL_STAGE_COLS = [
    "is_ideation",
    "is_design",
    "is_build",
    "is_test",
    "is_learn",
    "is_release",
]


@dataclass
class MultiLabelConfig:
    outcomes_input: Path
    labels_input: Path
    output_dir: Path
    bootstrap_repetitions: int = 5000
    cv_repeats: int = 20
    min_cell_n: int = 10
    seed: int = 42


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                text = "nan" if math.isnan(value) else f"{value:.6g}"
            else:
                text = str(value)
            vals.append(text.replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_command_environment(config: MultiLabelConfig, argv: list[str]) -> None:
    ensure_dir(config.output_dir)
    write_text(config.output_dir / "command.txt", " ".join(argv))
    proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    write_text(config.output_dir / "environment.txt", proc.stdout)
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    write_text(config.output_dir / "git_commit.txt", proc.stdout.strip() + "\n")


def append_run_log(config: MultiLabelConfig, message: str) -> None:
    with (ensure_dir(config.output_dir) / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_utc()}] {message}\n")


def load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    if labels["question_id"].duplicated().any():
        raise AssertionError("labels_df question_id must be unique")
    if labels["question_id"].isna().any():
        raise AssertionError("labels_df question_id must be non-null")
    for col in MULTILABEL_COLS + ["needs_review"]:
        labels[col] = labels[col].astype(bool)
    return labels


def load_outcomes(path: Path) -> pd.DataFrame:
    pair = pd.read_csv(path)
    q = (
        pair.loc[pair["checkpoint"].eq("ck8"), ["method", "question_idx", "subset_ck8"]]
        .drop_duplicates(["method", "question_idx"])
        .rename(columns={"question_idx": "question_id", "subset_ck8": "outcome"})
        .copy()
    )
    q = q.loc[q["outcome"].isin(["suppressed", "forgotten"])].copy()
    return q


def build_merged(config: MultiLabelConfig) -> pd.DataFrame:
    outcomes = load_outcomes(config.outcomes_input)
    labels = load_labels(config.labels_input)
    merged = outcomes.merge(labels, on="question_id", how="left", validate="many_to_one")
    if merged["question_id"].isna().any():
        raise AssertionError("merged_df question_id must be non-null")
    if merged.duplicated(["question_id", "method"]).sum() != 0:
        raise AssertionError("merged_df duplicated question_id/method rows")
    for col in MULTILABEL_COLS:
        if not merged[col].dropna().isin([True, False]).all():
            raise AssertionError(f"{col} must be boolean-like")
    merged["suppressed"] = merged["outcome"].eq("suppressed").astype(int)
    merged["background_only"] = merged["is_background_knowledge"] & ~merged[OPERATIONAL_STAGE_COLS].any(axis=1)
    merged["analysis_row_id"] = np.arange(len(merged))
    return merged


def save_eligible_rows(config: MultiLabelConfig, merged: pd.DataFrame) -> None:
    ensure_dir(config.output_dir / "data")
    merged.to_csv(config.output_dir / "data" / "eligible_question_method_rows.csv", index=False)


def stage_question_counts(labels: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in MULTILABEL_COLS:
        rows.append(
            {
                "label": col,
                "n_questions": int(labels[col].sum()),
                "percentage_of_questions": float(labels[col].mean() * 100),
                "n_eligible_question_method_rows": int(merged[col].sum()),
                "n_eligible_unique_questions": int(merged.loc[merged[col], "question_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def stage_cardinality_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    counts = labels["stage_count"].value_counts().sort_index()
    return counts.rename_axis("stage_count").reset_index(name="n_questions")


def overlap_matrix(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in MULTILABEL_COLS:
        a_vals = labels[a].astype(bool)
        for b in MULTILABEL_COLS:
            b_vals = labels[b].astype(bool)
            both = int((a_vals & b_vals).sum())
            union = int((a_vals | b_vals).sum())
            rows.append(
                {
                    "label_a": a,
                    "label_b": b,
                    "n_questions_with_both_labels": both,
                    "jaccard_similarity": float(both / union) if union else float("nan"),
                    "p_label_b_given_a": float(both / a_vals.sum()) if a_vals.sum() else float("nan"),
                    "p_label_a_given_b": float(both / b_vals.sum()) if b_vals.sum() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def phi_correlation_matrix(labels: pd.DataFrame) -> pd.DataFrame:
    mat = labels[MULTILABEL_COLS].astype(int).corr(method="pearson")
    out = mat.reset_index().rename(columns={"index": "label"})
    return out


def heatmap_from_square(df: pd.DataFrame, row_col: str, col_col: str, value_col: str, title: str, subtitle: str, png_path: Path) -> None:
    pivot = df.pivot(index=row_col, columns=col_col, values=value_col)
    arr = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(arr, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            text = "nan" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if not pd.isna(value) and value > np.nanmean(arr) else "black")
    fig.colorbar(im, ax=ax, shrink=0.92)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_audit_outputs(config: MultiLabelConfig, merged: pd.DataFrame) -> None:
    audit_dir = ensure_dir(config.output_dir / "audit")
    supp_dir = ensure_dir(config.output_dir / "figures" / "supplementary")
    labels = load_labels(config.labels_input)
    counts = stage_question_counts(labels, merged)
    counts.to_csv(audit_dir / "multilabel_stage_counts.csv", index=False)
    cardinality = stage_cardinality_distribution(labels)
    cardinality.to_csv(audit_dir / "multilabel_cardinality_distribution.csv", index=False)
    overlap = overlap_matrix(labels)
    overlap.to_csv(audit_dir / "multilabel_overlap_matrix.csv", index=False)
    phi = labels[MULTILABEL_COLS].astype(int).corr(method="pearson")
    phi.to_csv(audit_dir / "multilabel_phi_correlation_matrix.csv")
    heatmap_from_square(overlap, "label_a", "label_b", "jaccard_similarity", "Multilabel overlap heatmap", "Labels overlap; percentages do not sum to 100%.", audit_dir / "multilabel_overlap_heatmap.png")
    heatmap_from_square(overlap, "label_a", "label_b", "jaccard_similarity", "Multilabel overlap heatmap", "Labels overlap; percentages do not sum to 100%.", supp_dir / "multilabel_overlap_heatmap.png")
    phi_df = phi.reset_index().melt(id_vars="index", var_name="label_b", value_name="phi").rename(columns={"index": "label_a"})
    heatmap_from_square(phi_df, "label_a", "label_b", "phi", "Multilabel phi correlation heatmap", "Labels overlap; percentages do not sum to 100%.", audit_dir / "multilabel_phi_correlation_heatmap.png")
    heatmap_from_square(phi_df, "label_a", "label_b", "phi", "Multilabel phi correlation heatmap", "Labels overlap; percentages do not sum to 100%.", supp_dir / "multilabel_phi_correlation_heatmap.png")
    summary = {
        "questions_with_zero_operational_stage_labels": int((labels["stage_count"] == 0).sum()),
        "questions_with_multiple_operational_stage_labels": int((labels["stage_count"] > 1).sum()),
        "questions_marked_background_knowledge": int(labels["is_background_knowledge"].sum()),
        "questions_marked_cannot_determine": int(labels["is_cannot_determine"].sum()),
        "questions_flagged_needs_review": int(labels["needs_review"].sum()),
        "eligible_rows": int(len(merged)),
        "eligible_unique_questions": int(merged["question_id"].nunique()),
    }
    save_json(summary, audit_dir / "audit_summary.json")
    md = [
        "# Multilabel Audit Summary",
        "",
        df_to_markdown_table(pd.DataFrame([summary])),
        "",
        "## Stage Counts",
        "",
        df_to_markdown_table(counts),
        "",
        "## Cardinality",
        "",
        df_to_markdown_table(cardinality),
        "",
    ]
    write_text(audit_dir / "audit_summary.md", "\n".join(md))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(cardinality["stage_count"].astype(str), cardinality["n_questions"], color="#2f6690")
    ax.set_title("Stage cardinality distribution")
    ax.set_xlabel("Number of operational stage labels")
    ax.set_ylabel("Questions")
    ax.text(0, 1.02, "Labels overlap; percentages do not sum to 100%.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(supp_dir / "stage_cardinality_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sample_multivariate_hypergeometric(rng: np.random.Generator, counts: np.ndarray, n: int) -> np.ndarray:
    counts = counts.astype(int, copy=False)
    if n <= 0:
        return np.zeros_like(counts)
    if n >= counts.sum():
        return counts.copy()
    if hasattr(rng, "multivariate_hypergeometric"):
        return rng.multivariate_hypergeometric(counts, n)
    remaining_draws = n
    remaining_total = int(counts.sum())
    sampled = np.zeros_like(counts)
    for idx in range(len(counts) - 1):
        count = int(counts[idx])
        draw = rng.hypergeometric(ngood=count, nbad=remaining_total - count, nsample=remaining_draws)
        sampled[idx] = draw
        remaining_draws -= int(draw)
        remaining_total -= count
    sampled[-1] = remaining_draws
    return sampled


def question_stage_counts(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qids = np.array(sorted(df["question_id"].unique().tolist()))
    qid_to_idx = {qid: idx for idx, qid in enumerate(qids)}
    supp = np.zeros(len(qids), dtype=int)
    forg = np.zeros(len(qids), dtype=int)
    for _, row in df.iterrows():
        idx = qid_to_idx[row["question_id"]]
        if row["outcome"] == "suppressed" and row[label_col]:
            supp[idx] += 1
        elif row["outcome"] == "forgotten" and row[label_col]:
            forg[idx] += 1
    supp_total = df.groupby("question_id")["outcome"].apply(lambda s: (s == "suppressed").sum()).reindex(qids, fill_value=0).to_numpy(dtype=int)
    forg_total = df.groupby("question_id")["outcome"].apply(lambda s: (s == "forgotten").sum()).reindex(qids, fill_value=0).to_numpy(dtype=int)
    return supp, forg, np.vstack([supp_total, forg_total]).T


def balanced_stage_shift(df: pd.DataFrame, label_col: str, n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(df["question_id"].unique().tolist()))
    stage_supp, stage_forg, totals = question_stage_counts(df, label_col)
    draws = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(qids, size=len(qids), replace=True)
        sampled_pos = np.searchsorted(qids, sampled_ids)
        supp_stage_count = int(stage_supp[sampled_pos].sum())
        forg_stage_count = int(stage_forg[sampled_pos].sum())
        supp_total = int(totals[sampled_pos, 0].sum())
        forg_total = int(totals[sampled_pos, 1].sum())
        if supp_total == 0 or forg_total == 0:
            continue
        n = min(supp_total, forg_total)
        sampled_stage_supp = sample_multivariate_hypergeometric(rng, np.array([supp_stage_count, supp_total - supp_stage_count]), n)[0]
        sampled_stage_forg = sample_multivariate_hypergeometric(rng, np.array([forg_stage_count, forg_total - forg_stage_count]), n)[0]
        draws.append(sampled_stage_supp / n - sampled_stage_forg / n)
    arr = np.asarray(draws, dtype=float)
    return {
        "n_questions_with_label": int(df.groupby("question_id")[label_col].max().sum()),
        "n_eligible_rows_with_label": int(df[label_col].sum()),
        "mean_balanced_shift": float(arr.mean()) if len(arr) else float("nan"),
        "median_balanced_shift": float(np.median(arr)) if len(arr) else float("nan"),
        "ci95_low": float(np.quantile(arr, 0.025)) if len(arr) else float("nan"),
        "ci95_high": float(np.quantile(arr, 0.975)) if len(arr) else float("nan"),
        "sign_probability_positive": float((arr > 0).mean()) if len(arr) else float("nan"),
        "sign_probability_negative": float((arr < 0).mean()) if len(arr) else float("nan"),
        "bootstrap_repetitions": int(n_boot),
        "bootstrap_seed": int(seed),
    }


def run_balanced_bootstrap_outputs(config: MultiLabelConfig, merged: pd.DataFrame) -> pd.DataFrame:
    out_dir = ensure_dir(config.output_dir / "balanced_bootstrap")
    rows = []
    by_method_rows = []
    for col in MULTILABEL_COLS:
        result = {"label": col, **balanced_stage_shift(merged, col, config.bootstrap_repetitions, config.seed)}
        rows.append(result)
        for method, group in merged.groupby("method", sort=True):
            by_method_rows.append({"method": method, "label": col, **balanced_stage_shift(group, col, config.bootstrap_repetitions, config.seed)})
    overall = pd.DataFrame(rows)
    by_method = pd.DataFrame(by_method_rows)
    overall.to_csv(out_dir / "overall_balanced_multilabel_stage_shift.csv", index=False)
    by_method.to_csv(out_dir / "by_method_balanced_multilabel_stage_shift.csv", index=False)
    return by_method


def method_centered_ci(df: pd.DataFrame, label_col: str, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(df["question_id"].unique().tolist()))
    vals = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(qids, size=len(qids), replace=True)
        sampled = df[df["question_id"].isin(sampled_ids)].copy()
        base = sampled["suppressed"].mean()
        subset = sampled.loc[sampled[label_col]]
        if subset.empty:
            continue
        vals.append(float(subset["suppressed"].mean() - base))
    arr = np.asarray(vals, dtype=float)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))) if len(arr) else (float("nan"), float("nan"))


def present_absent_ci(df: pd.DataFrame, label_col: str, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(df["question_id"].unique().tolist()))
    vals = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(qids, size=len(qids), replace=True)
        sampled = df[df["question_id"].isin(sampled_ids)].copy()
        present = sampled.loc[sampled[label_col]]
        absent = sampled.loc[~sampled[label_col]]
        if present.empty or absent.empty:
            continue
        vals.append(float(present["suppressed"].mean() - absent["suppressed"].mean()))
    arr = np.asarray(vals, dtype=float)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))) if len(arr) else (float("nan"), float("nan"))


def run_intrinsic_outputs(config: MultiLabelConfig, merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = ensure_dir(config.output_dir / "intrinsic")
    centered_rows = []
    presence_rows = []
    for method, group in merged.groupby("method", sort=True):
        baseline = group["suppressed"].mean()
        for col in MULTILABEL_COLS:
            present = group.loc[group[col]]
            absent = group.loc[~group[col]]
            low1, high1 = method_centered_ci(group, col, config.bootstrap_repetitions, config.seed)
            centered_rows.append(
                {
                    "method": method,
                    "label": col,
                    "n_rows_with_label": int(len(present)),
                    "n_unique_questions_with_label": int(present["question_id"].nunique()),
                    "suppressed_count_with_label": int(present["suppressed"].sum()),
                    "forgotten_count_with_label": int(len(present) - present["suppressed"].sum()),
                    "suppressed_rate_with_label": float(present["suppressed"].mean()) if len(present) else float("nan"),
                    "suppressed_rate_for_method": float(baseline),
                    "method_centered_effect": float(present["suppressed"].mean() - baseline) if len(present) else float("nan"),
                    "ci95_low": low1,
                    "ci95_high": high1,
                }
            )
            low2, high2 = present_absent_ci(group, col, config.bootstrap_repetitions, config.seed)
            presence_rows.append(
                {
                    "method": method,
                    "label": col,
                    "n_present": int(present["question_id"].nunique()),
                    "n_absent": int(absent["question_id"].nunique()),
                    "suppressed_rate_present": float(present["suppressed"].mean()) if len(present) else float("nan"),
                    "suppressed_rate_absent": float(absent["suppressed"].mean()) if len(absent) else float("nan"),
                    "present_minus_absent_effect": float(present["suppressed"].mean() - absent["suppressed"].mean()) if len(present) and len(absent) else float("nan"),
                    "ci95_low": low2,
                    "ci95_high": high2,
                }
            )
    centered = pd.DataFrame(centered_rows)
    presence = pd.DataFrame(presence_rows)
    centered.to_csv(out_dir / "method_centered_multilabel_stage_effects.csv", index=False)
    presence.to_csv(out_dir / "stage_present_vs_absent_effects.csv", index=False)
    return centered, presence


def repeated_group_splits(question_ids: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    unique_q = np.array(sorted(set(question_ids.tolist())))
    rng = np.random.default_rng(seed)
    splits = []
    for repeat in range(n_repeats):
        shuffled = unique_q.copy()
        rng.shuffle(shuffled)
        fold_groups = np.array_split(shuffled, n_splits)
        for fold_idx, test_groups in enumerate(fold_groups):
            mask = np.isin(question_ids, test_groups)
            splits.append((repeat, np.where(~mask)[0], np.where(mask)[0]))
    return splits


def build_pipeline(cat_features: list[str], num_features: list[str]) -> Pipeline:
    transformers = []
    if cat_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_features,
            )
        )
    if num_features:
        transformers.append(("num", "passthrough", num_features))
    pre = ColumnTransformer(transformers=transformers)
    return Pipeline(steps=[("preprocess", pre), ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42))])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def summarize_metric_series(series: pd.Series) -> dict[str, float]:
    vals = series.to_numpy(dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "standard_deviation": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
    }


def prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["method"] = work["method"].astype(str)
    for col in MULTILABEL_COLS:
        work[col] = work[col].astype(int)
    work["method_x_design"] = work["method"] + "__" + work["is_design"].astype(str)
    work["method_x_build"] = work["method"] + "__" + work["is_build"].astype(str)
    work["method_x_release"] = work["method"] + "__" + work["is_release"].astype(str)
    work["method_x_test_learn"] = work["method"] + "__" + work["is_test_learn"].astype(str)
    return work


def run_predictive_outputs(config: MultiLabelConfig, merged: pd.DataFrame) -> pd.DataFrame:
    out_dir = ensure_dir(config.output_dir / "predictive")
    work = prepare_model_input(merged)
    y = work["suppressed"].to_numpy()
    qids = work["question_id"].to_numpy()
    n_splits = min(5, len(np.unique(qids)))
    splits = repeated_group_splits(qids, n_splits, config.cv_repeats, config.seed)
    specs = [
        ("Model A", ["method"], []),
        ("Model B", ["method"], ["is_ideation", "is_design", "is_build", "is_test", "is_learn", "is_release", "is_background_knowledge", "is_cannot_determine"]),
        ("Model C", ["method"], ["is_ideation", "is_design", "is_build", "is_test_learn", "is_release", "is_background_knowledge", "is_cannot_determine"]),
        ("Model D", ["method", "method_x_design", "method_x_build", "method_x_release", "method_x_test_learn"], ["is_ideation", "is_design", "is_build", "is_test_learn", "is_release", "is_background_knowledge", "is_cannot_determine"]),
    ]
    metric_rows = []
    oof_rows = []
    for model_name, cat_features, num_features in specs:
        for repeat in range(config.cv_repeats):
            repeat_splits = [item for item in splits if item[0] == repeat]
            oof_prob = np.zeros(len(work), dtype=float)
            for _, train_idx, test_idx in repeat_splits:
                if set(qids[train_idx]) & set(qids[test_idx]):
                    raise AssertionError("grouped_cv_has_no_question_id_overlap failed")
                pipe = build_pipeline(cat_features, num_features)
                feature_cols = cat_features + num_features
                pipe.fit(work.iloc[train_idx][feature_cols], y[train_idx])
                oof_prob[test_idx] = pipe.predict_proba(work.iloc[test_idx][feature_cols])[:, 1]
            metrics = compute_metrics(y, oof_prob)
            metrics["model"] = model_name
            metrics["repeat"] = repeat
            metric_rows.append(metrics)
            for idx, prob in enumerate(oof_prob):
                oof_rows.append(
                    {
                        "analysis_row_id": int(work.iloc[idx]["analysis_row_id"]),
                        "question_id": int(work.iloc[idx]["question_id"]),
                        "method": work.iloc[idx]["method"],
                        "outcome": work.iloc[idx]["outcome"],
                        "model": model_name,
                        "repeat": int(repeat),
                        "oof_pred_probability": float(prob),
                    }
                )
    metrics_df = pd.DataFrame(metric_rows)
    oof_df = pd.DataFrame(oof_rows)
    summary_rows = []
    base = metrics_df.loc[metrics_df["model"].eq("Model A")].sort_values("repeat").reset_index(drop=True)
    for model_name, group in metrics_df.groupby("model", sort=False):
        row = {"model": model_name}
        for metric in ["balanced_accuracy", "macro_f1", "roc_auc", "average_precision", "log_loss", "brier_score"]:
            stats = summarize_metric_series(group[metric])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        if model_name != "Model A":
            aligned = group.sort_values("repeat").reset_index(drop=True)
            for metric in ["balanced_accuracy", "macro_f1", "roc_auc", "log_loss", "brier_score"]:
                delta = aligned[metric] - base[metric]
                stats = summarize_metric_series(delta)
                for key, value in stats.items():
                    row[f"delta_{metric}_vs_A_{key}"] = value
        else:
            for metric in ["balanced_accuracy", "macro_f1", "roc_auc", "log_loss", "brier_score"]:
                for key in ["mean", "standard_deviation", "ci95_low", "ci95_high"]:
                    row[f"delta_{metric}_vs_A_{key}"] = 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "model_comparison_repeated_grouped_cv.csv", index=False)
    summary[["model"] + [c for c in summary.columns if c.startswith("delta_")]].to_csv(out_dir / "model_comparison_vs_method_only.csv", index=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    return summary


def subset_rows(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "S0":
        return df.copy()
    if subset == "S1":
        return df.loc[~df["needs_review"]].copy()
    if subset == "S2":
        return df.loc[~df["is_cannot_determine"]].copy()
    if subset == "S3":
        return df.loc[~df["background_only"]].copy()
    if subset == "S4":
        return df.loc[df["stage_count"] >= 1].copy()
    raise ValueError(subset)


def run_sensitivity_outputs(config: MultiLabelConfig, merged: pd.DataFrame) -> None:
    out_dir = ensure_dir(config.output_dir / "sensitivity")
    subset_counts = []
    balanced_rows = []
    predictive_rows = []
    for subset in ["S0", "S1", "S2", "S3", "S4"]:
        sub = subset_rows(merged, subset)
        subset_counts.append({"subset": subset, "n_rows": int(len(sub)), "n_unique_questions": int(sub["question_id"].nunique())})
        if sub.empty:
            continue
        for col in MULTILABEL_COLS:
            balanced_rows.append({"subset": subset, "label": col, **balanced_stage_shift(sub, col, config.bootstrap_repetitions, config.seed)})
        if sub["question_id"].nunique() >= 5:
            pred_dir = ensure_dir(out_dir / subset)
            summary = run_predictive_outputs(MultiLabelConfig(config.outcomes_input, config.labels_input, pred_dir, config.bootstrap_repetitions, config.cv_repeats, config.min_cell_n, config.seed), sub)
            summary.insert(0, "subset", subset)
            predictive_rows.append(summary)
    pd.DataFrame(subset_counts).to_csv(out_dir / "subset_counts.csv", index=False)
    pd.DataFrame(balanced_rows).to_csv(out_dir / "balanced_stage_shift_by_subset.csv", index=False)
    if predictive_rows:
        pd.concat(predictive_rows, ignore_index=True).to_csv(out_dir / "predictive_model_comparison_by_subset.csv", index=False)


def effect_heatmap(df: pd.DataFrame, value_col: str, count_col: str, title: str, subtitle: str, png_path: Path, min_cell_n: int, absent_count_col: str | None = None) -> None:
    mat = df.pivot(index="method", columns="label", values=value_col).fillna(0)
    nmat = df.pivot(index="method", columns="label", values=count_col).fillna(0)
    amat = df.pivot(index="method", columns="label", values=absent_count_col).fillna(9999) if absent_count_col else None
    arr = mat.to_numpy(dtype=float)
    mask = nmat.to_numpy() < min_cell_n
    if amat is not None:
        mask = mask | (amat.to_numpy() < min_cell_n)
    display = arr.copy()
    display[mask] = np.nan
    vmax = max(float(np.nanmax(np.abs(arr))), 1e-9)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#d9d9d9")
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * mat.shape[1]), 4.8))
    im = ax.imshow(display, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mask[i, j]:
                ax.text(j, i, "n<10", ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, f"{arr[i,j]*100:+.1f}\nq={int(nmat.iloc[i,j])}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.92)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def balanced_bar_plot(df: pd.DataFrame, title: str, subtitle: str, png_path: Path) -> None:
    work = df.sort_values("mean_balanced_shift")
    vals = work["mean_balanced_shift"].to_numpy() * 100
    low = work["ci95_low"].to_numpy() * 100
    high = work["ci95_high"].to_numpy() * 100
    labels = [f"{lab}\nq={nq}" for lab, nq in zip(work["label"], work["n_questions_with_label"])]
    y = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.5 * len(work))))
    ax.barh(y, vals, color=np.where(vals >= 0, "#2E8B57", "#B22222"))
    ax.errorbar(vals, y, xerr=[vals - low, high - vals], fmt="none", ecolor="black", capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Suppressed - Forgotten (percentage points)")
    ax.set_title(title)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def predictive_ci_plot(summary: pd.DataFrame, png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(summary))
    mean = summary["balanced_accuracy_mean"].to_numpy()
    low = summary["balanced_accuracy_ci95_low"].to_numpy()
    high = summary["balanced_accuracy_ci95_high"].to_numpy()
    ax.errorbar(x, mean, yerr=[mean - low, high - mean], fmt="o", color="#1f4e79", ecolor="#7aa6d1", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Predictive model comparison with CI")
    ax.text(0, 1.02, "Labels overlap; percentages do not sum to 100%.\nPositive = relatively more suppressed; negative = relatively more forgotten.", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_figures_outputs(config: MultiLabelConfig) -> None:
    main = ensure_dir(config.output_dir / "figures" / "main")
    supp = ensure_dir(config.output_dir / "figures" / "supplementary")
    overall = pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_multilabel_stage_shift.csv")
    by_method = pd.read_csv(config.output_dir / "intrinsic" / "method_centered_multilabel_stage_effects.csv")
    present_absent = pd.read_csv(config.output_dir / "intrinsic" / "stage_present_vs_absent_effects.csv")
    summary = pd.read_csv(config.output_dir / "predictive" / "model_comparison_repeated_grouped_cv.csv")
    subtitle = "Labels overlap; percentages do not sum to 100%.\nPositive = relatively more suppressed; negative = relatively more forgotten."
    balanced_bar_plot(overall, "Balanced multilabel stage shift overall", subtitle, main / "balanced_multilabel_stage_shift_overall.png")
    effect_heatmap(by_method, "method_centered_effect", "n_unique_questions_with_label", "Method-centered multilabel stage heatmap", subtitle, main / "method_centered_multilabel_stage_heatmap.png", config.min_cell_n)
    predictive_ci_plot(summary, main / "predictive_model_comparison_with_ci.png")
    effect_heatmap(present_absent, "present_minus_absent_effect", "n_present", "Stage present vs absent heatmap", subtitle, supp / "stage_present_vs_absent_heatmap.png", config.min_cell_n, "n_absent")


LATEX_CAPTION = "Question-level threat-chain labels are inferred overlapping metadata, not official WMDP annotations."


def save_table_with_latex(df: pd.DataFrame, csv_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    cols = [str(col).replace("_", r"\_") for col in df.columns]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{LATEX_CAPTION}}}",
        rf"\label{{tab:{csv_path.stem.replace('.', '_')}}}",
        r"\begin{tabular}{" + "l" * len(cols) + r"}",
        r"\hline",
        " & ".join(cols) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            v = row[col]
            vals.append(("nan" if isinstance(v, float) and math.isnan(v) else str(v)).replace("_", r"\_"))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    write_text(csv_path.with_suffix(".tex"), "\n".join(lines))


VALIDATION_SCRIPT = r'''from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
labels_df = pd.read_csv(ROOT.parent / "multilabel" / "wmdp_bio_threat_chain_multilabel_labels_chatgpt_assistant_v1.csv")
merged_df = pd.read_csv(ROOT / "data" / "eligible_question_method_rows.csv")
assert labels_df["question_id"].is_unique
assert merged_df.duplicated(["question_id", "method"]).sum() == 0
for col in ["is_ideation","is_design","is_build","is_test","is_learn","is_release","is_test_learn","is_background_knowledge","is_cannot_determine"]:
    assert merged_df[col].dropna().isin([True, False, 0, 1]).all()
assert (merged_df["stage_count"] >= 0).all()
assert (merged_df["stage_count"] <= 6).all()
assert pd.read_csv(ROOT / "balanced_bootstrap" / "overall_balanced_multilabel_stage_shift.csv")["bootstrap_repetitions"].ge(5000).all()
assert (ROOT / "figures" / "main" / "method_centered_multilabel_stage_heatmap.png").exists()
print("Validation checks passed.")
'''


def write_validation_script(config: MultiLabelConfig) -> Path:
    val_dir = ensure_dir(config.output_dir / "validation")
    path = val_dir / "run_multilabel_validation_checks.py"
    write_text(path, VALIDATION_SCRIPT)
    return path


def run_validation(config: MultiLabelConfig) -> None:
    path = write_validation_script(config)
    proc = subprocess.run([sys.executable, str(path)], cwd=ROOT, capture_output=True, text=True, check=False)
    report = [
        "# Validation Report",
        "",
        f"- Exit code: {proc.returncode}",
        "",
        "## Stdout",
        "",
        "```text",
        proc.stdout,
        "```",
        "",
        "## Stderr",
        "",
        "```text",
        proc.stderr,
        "```",
    ]
    write_text(config.output_dir / "validation" / "validation_report.md", "\n".join(report))
    if proc.returncode != 0:
        raise RuntimeError("Validation checks failed.")


def generate_report(config: MultiLabelConfig) -> None:
    audit = json.loads((config.output_dir / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
    overall = pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_multilabel_stage_shift.csv")
    centered = pd.read_csv(config.output_dir / "intrinsic" / "method_centered_multilabel_stage_effects.csv")
    presence = pd.read_csv(config.output_dir / "intrinsic" / "stage_present_vs_absent_effects.csv")
    predictive = pd.read_csv(config.output_dir / "predictive" / "model_comparison_repeated_grouped_cv.csv")
    sens = pd.read_csv(config.output_dir / "sensitivity" / "subset_counts.csv")
    lines = [
        "# Multilabel Analysis Report",
        "",
        "Labels overlap; percentages do not sum to 100%. All labels are inferred metadata, not official WMDP annotations.",
        "",
        "## Audit",
        "",
        df_to_markdown_table(pd.DataFrame([audit])),
        "",
        "## Balanced stage shifts",
        "",
        df_to_markdown_table(overall),
        "",
        "## Method-centered effects",
        "",
        df_to_markdown_table(centered.loc[centered["n_unique_questions_with_label"] >= config.min_cell_n].head(30)),
        "",
        "## Present vs absent effects",
        "",
        df_to_markdown_table(presence.head(30)),
        "",
        "## Predictive models",
        "",
        df_to_markdown_table(predictive),
        "",
        "## Sensitivity subsets",
        "",
        df_to_markdown_table(sens),
        "",
        "## Interpretation",
        "",
        "Using inferred overlapping threat-chain labels, stage-level signals should be read as overlapping prevalence effects rather than mutually exclusive category shifts. Positive values indicate relatively more suppressed rows; negative values indicate relatively more forgotten rows.",
        "",
    ]
    write_text(config.output_dir / "REPORT.md", "\n".join(lines))


def run_all(config: MultiLabelConfig) -> None:
    save_command_environment(config, sys.argv)
    append_run_log(config, "Starting multilabel pipeline.")
    merged = build_merged(config)
    save_eligible_rows(config, merged)
    append_run_log(config, f"Eligible rows prepared: {len(merged)} rows.")
    run_audit_outputs(config, merged)
    append_run_log(config, "Audit complete.")
    run_balanced_bootstrap_outputs(config, merged)
    append_run_log(config, "Balanced bootstrap complete.")
    run_intrinsic_outputs(config, merged)
    append_run_log(config, "Intrinsic analyses complete.")
    run_predictive_outputs(config, merged)
    append_run_log(config, "Predictive analyses complete.")
    run_sensitivity_outputs(config, merged)
    append_run_log(config, "Sensitivity analyses complete.")
    run_figures_outputs(config)
    append_run_log(config, "Figures complete.")
    run_validation(config)
    append_run_log(config, "Validation complete.")
    generate_report(config)
    append_run_log(config, "Report complete.")
