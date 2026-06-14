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
DEFAULT_LABELS_INPUT = ROOT / "stem multilabel" / "wmdp_bio_threat_chain_multilabel_labels_stem_only_chatgpt_assistant_v1.csv"
DEFAULT_COMPARISON_INPUT = ROOT / "stem multilabel" / "wmdp_bio_threat_chain_multilabel_stem_only_vs_full_mcq_comparison.csv"
DEFAULT_STERILE_AUDIT_INPUT = ROOT / "stem multilabel" / "wmdp_bio_threat_chain_multilabel_sterile_stem_audit.csv"
MULTILABEL_COLS = [
    "is_ideation",
    "is_design",
    "is_build",
    "is_test",
    "is_learn",
    "is_release",
    "is_test_learn",
    "is_background_knowledge",
]


@dataclass
class StemOnlyConfig:
    outcomes_input: Path
    labels_input: Path
    comparison_input: Path
    sterile_audit_input: Path
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
                vals.append("nan" if math.isnan(value) else f"{value:.6g}")
            else:
                vals.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_command_environment(config: StemOnlyConfig, argv: list[str]) -> None:
    ensure_dir(config.output_dir)
    write_text(config.output_dir / "command.txt", " ".join(argv))
    proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    write_text(config.output_dir / "environment.txt", proc.stdout)
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    write_text(config.output_dir / "git_commit.txt", proc.stdout.strip() + "\n")


def append_run_log(config: StemOnlyConfig, message: str) -> None:
    with (ensure_dir(config.output_dir) / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_utc()}] {message}\n")


def load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    assert labels["question_id"].is_unique
    for col in MULTILABEL_COLS + ["needs_review", "is_sterile_stem_question"]:
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
    return q.loc[q["outcome"].isin(["suppressed", "forgotten"])].copy()


def build_merged(config: StemOnlyConfig) -> pd.DataFrame:
    outcomes = load_outcomes(config.outcomes_input)
    labels = load_labels(config.labels_input)
    comp = pd.read_csv(config.comparison_input)
    merged = outcomes.merge(labels, on="question_id", how="left", validate="many_to_one")
    merged = merged.merge(
        comp[
            [
                "question_id",
                "stem_only_stage_count",
                "full_mcq_stage_count",
                "choice_added_stage_count",
                "choice_removed_stage_count",
                "labels_added_with_choices",
                "labels_present_stem_only_not_full_mcq",
                "stage_set_jaccard",
                "is_sterile_stem_question",
                "is_background_knowledge_stem_only",
            ]
        ],
        on="question_id",
        how="left",
        suffixes=("", "_comp"),
    )
    assert merged["question_id"].notna().all()
    assert merged.duplicated(["question_id", "method"]).sum() == 0
    merged["suppressed"] = merged["outcome"].eq("suppressed").astype(int)
    merged["background_only"] = merged["is_background_knowledge"] & merged["stage_count"].eq(0)
    merged["primary_subset"] = ~merged["is_sterile_stem_question"]
    merged["analysis_row_id"] = np.arange(len(merged))
    return merged


def save_eligible_rows(config: StemOnlyConfig, merged: pd.DataFrame) -> None:
    ensure_dir(config.output_dir / "data")
    merged.to_csv(config.output_dir / "data" / "eligible_question_method_rows.csv", index=False)


def sample_multivariate_hypergeometric(rng: np.random.Generator, counts: np.ndarray, n: int) -> np.ndarray:
    counts = counts.astype(int, copy=False)
    if n <= 0:
        return np.zeros_like(counts)
    if n >= counts.sum():
        return counts.copy()
    if hasattr(rng, "multivariate_hypergeometric"):
        return rng.multivariate_hypergeometric(counts, n)
    remaining_draws = int(n)
    remaining_total = int(counts.sum())
    sampled = np.zeros_like(counts)
    for idx in range(len(counts) - 1):
        count = int(counts[idx])
        draw = rng.hypergeometric(count, remaining_total - count, remaining_draws)
        sampled[idx] = draw
        remaining_draws -= int(draw)
        remaining_total -= count
    sampled[-1] = remaining_draws
    return sampled


def question_stage_counts(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qids = np.array(sorted(df["question_id"].unique().tolist()))
    qid_to_idx = {qid: idx for idx, qid in enumerate(qids)}
    supp_label = np.zeros(len(qids), dtype=int)
    forg_label = np.zeros(len(qids), dtype=int)
    supp_total = np.zeros(len(qids), dtype=int)
    forg_total = np.zeros(len(qids), dtype=int)
    for _, row in df.iterrows():
        idx = qid_to_idx[row["question_id"]]
        if row["outcome"] == "suppressed":
            supp_total[idx] += 1
            if row[label_col]:
                supp_label[idx] += 1
        else:
            forg_total[idx] += 1
            if row[label_col]:
                forg_label[idx] += 1
    return supp_label, forg_label, np.vstack([supp_total, forg_total]).T


def balanced_stage_shift(df: pd.DataFrame, label_col: str, n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(df["question_id"].unique().tolist()))
    supp_label, forg_label, totals = question_stage_counts(df, label_col)
    draws = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(qids, size=len(qids), replace=True)
        sampled_pos = np.searchsorted(qids, sampled_ids)
        supp_label_count = int(supp_label[sampled_pos].sum())
        forg_label_count = int(forg_label[sampled_pos].sum())
        supp_total = int(totals[sampled_pos, 0].sum())
        forg_total = int(totals[sampled_pos, 1].sum())
        if supp_total == 0 or forg_total == 0:
            continue
        n = min(supp_total, forg_total)
        sampled_supp = sample_multivariate_hypergeometric(rng, np.array([supp_label_count, supp_total - supp_label_count]), n)[0]
        sampled_forg = sample_multivariate_hypergeometric(rng, np.array([forg_label_count, forg_total - forg_label_count]), n)[0]
        draws.append(sampled_supp / n - sampled_forg / n)
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


def method_centered_ci(df: pd.DataFrame, label_col: str, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(df["question_id"].unique().tolist()))
    vals = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(qids, size=len(qids), replace=True)
        sampled = df[df["question_id"].isin(sampled_ids)]
        base = sampled["suppressed"].mean()
        subset = sampled.loc[sampled[label_col]]
        if subset.empty:
            continue
        vals.append(float(subset["suppressed"].mean() - base))
    arr = np.asarray(vals, dtype=float)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))) if len(arr) else (float("nan"), float("nan"))


def stage_counts(labels: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = merged.loc[merged["primary_subset"]].copy()
    for col in MULTILABEL_COLS:
        rows.append(
            {
                "label": col,
                "n_questions": int(labels[col].sum()),
                "percentage_of_questions": float(labels[col].mean() * 100),
                "n_primary_eligible_rows_with_label": int(primary[col].sum()),
                "n_primary_eligible_unique_questions": int(primary.loc[primary[col], "question_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def run_audit_outputs(config: StemOnlyConfig, merged: pd.DataFrame) -> None:
    audit_dir = ensure_dir(config.output_dir / "audit")
    fig_dir = ensure_dir(config.output_dir / "figures")
    labels = load_labels(config.labels_input)
    primary = merged.loc[merged["primary_subset"]].copy()
    sterile_summary = pd.DataFrame(
        [
            {
                "n_questions_sterile": int(labels["is_sterile_stem_question"].sum()),
                "sterile_stem_rate": float(labels["is_sterile_stem_question"].mean()),
                "n_primary_rows": int(len(primary)),
                "n_all_rows": int(len(merged)),
                "n_primary_unique_questions": int(primary["question_id"].nunique()),
            }
        ]
    )
    sterile_summary.to_csv(audit_dir / "sterile_stem_summary.csv", index=False)
    stage_counts(labels, merged).to_csv(audit_dir / "stem_only_stage_counts.csv", index=False)
    comp = pd.read_csv(config.comparison_input)
    comp_summary = pd.DataFrame(
        [
            {
                "mean_stage_set_jaccard": float(comp["stage_set_jaccard"].mean()),
                "median_stage_set_jaccard": float(comp["stage_set_jaccard"].median()),
                "mean_choice_added_stage_count": float(comp["choice_added_stage_count"].mean()),
                "sterile_stem_rate": float(comp["is_sterile_stem_question"].mean()),
                "n_questions": int(len(comp)),
            }
        ]
    )
    comp_summary.to_csv(audit_dir / "full_mcq_vs_stem_only_comparison_summary.csv", index=False)
    freq = comp["labels_added_with_choices"].fillna("").replace("", "<none>").value_counts().rename_axis("labels_added_with_choices").reset_index(name="count")
    freq.to_csv(audit_dir / "choice_added_labels_frequency.csv", index=False)
    sterile_audit = pd.read_csv(config.sterile_audit_input)
    sterile_audit.to_csv(audit_dir / "sterile_stem_examples.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sterile_counts = merged.groupby("outcome")["is_sterile_stem_question"].mean().reindex(["suppressed", "forgotten"]).fillna(0) * 100
    ax.bar(sterile_counts.index, sterile_counts.values, color="#8c564b")
    ax.set_ylabel("Sterile stem rate (%)")
    ax.set_title("Sterile stem rate in suppressed vs forgotten")
    ax.text(0, 1.02, "Inferred multi-select labels; sterile stems are excluded from the primary stage analysis.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "sterile_stem_suppressed_vs_forgotten.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(comp["stem_only_stage_count"], bins=np.arange(-0.5, comp["stem_only_stage_count"].max() + 1.5, 1), alpha=0.6, label="stem only")
    ax.hist(comp["full_mcq_stage_count"], bins=np.arange(-0.5, comp["full_mcq_stage_count"].max() + 1.5, 1), alpha=0.6, label="full MCQ")
    ax.set_title("Full-MCQ vs stem-only stage counts")
    ax.set_xlabel("Stage count")
    ax.set_ylabel("Questions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "full_mcq_vs_stem_only_stage_count.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(comp["choice_added_stage_count"], bins=np.arange(-0.5, comp["choice_added_stage_count"].max() + 1.5, 1), color="#2f6690")
    ax.set_title("Choice-added stage count distribution")
    ax.set_xlabel("Stages added with choices")
    ax.set_ylabel("Questions")
    fig.tight_layout()
    fig.savefig(fig_dir / "choice_added_stage_count_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(comp["stage_set_jaccard"], bins=20, color="#2ca02c")
    ax.set_title("Stage-set Jaccard distribution")
    ax.set_xlabel("Jaccard")
    ax.set_ylabel("Questions")
    fig.tight_layout()
    fig.savefig(fig_dir / "stage_set_jaccard_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "eligible_rows": int(len(merged)),
        "primary_rows_excluding_sterile": int(len(primary)),
        "sterile_rows": int(merged["is_sterile_stem_question"].sum()),
        "background_only_rows": int(merged["background_only"].sum()),
        "needs_review_rows": int(merged["needs_review"].sum()),
        "mean_stage_set_jaccard": float(comp["stage_set_jaccard"].mean()),
        "mean_choice_added_stage_count": float(comp["choice_added_stage_count"].mean()),
    }
    save_json(summary, audit_dir / "audit_summary.json")
    write_text(audit_dir / "audit_summary.md", "# Stem-only audit\n\n" + df_to_markdown_table(pd.DataFrame([summary])))


def run_balanced_outputs(config: StemOnlyConfig, merged: pd.DataFrame) -> pd.DataFrame:
    out_dir = ensure_dir(config.output_dir / "balanced_bootstrap")
    primary = merged.loc[merged["primary_subset"]].copy()
    rows = []
    for col in MULTILABEL_COLS:
        rows.append({"label": col, **balanced_stage_shift(primary, col, config.bootstrap_repetitions, config.seed)})
    overall = pd.DataFrame(rows)
    overall.to_csv(out_dir / "overall_balanced_multilabel_stage_shift.csv", index=False)
    return overall


def run_intrinsic_outputs(config: StemOnlyConfig, merged: pd.DataFrame) -> pd.DataFrame:
    out_dir = ensure_dir(config.output_dir / "intrinsic")
    primary = merged.loc[merged["primary_subset"]].copy()
    rows = []
    for method, group in primary.groupby("method", sort=True):
        baseline = group["suppressed"].mean()
        for col in MULTILABEL_COLS:
            present = group.loc[group[col]]
            low, high = method_centered_ci(group, col, config.bootstrap_repetitions, config.seed)
            rows.append(
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
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "method_centered_multilabel_stage_effects.csv", index=False)
    return out


def repeated_group_splits(question_ids: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    unique_q = np.array(sorted(set(question_ids.tolist())))
    rng = np.random.default_rng(seed)
    splits = []
    for repeat in range(n_repeats):
        shuffled = unique_q.copy()
        rng.shuffle(shuffled)
        fold_groups = np.array_split(shuffled, n_splits)
        for test_groups in fold_groups:
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


def run_predictive_outputs(config: StemOnlyConfig, merged: pd.DataFrame) -> pd.DataFrame:
    out_dir = ensure_dir(config.output_dir / "predictive")
    primary = merged.loc[merged["primary_subset"]].copy()
    primary["method"] = primary["method"].astype(str)
    for col in MULTILABEL_COLS:
        primary[col] = primary[col].astype(int)
    primary["is_sterile_stem_question"] = primary["is_sterile_stem_question"].astype(int)
    y = primary["suppressed"].to_numpy()
    qids = primary["question_id"].to_numpy()
    n_splits = min(5, len(np.unique(qids)))
    splits = repeated_group_splits(qids, n_splits, config.cv_repeats, config.seed)
    specs = [
        ("Model A", ["method"], []),
        ("Model B", ["method"], ["is_ideation", "is_design", "is_build", "is_test", "is_learn", "is_release", "is_background_knowledge"]),
        ("Model C", ["method"], ["is_ideation", "is_design", "is_build", "is_test_learn", "is_release", "is_background_knowledge"]),
        ("Model D", ["method"], ["is_ideation", "is_design", "is_build", "is_test_learn", "is_release", "is_background_knowledge", "stem_only_stage_count", "choice_added_stage_count", "stage_set_jaccard", "is_sterile_stem_question"]),
    ]
    metric_rows = []
    for model_name, cat_features, num_features in specs:
        features = cat_features + num_features
        for repeat in range(config.cv_repeats):
            repeat_splits = [item for item in splits if item[0] == repeat]
            oof_prob = np.zeros(len(primary), dtype=float)
            for _, train_idx, test_idx in repeat_splits:
                if set(qids[train_idx]) & set(qids[test_idx]):
                    raise AssertionError("grouped_cv_has_no_question_id_overlap failed")
                pipe = build_pipeline(cat_features, num_features)
                pipe.fit(primary.iloc[train_idx][features], y[train_idx])
                oof_prob[test_idx] = pipe.predict_proba(primary.iloc[test_idx][features])[:, 1]
            metrics = compute_metrics(y, oof_prob)
            metrics["model"] = model_name
            metrics["repeat"] = repeat
            metric_rows.append(metrics)
    metrics_df = pd.DataFrame(metric_rows)
    base = metrics_df.loc[metrics_df["model"].eq("Model A")].sort_values("repeat").reset_index(drop=True)
    rows = []
    for model_name, group in metrics_df.groupby("model", sort=False):
        row = {"model": model_name}
        for metric in ["balanced_accuracy", "macro_f1", "roc_auc", "average_precision", "log_loss", "brier_score"]:
            stats = summarize_metric_series(group[metric])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        if model_name != "Model A":
            aligned = group.sort_values("repeat").reset_index(drop=True)
            for metric in ["balanced_accuracy", "macro_f1", "roc_auc", "log_loss", "brier_score"]:
                stats = summarize_metric_series(aligned[metric] - base[metric])
                for key, value in stats.items():
                    row[f"delta_{metric}_vs_A_{key}"] = value
        else:
            for metric in ["balanced_accuracy", "macro_f1", "roc_auc", "log_loss", "brier_score"]:
                for key in ["mean", "standard_deviation", "ci95_low", "ci95_high"]:
                    row[f"delta_{metric}_vs_A_{key}"] = 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "model_comparison_repeated_grouped_cv.csv", index=False)
    out.to_csv(out_dir / "distractor_sensitivity_model_comparison.csv", index=False)
    return out


def subset_rows(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "S0":
        return df.copy()
    if subset == "S1":
        return df.loc[~df["is_sterile_stem_question"]].copy()
    if subset == "S2":
        return df.loc[~df["is_sterile_stem_question"] & ~df["background_only"]].copy()
    if subset == "S3":
        return df.loc[~df["is_sterile_stem_question"] & ~df["background_only"] & ~df["needs_review"]].copy()
    raise ValueError(subset)


def run_sensitivity_outputs(config: StemOnlyConfig, merged: pd.DataFrame) -> None:
    out_dir = ensure_dir(config.output_dir / "sensitivity")
    subset_counts = []
    balanced_rows = []
    predictive_rows = []
    for subset in ["S0", "S1", "S2", "S3"]:
        sub = subset_rows(merged, subset)
        subset_counts.append({"subset": subset, "n_rows": int(len(sub)), "n_unique_questions": int(sub["question_id"].nunique())})
        if sub.empty:
            continue
        for col in MULTILABEL_COLS:
            balanced_rows.append({"subset": subset, "label": col, **balanced_stage_shift(sub, col, config.bootstrap_repetitions, config.seed)})
        if sub["question_id"].nunique() >= 5:
            pred = run_predictive_outputs(
                StemOnlyConfig(
                    outcomes_input=config.outcomes_input,
                    labels_input=config.labels_input,
                    comparison_input=config.comparison_input,
                    sterile_audit_input=config.sterile_audit_input,
                    output_dir=ensure_dir(out_dir / subset),
                    bootstrap_repetitions=config.bootstrap_repetitions,
                    cv_repeats=config.cv_repeats,
                    min_cell_n=config.min_cell_n,
                    seed=config.seed,
                ),
                sub,
            )
            pred.insert(0, "subset", subset)
            predictive_rows.append(pred)
    pd.DataFrame(subset_counts).to_csv(out_dir / "subset_counts.csv", index=False)
    pd.DataFrame(balanced_rows).to_csv(out_dir / "balanced_stage_shift_by_subset.csv", index=False)
    if predictive_rows:
        pd.concat(predictive_rows, ignore_index=True).to_csv(out_dir / "predictive_model_comparison_by_subset.csv", index=False)


def balanced_bar_plot(df: pd.DataFrame, title: str, subtitle: str, png_path: Path) -> None:
    work = df.sort_values("mean_balanced_shift")
    vals = work["mean_balanced_shift"].to_numpy() * 100
    low = work["ci95_low"].to_numpy() * 100
    high = work["ci95_high"].to_numpy() * 100
    y = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.5 * len(work))))
    ax.barh(y, vals, color=np.where(vals >= 0, "#2E8B57", "#B22222"))
    ax.errorbar(vals, y, xerr=[vals - low, high - vals], fmt="none", ecolor="black", capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}\nq={nq}" for lab, nq in zip(work["label"], work["n_questions_with_label"])])
    ax.set_xlabel("Suppressed - Forgotten (percentage points)")
    ax.set_title(title)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def heatmap(df: pd.DataFrame, title: str, subtitle: str, png_path: Path, min_cell_n: int) -> None:
    mat = df.pivot(index="method", columns="label", values="method_centered_effect").fillna(0)
    nmat = df.pivot(index="method", columns="label", values="n_unique_questions_with_label").fillna(0)
    arr = mat.to_numpy(dtype=float)
    mask = nmat.to_numpy() < min_cell_n
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
    ax.text(0, 1.02, "Inferred multi-select labels; primary stage analysis excludes sterile stems.", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_figures_outputs(config: StemOnlyConfig) -> None:
    fig_dir = ensure_dir(config.output_dir / "figures")
    overall = pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_multilabel_stage_shift.csv")
    intrinsic = pd.read_csv(config.output_dir / "intrinsic" / "method_centered_multilabel_stage_effects.csv")
    predictive = pd.read_csv(config.output_dir / "predictive" / "model_comparison_repeated_grouped_cv.csv")
    subtitle = "Labels are inferred and multi-select; percentages do not sum to 100%.\nPositive = relatively more suppressed; negative = relatively more forgotten."
    balanced_bar_plot(overall, "Stem-only balanced multilabel stage shift", subtitle, fig_dir / "stem_only_balanced_multilabel_stage_shift.png")
    heatmap(intrinsic, "Stem-only method-centered stage heatmap", subtitle, fig_dir / "stem_only_method_centered_stage_heatmap.png", config.min_cell_n)
    predictive_ci_plot(predictive, fig_dir / "predictive_model_comparison_with_ci.png")


VALIDATION_SCRIPT = r'''from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
merged = pd.read_csv(ROOT / "data" / "eligible_question_method_rows.csv")
assert merged["question_id"].notna().all()
assert merged.duplicated(["question_id", "method"]).sum() == 0
assert "is_sterile_stem_question" in merged.columns
assert "is_cannot_determine" in merged.columns
assert pd.read_csv(ROOT / "balanced_bootstrap" / "overall_balanced_multilabel_stage_shift.csv")["bootstrap_repetitions"].ge(5000).all()
assert (ROOT / "figures" / "stem_only_balanced_multilabel_stage_shift.png").exists()
assert (ROOT / "figures" / "stem_only_method_centered_stage_heatmap.png").exists()
print("Validation checks passed.")
'''


def write_validation_script(config: StemOnlyConfig) -> Path:
    val_dir = ensure_dir(config.output_dir / "validation")
    path = val_dir / "run_stem_only_validation_checks.py"
    write_text(path, VALIDATION_SCRIPT)
    return path


def run_validation(config: StemOnlyConfig) -> None:
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


def generate_report(config: StemOnlyConfig) -> None:
    audit = json.loads((config.output_dir / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
    overall = pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_multilabel_stage_shift.csv")
    predictive = pd.read_csv(config.output_dir / "predictive" / "model_comparison_repeated_grouped_cv.csv")
    sens = pd.read_csv(config.output_dir / "sensitivity" / "subset_counts.csv")
    lines = [
        "# Stem-only multilabel report",
        "",
        "Labels are inferred and multi-select. Primary stage analysis excludes sterile stems. `is_sterile_stem_question` is not a threat-chain stage.",
        "",
        "## Audit",
        "",
        df_to_markdown_table(pd.DataFrame([audit])),
        "",
        "## Balanced stage shifts",
        "",
        df_to_markdown_table(overall),
        "",
        "## Predictive models",
        "",
        df_to_markdown_table(predictive),
        "",
        "## Sensitivity subsets",
        "",
        df_to_markdown_table(sens),
        "",
    ]
    write_text(config.output_dir / "REPORT.md", "\n".join(lines))


def run_all(config: StemOnlyConfig) -> None:
    save_command_environment(config, sys.argv)
    append_run_log(config, "Starting stem-only multilabel pipeline.")
    merged = build_merged(config)
    save_eligible_rows(config, merged)
    append_run_log(config, f"Eligible rows prepared: {len(merged)} rows.")
    run_audit_outputs(config, merged)
    append_run_log(config, "Audit complete.")
    run_balanced_outputs(config, merged)
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
