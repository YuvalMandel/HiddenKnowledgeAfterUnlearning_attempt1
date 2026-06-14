from __future__ import annotations

import json
import math
import os
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
DEFAULT_PAIR_DATASET = ROOT / "plots" / "pair_level_suppressed_forgotten" / "pair_level_dataset.csv"
DEFAULT_LABELS_INPUT = ROOT / "out_threat_model_judge_v1" / "labels" / "wmdp_bio_threat_model_labels_inferred_chatgpt_assistant_v1.csv"
DEFAULT_TOPIC_INPUT = ROOT / "LLM as as Judge" / "wmdp_bio_inferred_categories_first_pass.csv"
EXPECTED_METHODS = {
    "ELM",
    "GradDiff",
    "PB_J",
    "RMU",
    "RMU-LAT",
    "RR",
    "RepNoise",
    "TAR",
}
RMU_ALIASES = {"rmu", "RMU", "rmu_base", "rmu-standard"}


@dataclass
class V2Config:
    pair_dataset: Path
    labels_input: Path
    topic_input: Path
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
                if math.isnan(value):
                    text = "nan"
                else:
                    text = f"{value:.6g}"
            else:
                text = str(value)
            vals.append(text.replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_command_environment(config: V2Config, argv: list[str]) -> None:
    ensure_dir(config.output_dir)
    write_text(config.output_dir / "command.txt", " ".join(argv))
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        write_text(config.output_dir / "environment.txt", proc.stdout)
    except Exception as exc:
        write_text(config.output_dir / "environment.txt", f"Could not capture environment: {exc}\n")
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        write_text(config.output_dir / "git_commit.txt", proc.stdout.strip() + "\n")
    except Exception as exc:
        write_text(config.output_dir / "git_commit.txt", f"Could not capture git commit: {exc}\n")


def append_run_log(config: V2Config, message: str) -> None:
    ensure_dir(config.output_dir)
    with (config.output_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_utc()}] {message}\n")


def load_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_pair_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_topic_categories(path: Path) -> pd.DataFrame:
    topic = pd.read_csv(path)
    return topic[["question_id", "category_first_pass"]].rename(columns={"category_first_pass": "topic_category_inferred"})


def question_level_from_pair(pair: pd.DataFrame) -> pd.DataFrame:
    cols = ["method", "checkpoint", "question_idx", "subset_ck8"]
    available = [col for col in cols if col in pair.columns]
    q = pair.loc[pair["checkpoint"].eq("ck8"), available].drop_duplicates(["method", "question_idx"]).copy()
    q = q.rename(columns={"question_idx": "question_id", "subset_ck8": "outcome"})
    return q


def build_merged_table(config: V2Config) -> pd.DataFrame:
    pair = load_pair_dataset(config.pair_dataset)
    labels = load_labels(config.labels_input)
    topics = load_topic_categories(config.topic_input)
    question_level = question_level_from_pair(pair)
    keep = [
        "question_id",
        "question_hash",
        "question",
        "choices",
        "risk_mechanism_final",
        "risk_stage_final",
        "risk_stage_analysis_final",
        "proxy_relation_final",
        "needs_review",
        "risk_mechanism_confidence",
        "risk_stage_confidence",
        "proxy_relation_confidence",
        "rationale_short",
        "judge_model",
        "judge_prompt_version",
    ]
    merged = question_level.merge(labels[keep], on="question_id", how="left")
    merged = merged.merge(topics, on="question_id", how="left")
    merged["method"] = merged["method"].astype(str)
    merged["outcome"] = merged["outcome"].astype(str)
    return merged


def eligible_rows(merged: pd.DataFrame) -> pd.DataFrame:
    eligible = merged.loc[merged["outcome"].isin(["suppressed", "forgotten"])].copy()
    eligible = eligible.reset_index(drop=True)
    eligible["analysis_row_id"] = np.arange(len(eligible))
    eligible["is_needs_review"] = eligible["needs_review"].fillna(False).astype(bool)
    eligible["is_low_confidence_label"] = (
        (eligible["risk_stage_confidence"].fillna(0) < 0.60)
        | (eligible["proxy_relation_confidence"].fillna(0) < 0.60)
        | (eligible["risk_mechanism_confidence"].fillna(0) < 0.60)
    )
    eligible["suppressed"] = eligible["outcome"].eq("suppressed").astype(int)
    return eligible


def audit_method_presence(merged: pd.DataFrame) -> dict[str, Any]:
    present = set(merged["method"].dropna().unique())
    alias_hits = sorted([method for method in present if method in RMU_ALIASES or method.lower() in {alias.lower() for alias in RMU_ALIASES}])
    missing = sorted(EXPECTED_METHODS - present)
    rmu_present = "RMU" in present
    note = None
    if not rmu_present:
        if alias_hits:
            note = f"Plain RMU not present by exact name, but alias candidates exist: {alias_hits}"
        else:
            note = "Plain RMU is unavailable in the current eligible suppressed-vs-forgotten subset."
    return {
        "expected_methods": sorted(EXPECTED_METHODS),
        "present_methods": sorted(present),
        "missing_methods": missing,
        "rmu_present": rmu_present,
        "rmu_alias_hits": alias_hits,
        "note": note,
    }


def ideation_audit(labels: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    total_questions = labels.loc[labels["risk_stage_final"].eq("ideation"), "question_id"].nunique()
    present_questions = eligible.loc[eligible["risk_stage_final"].eq("ideation"), "question_id"].nunique()
    rows = []
    if present_questions:
        grouped = eligible.loc[eligible["risk_stage_final"].eq("ideation")].groupby(["method", "outcome"]).size()
        for (method, outcome), count in grouped.items():
            rows.append({
                "method": method,
                "outcome": outcome,
                "rows": int(count),
            })
    summary = pd.DataFrame(
        [
            {
                "total_ideation_labeled_questions": int(total_questions),
                "ideation_labeled_questions_present_in_merged_dataset": int(present_questions),
                "ideation_rows_eligible_for_analysis": int(len(eligible.loc[eligible["risk_stage_final"].eq("ideation")])),
                "ideation_suppressed_count": int(
                    eligible.loc[eligible["risk_stage_final"].eq("ideation") & eligible["outcome"].eq("suppressed")].shape[0]
                ),
                "ideation_forgotten_count": int(
                    eligible.loc[eligible["risk_stage_final"].eq("ideation") & eligible["outcome"].eq("forgotten")].shape[0]
                ),
            }
        ]
    )
    if rows:
        detail = pd.DataFrame(rows)
        summary = pd.concat([summary, detail], axis=0, ignore_index=True, sort=False)
    return summary


def write_audit_outputs(config: V2Config, merged: pd.DataFrame, eligible: pd.DataFrame) -> None:
    audit_dir = ensure_dir(config.output_dir / "audit")
    labels = load_labels(config.labels_input)
    method_info = audit_method_presence(eligible)
    label_counts = []
    for axis, column in [
        ("risk_mechanism_final", "risk_mechanism_final"),
        ("risk_stage_final", "risk_stage_final"),
        ("risk_stage_analysis_final", "risk_stage_analysis_final"),
        ("proxy_relation_final", "proxy_relation_final"),
        ("topic_category_inferred", "topic_category_inferred"),
    ]:
        counts = eligible.groupby(column).agg(n_rows=("question_id", "size"), n_unique_questions=("question_id", "nunique")).reset_index()
        counts.insert(0, "axis", axis)
        counts = counts.rename(columns={column: "label"})
        label_counts.append(counts)
    label_counts_df = pd.concat(label_counts, ignore_index=True)
    label_counts_df.to_csv(audit_dir / "label_counts.csv", index=False)

    method_counts = eligible.groupby("method").agg(
        rows=("question_id", "size"),
        unique_questions=("question_id", "nunique"),
        suppressed_count=("suppressed", "sum"),
    ).reset_index()
    method_counts["forgotten_count"] = method_counts["rows"] - method_counts["suppressed_count"]
    method_counts.to_csv(audit_dir / "method_counts.csv", index=False)

    question_coverage = eligible.groupby("question_id").agg(
        n_methods=("method", "nunique"),
        methods=("method", lambda s: "|".join(sorted(set(map(str, s))))),
        outcomes=("outcome", lambda s: "|".join(sorted(set(map(str, s))))),
    ).reset_index()
    question_coverage.to_csv(audit_dir / "question_method_coverage.csv", index=False)

    missing = merged.isna().sum().rename_axis("column").reset_index(name="missing_count")
    missing.to_csv(audit_dir / "missing_values_by_column.csv", index=False)

    ideation_df = ideation_audit(labels, eligible)
    ideation_df.to_csv(audit_dir / "ideation_stage_audit.csv", index=False)

    summary = {
        "total_rows": int(len(eligible)),
        "unique_question_ids": int(eligible["question_id"].nunique()),
        "unique_methods": int(eligible["method"].nunique()),
        "rows_per_method": {str(k): int(v) for k, v in eligible["method"].value_counts().sort_index().items()},
        "unique_questions_per_method": {str(k): int(v) for k, v in eligible.groupby("method")["question_id"].nunique().sort_index().items()},
        "suppressed_count": int(eligible["outcome"].eq("suppressed").sum()),
        "forgotten_count": int(eligible["outcome"].eq("forgotten").sum()),
        "suppressed_rate": float(eligible["outcome"].eq("suppressed").mean()),
        "duplicate_question_method_rows": int(eligible.duplicated(["question_id", "method"]).sum()),
        "missing_values_per_column": {str(k): int(v) for k, v in merged.isna().sum().items()},
        "method_audit": method_info,
        "review_queue_rate": float(eligible["is_needs_review"].mean()),
        "low_confidence_rate": float(eligible["is_low_confidence_label"].mean()),
        "ideation_present_in_label_file": int(labels["risk_stage_final"].eq("ideation").sum()),
        "ideation_eligible_rows": int(eligible["risk_stage_final"].eq("ideation").sum()),
    }
    save_json(summary, audit_dir / "dataset_audit_summary.json")

    md_lines = [
        "# Dataset Audit Summary",
        "",
        "This audit covers the eligible suppressed-vs-forgotten ck8 question-method rows used in the v2 analysis.",
        "",
        "## Summary",
        "",
        df_to_markdown_table(pd.DataFrame([summary]).drop(columns=["missing_values_per_column", "method_audit"])),
        "",
        "## Method Audit",
        "",
        df_to_markdown_table(pd.DataFrame([method_info])),
        "",
        "## Method Counts",
        "",
        df_to_markdown_table(method_counts),
        "",
        "## Label Counts",
        "",
        df_to_markdown_table(label_counts_df),
        "",
        "## Ideation Audit",
        "",
        df_to_markdown_table(ideation_df.fillna("")),
        "",
    ]
    write_text(audit_dir / "dataset_audit_summary.md", "\n".join(md_lines))


def prepare_eligible_outputs(config: V2Config) -> pd.DataFrame:
    merged = build_merged_table(config)
    eligible = eligible_rows(merged)
    data_dir = ensure_dir(config.output_dir / "data")
    eligible.to_csv(data_dir / "eligible_question_method_rows.csv", index=False)
    write_audit_outputs(config, merged, eligible)
    return eligible


def sample_group_ids(rng: np.random.Generator, question_ids: np.ndarray) -> np.ndarray:
    return rng.choice(question_ids, size=len(question_ids), replace=True)


def build_sampled_rows(df: pd.DataFrame, sampled_ids: np.ndarray) -> pd.DataFrame:
    groups = {qid: grp for qid, grp in df.groupby("question_id", sort=False)}
    parts = [groups[qid] for qid in sampled_ids if qid in groups]
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def build_group_index_cache(df: pd.DataFrame) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    row_indices = np.arange(len(df), dtype=int)
    groups = {qid: row_indices[df["question_id"].to_numpy() == qid] for qid in df["question_id"].dropna().unique()}
    qids = np.array(list(groups.keys()))
    return qids, groups


def sampled_index_array(sampled_ids: np.ndarray, group_map: dict[Any, np.ndarray]) -> np.ndarray:
    parts = [group_map[qid] for qid in sampled_ids if qid in group_map]
    if not parts:
        return np.array([], dtype=int)
    return np.concatenate(parts)


def sample_multivariate_hypergeometric(rng: np.random.Generator, counts: np.ndarray, n: int) -> np.ndarray:
    counts = counts.astype(int, copy=False)
    if n <= 0 or counts.sum() == 0:
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
        if remaining_draws <= 0 or remaining_total <= 0:
            break
        draw = rng.hypergeometric(ngood=count, nbad=remaining_total - count, nsample=remaining_draws)
        sampled[idx] = draw
        remaining_draws -= int(draw)
        remaining_total -= count
    sampled[-1] = remaining_draws
    return sampled


def aggregated_count_matrices(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    labels = sorted(df[label_col].fillna("missing").astype(str).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    qids = np.array(sorted(df["question_id"].dropna().unique().tolist()))
    supp = np.zeros((len(qids), len(labels)), dtype=int)
    forg = np.zeros((len(qids), len(labels)), dtype=int)
    qid_to_idx = {qid: idx for idx, qid in enumerate(qids)}
    for _, row in df.iterrows():
        qi = qid_to_idx[row["question_id"]]
        li = label_to_idx[str(row[label_col]) if pd.notna(row[label_col]) else "missing"]
        if row["outcome"] == "suppressed":
            supp[qi, li] += 1
        elif row["outcome"] == "forgotten":
            forg[qi, li] += 1
    return qids, supp, forg, labels


def clustered_balanced_shift(
    df: pd.DataFrame,
    label_col: str,
    axis_name: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    qids, supp_matrix, forg_matrix, labels = aggregated_count_matrices(df, label_col)
    draws = {label: [] for label in labels}
    for _ in range(n_boot):
        sampled_qids = sample_group_ids(rng, qids)
        sampled_pos = np.searchsorted(qids, sampled_qids)
        supp_counts = supp_matrix[sampled_pos].sum(axis=0)
        forg_counts = forg_matrix[sampled_pos].sum(axis=0)
        ns = int(supp_counts.sum())
        nf = int(forg_counts.sum())
        if ns == 0 or nf == 0:
            continue
        n = min(ns, nf)
        sampled_supp = sample_multivariate_hypergeometric(rng, supp_counts, n)
        sampled_forg = sample_multivariate_hypergeometric(rng, forg_counts, n)
        for idx, label in enumerate(labels):
            ps = sampled_supp[idx] / n
            pf = sampled_forg[idx] / n
            draws[label].append(ps - pf)
    rows = []
    for label in labels:
        arr = np.asarray(draws[label], dtype=float)
        rows.append(
            {
                "axis": axis_name,
                "label": label,
                "n_rows_total": int(len(df)),
                "n_unique_questions": int(df["question_id"].nunique()),
                "n_suppressed_rows": int(df["outcome"].eq("suppressed").sum()),
                "n_forgotten_rows": int(df["outcome"].eq("forgotten").sum()),
                "mean_balanced_shift": float(np.mean(arr)) if len(arr) else float("nan"),
                "median_balanced_shift": float(np.median(arr)) if len(arr) else float("nan"),
                "ci95_low": float(np.quantile(arr, 0.025)) if len(arr) else float("nan"),
                "ci95_high": float(np.quantile(arr, 0.975)) if len(arr) else float("nan"),
                "sign_probability_positive": float((arr > 0).mean()) if len(arr) else float("nan"),
                "sign_probability_negative": float((arr < 0).mean()) if len(arr) else float("nan"),
                "bootstrap_repetitions": int(n_boot),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows)


def run_balanced_bootstrap_outputs(config: V2Config, eligible: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out_dir = ensure_dir(config.output_dir / "balanced_bootstrap")
    axes = {
        "risk_mechanism": "risk_mechanism_final",
        "risk_stage": "risk_stage_analysis_final",
        "proxy_relation": "proxy_relation_final",
        "topic_category": "topic_category_inferred",
    }
    results: dict[str, pd.DataFrame] = {}
    for idx, (axis_name, column) in enumerate(axes.items()):
        overall = clustered_balanced_shift(eligible, column, axis_name, config.bootstrap_repetitions, config.seed)
        overall.to_csv(out_dir / f"overall_balanced_shift_{axis_name}.csv", index=False)
        parts = []
        for method, group in eligible.groupby("method", sort=True):
            table = clustered_balanced_shift(group, column, axis_name, config.bootstrap_repetitions, config.seed)
            table.insert(0, "method", method)
            parts.append(table)
        by_method = pd.concat(parts, ignore_index=True)
        by_method.to_csv(out_dir / f"by_method_balanced_shift_{axis_name}.csv", index=False)
        results[f"overall_{axis_name}"] = overall
        results[f"by_method_{axis_name}"] = by_method
    return results


def grouped_rates(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    grouped = df.groupby(["method", label_col]).agg(
        n_rows=("question_id", "size"),
        n_unique_questions=("question_id", "nunique"),
        suppressed_count=("suppressed", "sum"),
    ).reset_index()
    grouped["forgotten_count"] = grouped["n_rows"] - grouped["suppressed_count"]
    grouped["suppressed_rate_for_label"] = grouped["suppressed_count"] / grouped["n_rows"]
    base = df.groupby("method")["suppressed"].mean().rename("suppressed_rate_for_method").reset_index()
    grouped = grouped.merge(base, on="method", how="left")
    grouped["method_centered_effect"] = grouped["suppressed_rate_for_label"] - grouped["suppressed_rate_for_method"]
    grouped = grouped.rename(columns={label_col: "label"})
    return grouped


def bootstrap_method_centered_ci(group: pd.DataFrame, label: str, label_col: str, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    qids, supp_matrix, forg_matrix, labels = aggregated_count_matrices(group, label_col)
    label_idx = {lab: idx for idx, lab in enumerate(labels)}
    if label not in label_idx:
        return float("nan"), float("nan")
    li = label_idx[label]
    vals = []
    for _ in range(n_boot):
        sampled_qids = sample_group_ids(rng, qids)
        sampled_pos = np.searchsorted(qids, sampled_qids)
        supp_counts = supp_matrix[sampled_pos].sum(axis=0)
        forg_counts = forg_matrix[sampled_pos].sum(axis=0)
        total_rows = int(supp_counts.sum() + forg_counts.sum())
        if total_rows == 0:
            continue
        base = float(supp_counts.sum() / total_rows)
        label_total = int(supp_counts[li] + forg_counts[li])
        if label_total == 0:
            continue
        label_rate = float(supp_counts[li] / label_total)
        vals.append(label_rate - base)
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def run_intrinsic_outputs(config: V2Config, eligible: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out_dir = ensure_dir(config.output_dir / "intrinsic")
    axes = {
        "risk_mechanism": "risk_mechanism_final",
        "risk_stage": "risk_stage_analysis_final",
        "proxy_relation": "proxy_relation_final",
        "topic_category": "topic_category_inferred",
    }
    results = {}
    for axis_name, column in axes.items():
        table = grouped_rates(eligible, column)
        table.insert(1, "axis", axis_name)
        ci_low = []
        ci_high = []
        for _, row in table.iterrows():
            group = eligible.loc[eligible["method"].eq(row["method"])].copy()
            low, high = bootstrap_method_centered_ci(group, str(row["label"]), column, config.bootstrap_repetitions, config.seed)
            ci_low.append(low)
            ci_high.append(high)
        table["ci95_low"] = ci_low
        table["ci95_high"] = ci_high
        table.to_csv(out_dir / f"method_centered_effect_{axis_name}.csv", index=False)
        results[axis_name] = table
    return results


def repeated_group_splits(question_ids: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    unique_q = np.array(sorted(set(question_ids.tolist())))
    rng = np.random.default_rng(seed)
    splits = []
    for repeat in range(n_repeats):
        shuffled = unique_q.copy()
        rng.shuffle(shuffled)
        fold_groups = np.array_split(shuffled, n_splits)
        for fold_idx, test_groups in enumerate(fold_groups):
            test_mask = np.isin(question_ids, test_groups)
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            splits.append((repeat, train_idx, test_idx))
    return splits


def build_model_specs(df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    work = df.copy()
    work["method_x_proxy"] = work["method"].astype(str) + "__" + work["proxy_relation_final"].fillna("missing").astype(str)
    work["method_x_stage"] = work["method"].astype(str) + "__" + work["risk_stage_analysis_final"].fillna("missing").astype(str)
    return [
        ("Model A", ["method"]),
        ("Model B", ["method", "topic_category_inferred"]),
        ("Model C", ["method", "risk_stage_analysis_final"]),
        ("Model D", ["method", "risk_mechanism_final"]),
        ("Model E", ["method", "proxy_relation_final"]),
        ("Model F", ["method", "topic_category_inferred", "risk_stage_analysis_final", "risk_mechanism_final", "proxy_relation_final"]),
        ("Model G", ["method", "topic_category_inferred", "risk_stage_analysis_final", "proxy_relation_final", "method_x_proxy", "method_x_stage"]),
    ]


def make_model_input(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["topic_category_inferred", "risk_stage_analysis_final", "risk_mechanism_final", "proxy_relation_final"]:
        work[col] = work[col].fillna("missing").astype(str)
    work["method"] = work["method"].fillna("missing").astype(str)
    work["method_x_proxy"] = work["method"] + "__" + work["proxy_relation_final"]
    work["method_x_stage"] = work["method"] + "__" + work["risk_stage_analysis_final"]
    return work


def build_pipeline(features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
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
    return Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
        ]
    )


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


def run_predictive_outputs(config: V2Config, eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = ensure_dir(config.output_dir / "predictive")
    work = make_model_input(eligible)
    y = work["suppressed"].to_numpy()
    qids = work["question_id"].to_numpy()
    n_splits = min(5, len(np.unique(qids)))
    specs = build_model_specs(work)
    split_list = repeated_group_splits(qids, n_splits, config.cv_repeats, config.seed)
    metric_rows = []
    oof_rows = []
    calibration_rows = []

    for model_name, features in specs:
        for repeat in range(config.cv_repeats):
            repeat_mask = [item for item in split_list if item[0] == repeat]
            oof_prob = np.zeros(len(work), dtype=float)
            for _, train_idx, test_idx in repeat_mask:
                train_groups = set(qids[train_idx].tolist())
                test_groups = set(qids[test_idx].tolist())
                if train_groups & test_groups:
                    raise AssertionError("question_id overlap detected between train and test folds")
                pipe = build_pipeline(features)
                pipe.fit(work.iloc[train_idx][features], y[train_idx])
                oof_prob[test_idx] = pipe.predict_proba(work.iloc[test_idx][features])[:, 1]
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
            bins = pd.cut(oof_prob, bins=np.linspace(0, 1, 11), include_lowest=True)
            cal = pd.DataFrame({"bin": bins.astype(str), "y": y, "p": oof_prob}).groupby("bin").agg(
                n_rows=("y", "size"),
                observed_rate=("y", "mean"),
                predicted_mean=("p", "mean"),
            ).reset_index()
            cal["model"] = model_name
            cal["repeat"] = repeat
            calibration_rows.append(cal)

    metrics_df = pd.DataFrame(metric_rows)
    oof_df = pd.DataFrame(oof_rows)
    calibration_df = pd.concat(calibration_rows, ignore_index=True)

    summary_rows = []
    model_a = metrics_df.loc[metrics_df["model"].eq("Model A")].reset_index(drop=True)
    for model_name, group in metrics_df.groupby("model", sort=False):
        row = {"model": model_name}
        for metric in ["balanced_accuracy", "macro_f1", "roc_auc", "average_precision", "log_loss", "brier_score"]:
            stats = summarize_metric_series(group[metric])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        if model_name != "Model A":
            aligned = group.sort_values("repeat").reset_index(drop=True)
            base = model_a.sort_values("repeat").reset_index(drop=True)
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
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "model_comparison_repeated_grouped_cv.csv", index=False)

    vs_a_cols = ["model"] + [col for col in summary_df.columns if col.startswith("delta_")]
    summary_df[vs_a_cols].to_csv(out_dir / "model_comparison_vs_method_only.csv", index=False)
    oof_df.to_csv(out_dir / "oof_predictions_all_models.csv", index=False)

    cal_summary = calibration_df.groupby(["model", "bin"]).agg(
        n_rows_mean=("n_rows", "mean"),
        observed_rate_mean=("observed_rate", "mean"),
        predicted_mean_mean=("predicted_mean", "mean"),
    ).reset_index()
    cal_summary.to_csv(out_dir / "calibration_summary.csv", index=False)
    return summary_df, oof_df, cal_summary


def sensitivity_subset(df: pd.DataFrame, subset_name: str, axis_name: str | None = None) -> pd.DataFrame:
    if subset_name == "S0_all":
        return df.copy()
    if subset_name == "S1_no_review":
        return df.loc[~df["is_needs_review"]].copy()
    if subset_name == "S2_high_confidence":
        return df.loc[
            (df["risk_stage_confidence"] >= 0.70)
            & (df["proxy_relation_confidence"] >= 0.70)
            & (df["risk_mechanism_confidence"] >= 0.70)
        ].copy()
    if subset_name == "S3_collapse_uncertain":
        out = df.copy()
        if axis_name == "risk_stage":
            out = out.loc[out["risk_stage_analysis_final"].ne("background_or_cross_cutting")].copy()
        if axis_name == "proxy_relation":
            out = out.loc[out["proxy_relation_final"].ne("unclear")].copy()
        return out
    raise ValueError(subset_name)


def run_sensitivity_outputs(config: V2Config, eligible: pd.DataFrame) -> None:
    out_dir = ensure_dir(config.output_dir / "sensitivity")
    axes = {
        "risk_mechanism": "risk_mechanism_final",
        "risk_stage": "risk_stage_analysis_final",
        "proxy_relation": "proxy_relation_final",
        "topic_category": "topic_category_inferred",
    }
    subset_counts = []
    balanced_rows = []
    predictive_rows = []
    for subset_name in ["S0_all", "S1_no_review", "S2_high_confidence", "S3_collapse_uncertain"]:
        for axis_name, column in axes.items():
            sub = sensitivity_subset(eligible, subset_name, axis_name)
            subset_counts.append(
                {
                    "subset": subset_name,
                    "axis": axis_name,
                    "n_rows": int(len(sub)),
                    "n_unique_questions": int(sub["question_id"].nunique()),
                }
            )
            if sub.empty:
                continue
            table = clustered_balanced_shift(sub, column, axis_name, config.bootstrap_repetitions, config.seed)
            table.insert(0, "subset", subset_name)
            balanced_rows.append(table)
        sub_pred = sensitivity_subset(eligible, subset_name)
        if not sub_pred.empty and sub_pred["question_id"].nunique() >= 5:
            summary_df, _, _ = run_predictive_outputs(
                V2Config(
                    pair_dataset=config.pair_dataset,
                    labels_input=config.labels_input,
                    topic_input=config.topic_input,
                    output_dir=ensure_dir(out_dir / subset_name),
                    bootstrap_repetitions=config.bootstrap_repetitions,
                    cv_repeats=config.cv_repeats,
                    min_cell_n=config.min_cell_n,
                    seed=config.seed,
                ),
                sub_pred,
            )
            summary_df.insert(0, "subset", subset_name)
            predictive_rows.append(summary_df)
    pd.DataFrame(subset_counts).to_csv(out_dir / "sensitivity_subset_counts.csv", index=False)
    if balanced_rows:
        pd.concat(balanced_rows, ignore_index=True).to_csv(out_dir / "sensitivity_balanced_shift_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "sensitivity_balanced_shift_summary.csv", index=False)
    if predictive_rows:
        pd.concat(predictive_rows, ignore_index=True).to_csv(out_dir / "sensitivity_predictive_model_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "sensitivity_predictive_model_summary.csv", index=False)


def run_unclear_proxy_audit(config: V2Config, eligible: pd.DataFrame) -> None:
    audit_dir = ensure_dir(config.output_dir / "audit")
    unclear = eligible.loc[eligible["proxy_relation_final"].eq("unclear")].copy()
    cols = [
        "question_id",
        "question",
        "method",
        "outcome",
        "proxy_relation_final",
        "proxy_relation_confidence",
        "risk_stage_final",
        "risk_stage_confidence",
        "risk_mechanism_final",
        "needs_review",
        "rationale_short",
    ]
    unclear[cols].to_csv(audit_dir / "unclear_proxy_audit.csv", index=False)
    rng = np.random.default_rng(config.seed)
    sample_n = min(20, len(unclear))
    sample = unclear.sample(n=sample_n, random_state=int(rng.integers(0, 1_000_000_000))) if sample_n else unclear
    md = [
        "# Unclear Proxy Audit",
        "",
        f"- Count of unclear proxy rows: {len(unclear)}",
        f"- Unique questions: {unclear['question_id'].nunique()}",
        f"- Suppressed rows: {int(unclear['outcome'].eq('suppressed').sum())}",
        f"- Forgotten rows: {int(unclear['outcome'].eq('forgotten').sum())}",
        "",
        "## Distribution by method",
        "",
        df_to_markdown_table(unclear["method"].value_counts().rename_axis("method").reset_index(name="count")),
        "",
        "## Distribution by risk stage",
        "",
        df_to_markdown_table(unclear["risk_stage_final"].value_counts().rename_axis("risk_stage_final").reset_index(name="count")),
        "",
        "## Distribution by needs_review",
        "",
        df_to_markdown_table(unclear["needs_review"].value_counts().rename_axis("needs_review").reset_index(name="count")),
        "",
        "## Proxy confidence summary",
        "",
        df_to_markdown_table(
            pd.DataFrame(
                [
                    {
                        "mean": unclear["proxy_relation_confidence"].mean(),
                        "median": unclear["proxy_relation_confidence"].median(),
                        "p10": unclear["proxy_relation_confidence"].quantile(0.10),
                        "p90": unclear["proxy_relation_confidence"].quantile(0.90),
                    }
                ]
            )
        ),
        "",
        "## Random sample of questions",
        "",
        df_to_markdown_table(sample[cols].fillna("")),
        "",
        "## Conclusion",
        "",
        "The unclear-proxy signal should be treated as a mixture of potential substantive structure and annotation-confidence artifact until checked against confidence, review flags, and stage composition.",
        "",
    ]
    write_text(audit_dir / "unclear_proxy_examples.md", "\n".join(md))


def save_dual_format(fig: plt.Figure, png_path: Path) -> None:
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


def overall_bar_plot(df: pd.DataFrame, title: str, subtitle: str, png_path: Path) -> None:
    if df.empty:
        return
    work = df.sort_values("mean_balanced_shift")
    vals = work["mean_balanced_shift"].to_numpy() * 100
    low = work["ci95_low"].to_numpy() * 100
    high = work["ci95_high"].to_numpy() * 100
    labels = [f"{lab}\n(q={nq})" for lab, nq in zip(work["label"], work["n_unique_questions"])]
    y = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(work))))
    colors = np.where(vals >= 0, "#2E8B57", "#B22222")
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.errorbar(vals, y, xerr=[vals - low, high - vals], fmt="none", ecolor="black", capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Suppressed - Forgotten (percentage points)")
    ax.set_title(title)
    ax.text(0, 1.02, subtitle + "\npositive = relatively more suppressed; negative = relatively more forgotten", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_dual_format(fig, png_path)
    plt.close(fig)


def masked_heatmap(df: pd.DataFrame, value_col: str, title: str, subtitle: str, png_path: Path, min_cell_n: int, mask_low_n: bool) -> None:
    if df.empty:
        return
    mat = df.pivot(index="method", columns="label", values=value_col).fillna(0)
    nmat = df.pivot(index="method", columns="label", values="n_unique_questions").fillna(0)
    vmax = max(float(np.abs(mat.to_numpy()).max()), 1e-9)
    fig, ax = plt.subplots(figsize=(max(8, 0.95 * mat.shape[1]), 4.8))
    display = mat.to_numpy(copy=True)
    if mask_low_n:
        display = display.copy()
        display[nmat.to_numpy() < min_cell_n] = np.nan
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#d9d9d9")
    im = ax.imshow(display, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([f"{col}\nq={int(nmat[col].sum())}" for col in mat.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    ax.text(0, 1.02, subtitle + "\npositive = relatively more suppressed; negative = relatively more forgotten", transform=ax.transAxes, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            qn = int(nmat.iloc[i, j])
            if mask_low_n and qn < min_cell_n:
                ax.text(j, i, "n<10", ha="center", va="center", fontsize=8, color="black")
            else:
                ax.text(j, i, f"{mat.iloc[i, j]*100:+.1f}\nq={qn}", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Percentage-point effect")
    fig.tight_layout()
    save_dual_format(fig, png_path)
    plt.close(fig)


def predictive_ci_plot(summary_df: pd.DataFrame, png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    order = summary_df["model"].tolist()
    mean = summary_df["balanced_accuracy_mean"].to_numpy()
    low = summary_df["balanced_accuracy_ci95_low"].to_numpy()
    high = summary_df["balanced_accuracy_ci95_high"].to_numpy()
    x = np.arange(len(order))
    ax.errorbar(x, mean, yerr=[mean - low, high - mean], fmt="o", color="#1f4e79", ecolor="#7aa6d1", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Predictive model comparison with repeated grouped-CV uncertainty")
    ax.text(0, 1.02, "Inferred labels; exploratory analysis", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_dual_format(fig, png_path)
    plt.close(fig)


def calibration_plot(calibration_df: pd.DataFrame, png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name in sorted(calibration_df["model"].unique()):
        sub = calibration_df.loc[calibration_df["model"].eq(model_name)].copy()
        ax.plot(sub["predicted_mean_mean"], sub["observed_rate_mean"], marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Predictive model calibration")
    ax.text(0, 1.02, "Inferred labels; exploratory analysis", transform=ax.transAxes, fontsize=9)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_dual_format(fig, png_path)
    plt.close(fig)


def run_figures_outputs(config: V2Config) -> None:
    main_dir = ensure_dir(config.output_dir / "figures" / "main")
    supp_dir = ensure_dir(config.output_dir / "figures" / "supplementary")
    boot_dir = config.output_dir / "balanced_bootstrap"
    intrinsic_dir = config.output_dir / "intrinsic"
    predictive_dir = config.output_dir / "predictive"

    overall_proxy = pd.read_csv(boot_dir / "overall_balanced_shift_proxy_relation.csv")
    overall_stage = pd.read_csv(boot_dir / "overall_balanced_shift_risk_stage.csv")
    overall_mech = pd.read_csv(boot_dir / "overall_balanced_shift_risk_mechanism.csv")
    overall_topic = pd.read_csv(boot_dir / "overall_balanced_shift_topic_category.csv")
    by_method_proxy = pd.read_csv(intrinsic_dir / "method_centered_effect_proxy_relation.csv")
    by_method_stage = pd.read_csv(intrinsic_dir / "method_centered_effect_risk_stage.csv")
    by_method_mech = pd.read_csv(intrinsic_dir / "method_centered_effect_risk_mechanism.csv")
    by_method_topic = pd.read_csv(intrinsic_dir / "method_centered_effect_topic_category.csv")
    pred_summary = pd.read_csv(predictive_dir / "model_comparison_repeated_grouped_cv.csv")
    calibration_df = pd.read_csv(predictive_dir / "calibration_summary.csv")

    subtitle = "Inferred labels; exploratory analysis"
    masked_heatmap(by_method_proxy, "method_centered_effect", "Method-centered proxy relation effect", subtitle, main_dir / "method_centered_proxy_relation_effect_heatmap.png", config.min_cell_n, True)
    masked_heatmap(by_method_stage, "method_centered_effect", "Method-centered threat stage effect", subtitle, main_dir / "method_centered_threat_stage_effect_heatmap.png", config.min_cell_n, True)
    overall_bar_plot(overall_proxy, "Balanced proxy relation shift overall", subtitle, main_dir / "balanced_proxy_relation_shift_overall.png")
    predictive_ci_plot(pred_summary, main_dir / "predictive_model_comparison_with_ci.png")

    overall_bar_plot(overall_mech, "Balanced risk mechanism shift overall", subtitle, supp_dir / "balanced_risk_mechanism_shift_overall.png")
    overall_bar_plot(overall_stage, "Balanced threat stage shift overall", subtitle, supp_dir / "balanced_threat_stage_shift_overall.png")
    overall_bar_plot(overall_topic, "Balanced topic category shift overall", subtitle, supp_dir / "balanced_topic_category_shift_overall.png")
    masked_heatmap(by_method_mech, "method_centered_effect", "Method-centered risk mechanism effect", subtitle, supp_dir / "method_centered_risk_mechanism_effect_heatmap.png", config.min_cell_n, True)
    masked_heatmap(by_method_topic, "method_centered_effect", "Method-centered topic category effect", subtitle, supp_dir / "method_centered_topic_category_effect_heatmap.png", config.min_cell_n, True)
    calibration_plot(calibration_df, supp_dir / "predictive_model_calibration.png")
    masked_heatmap(by_method_proxy, "method_centered_effect", "All cells unmasked proxy relation heatmap", subtitle, supp_dir / "all_cells_unmasked_proxy_relation_heatmap.png", config.min_cell_n, False)
    masked_heatmap(by_method_stage, "method_centered_effect", "All cells unmasked threat stage heatmap", subtitle, supp_dir / "all_cells_unmasked_threat_stage_heatmap.png", config.min_cell_n, False)


def save_table_with_latex(df: pd.DataFrame, csv_path: Path, caption: str) -> None:
    df.to_csv(csv_path, index=False)
    latex_path = csv_path.with_suffix(".tex")
    cols = [str(col).replace("_", r"\_") for col in df.columns]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{csv_path.stem.replace('.', '_')}}}",
        r"\begin{tabular}{" + "l" * len(cols) + r"}",
        r"\hline",
        " & ".join(cols) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                if math.isnan(value):
                    text = "nan"
                else:
                    text = f"{value:.6g}"
            else:
                text = str(value)
            vals.append(text.replace("_", r"\_"))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    latex = "\n".join(lines)
    write_text(latex_path, latex)


def run_tables_outputs(config: V2Config) -> None:
    tables_dir = ensure_dir(config.output_dir / "tables")
    caption = "Question-level threat-model labels are inferred metadata, not official WMDP annotations."
    save_table_with_latex(pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_shift_proxy_relation.csv"), tables_dir / "table_main_balanced_proxy_relation.csv", caption)
    save_table_with_latex(pd.read_csv(config.output_dir / "intrinsic" / "method_centered_effect_proxy_relation.csv"), tables_dir / "table_main_method_centered_proxy_relation.csv", caption)
    save_table_with_latex(pd.read_csv(config.output_dir / "predictive" / "model_comparison_repeated_grouped_cv.csv"), tables_dir / "table_main_predictive_models.csv", caption)
    save_table_with_latex(pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_shift_risk_stage.csv"), tables_dir / "table_supplementary_threat_stage.csv", caption)
    save_table_with_latex(pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_shift_risk_mechanism.csv"), tables_dir / "table_supplementary_risk_mechanism.csv", caption)
    save_table_with_latex(pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_shift_topic_category.csv"), tables_dir / "table_supplementary_topic_category.csv", caption)


VALIDATION_SCRIPT = r'''from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
eligible = pd.read_csv(ROOT / "data" / "eligible_question_method_rows.csv")
assert eligible["question_id"].notna().all()
assert eligible["method"].notna().all()
assert eligible["outcome"].isin(["suppressed", "forgotten"]).all()
assert eligible.duplicated(["question_id", "method"]).sum() == 0

predictive = pd.read_csv(ROOT / "predictive" / "oof_predictions_all_models.csv")
if not predictive.empty:
    assert predictive["question_id"].notna().all()

summary = pd.read_csv(ROOT / "predictive" / "model_comparison_repeated_grouped_cv.csv")
assert not summary.empty

audit = pd.read_json(ROOT / "audit" / "dataset_audit_summary.json")
text = (ROOT / "audit" / "dataset_audit_summary.md").read_text(encoding="utf-8")
assert "RMU" in text
assert "ideation" in text.lower()

for path in [
    ROOT / "balanced_bootstrap" / "overall_balanced_shift_proxy_relation.csv",
    ROOT / "balanced_bootstrap" / "overall_balanced_shift_risk_stage.csv",
]:
    df = pd.read_csv(path)
    assert (df["bootstrap_seed"] == 42).all()
    assert (df["bootstrap_repetitions"] >= 5000).all()

main_fig_dir = ROOT / "figures" / "main"
assert (main_fig_dir / "method_centered_proxy_relation_effect_heatmap.png").exists()
assert (main_fig_dir / "method_centered_threat_stage_effect_heatmap.png").exists()

print("Validation checks passed.")
'''


def write_validation_script(config: V2Config) -> Path:
    val_dir = ensure_dir(config.output_dir / "validation")
    script_path = val_dir / "run_validation_checks.py"
    write_text(script_path, VALIDATION_SCRIPT)
    return script_path


def run_validation(config: V2Config) -> None:
    script_path = write_validation_script(config)
    proc = subprocess.run([sys.executable, str(script_path)], cwd=ROOT, capture_output=True, text=True, check=False)
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
        "",
    ]
    write_text(config.output_dir / "validation" / "validation_report.md", "\n".join(report))
    if proc.returncode != 0:
        raise RuntimeError("Validation checks failed.")


def generate_report(config: V2Config) -> None:
    audit_json = json.loads((config.output_dir / "audit" / "dataset_audit_summary.json").read_text(encoding="utf-8"))
    proxy = pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_shift_proxy_relation.csv")
    stage = pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_shift_risk_stage.csv")
    mech = pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_shift_risk_mechanism.csv")
    topic = pd.read_csv(config.output_dir / "balanced_bootstrap" / "overall_balanced_shift_topic_category.csv")
    intrinsic_proxy = pd.read_csv(config.output_dir / "intrinsic" / "method_centered_effect_proxy_relation.csv")
    predictive = pd.read_csv(config.output_dir / "predictive" / "model_comparison_repeated_grouped_cv.csv")
    sens_counts = pd.read_csv(config.output_dir / "sensitivity" / "sensitivity_subset_counts.csv")
    sens_bal = pd.read_csv(config.output_dir / "sensitivity" / "sensitivity_balanced_shift_summary.csv")
    sens_pred = pd.read_csv(config.output_dir / "sensitivity" / "sensitivity_predictive_model_summary.csv")

    def top_rows(df: pd.DataFrame, col: str, n: int = 5) -> pd.DataFrame:
        return df.sort_values(col, ascending=False).head(n)

    lines = [
        "# Threat-Model Analysis v2 Report",
        "",
        "## A. Dataset and audit",
        "",
        f"- Rows: {audit_json['total_rows']}",
        f"- Unique questions: {audit_json['unique_question_ids']}",
        f"- Methods included: {', '.join(audit_json['method_audit']['present_methods'])}",
        f"- Plain RMU present: {audit_json['method_audit']['rmu_present']}",
        f"- Ideation eligible rows: {audit_json['ideation_eligible_rows']}",
        f"- Suppressed count: {audit_json['suppressed_count']}",
        f"- Forgotten count: {audit_json['forgotten_count']}",
        f"- Review-queue rate: {audit_json['review_queue_rate']:.3f}",
        "",
        "## B. Balanced pooled results",
        "",
        "### Proxy relation",
        "",
        df_to_markdown_table(proxy),
        "",
        "### Risk stage",
        "",
        df_to_markdown_table(stage),
        "",
        "### Risk mechanism",
        "",
        df_to_markdown_table(mech),
        "",
        "### Topic category",
        "",
        df_to_markdown_table(topic),
        "",
        "## C. Method-centered results",
        "",
        "Only cells with n_unique_questions >= 10 should be treated as paper-facing.",
        "",
        df_to_markdown_table(intrinsic_proxy.loc[intrinsic_proxy['n_unique_questions'] >= config.min_cell_n]),
        "",
        "## D. Predictive results",
        "",
        df_to_markdown_table(predictive),
        "",
        "## E. Sensitivity results",
        "",
        df_to_markdown_table(sens_counts),
        "",
        df_to_markdown_table(sens_bal.head(20)),
        "",
        df_to_markdown_table(sens_pred.head(20)),
        "",
        "## F. Recommended paper framing",
        "",
        "Using inferred threat-model metadata, we observe suggestive method-specific differences in the suppression-versus-forgetting profile. Proxy-relation labels appear more informative than broad risk-mechanism labels, although predictive gains remain modest and should be interpreted as exploratory.",
        "",
    ]
    write_text(config.output_dir / "REPORT.md", "\n".join(lines))


def run_all(config: V2Config) -> None:
    save_command_environment(config, sys.argv)
    append_run_log(config, "Starting v2 pipeline.")
    eligible = prepare_eligible_outputs(config)
    append_run_log(config, f"Eligible rows prepared: {len(eligible)} rows.")
    run_balanced_bootstrap_outputs(config, eligible)
    append_run_log(config, "Balanced clustered bootstrap complete.")
    run_intrinsic_outputs(config, eligible)
    append_run_log(config, "Intrinsic method-centered effects complete.")
    run_predictive_outputs(config, eligible)
    append_run_log(config, "Repeated grouped CV complete.")
    run_sensitivity_outputs(config, eligible)
    append_run_log(config, "Sensitivity analyses complete.")
    run_unclear_proxy_audit(config, eligible)
    append_run_log(config, "Unclear proxy audit complete.")
    run_figures_outputs(config)
    append_run_log(config, "Figures complete.")
    run_tables_outputs(config)
    append_run_log(config, "Tables complete.")
    run_validation(config)
    append_run_log(config, "Validation complete.")
    generate_report(config)
    append_run_log(config, "Report complete.")
