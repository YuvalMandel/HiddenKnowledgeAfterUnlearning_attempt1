from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CANDIDATES = [
    ROOT / "LLM as as Judge" / "wmdp_bio_inferred_categories_first_pass.parquet",
    ROOT / "LLM as as Judge" / "wmdp_bio_inferred_categories_first_pass.csv",
]
DEFAULT_APPENDIX_CATEGORIES = ROOT / "LLM as as Judge" / "wmdp_bio_inferred_categories_first_pass.csv"
DEFAULT_PROMPT = ROOT / "prompts" / "wmdp_bio_threat_model_judge_v1.txt"
DEFAULT_OUTPUT_DIR = ROOT / "out_threat_model_judge_v1"

RISK_MECHANISMS = {
    "expanding_access",
    "unlocking_new_biology",
    "cross_cutting_or_unclear",
}
RISK_STAGES = {
    "ideation",
    "design",
    "build",
    "test",
    "learn",
    "release",
    "background_or_cross_cutting",
}
PROXY_RELATIONS = {
    "precursor",
    "component",
    "neighbor",
    "general_proxy",
    "unclear",
}
EXPECTED_KEYS = {
    "risk_mechanism_primary",
    "risk_mechanism_secondary",
    "risk_stage_primary",
    "risk_stage_secondary",
    "proxy_relation_primary",
    "risk_mechanism_confidence",
    "risk_stage_confidence",
    "proxy_relation_confidence",
    "rationale_short",
    "needs_review",
}
DEFAULT_REVIEW_ROW = {
    "risk_mechanism_primary": "cross_cutting_or_unclear",
    "risk_mechanism_secondary": None,
    "risk_stage_primary": "background_or_cross_cutting",
    "risk_stage_secondary": None,
    "proxy_relation_primary": "unclear",
    "risk_mechanism_confidence": 0.0,
    "risk_stage_confidence": 0.0,
    "proxy_relation_confidence": 0.0,
    "rationale_short": "Automatic review fallback; inferred labels require manual verification.",
    "needs_review": True,
}


def infer_default_input() -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not infer a default WMDP-Bio input file.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--appendix-categories", default=str(DEFAULT_APPENDIX_CATEGORIES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--second-pass", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--review-threshold-stage", type=float, default=0.65)
    parser.add_argument("--review-threshold-mechanism", type=float, default=0.65)
    parser.add_argument("--review-threshold-proxy", type=float, default=0.55)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def parse_choices(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except Exception:
            pass
        if text.startswith("[") and text.endswith("]"):
            return [text]
        return [line.strip() for line in text.splitlines() if line.strip()]
    if pd.isna(value):
        return []
    return [str(value).strip()]


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def compute_question_hash(question: str, choices: list[str]) -> str:
    payload = normalize_text(question) + "\n" + json.dumps([normalize_text(c) for c in choices], ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "question_id" not in out.columns:
        out = out.reset_index().rename(columns={"index": "question_id"})
    if "choices" not in out.columns:
        if "choices_json" in out.columns:
            out["choices"] = out["choices_json"]
        else:
            raise ValueError("Input must contain `choices` or `choices_json`.")
    required = {"question_id", "question", "choices"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")
    out["question"] = out["question"].astype(str)
    out["choices_list"] = out["choices"].apply(parse_choices)
    out["choices_json"] = out["choices_list"].apply(lambda items: json.dumps(items, ensure_ascii=False))
    out["question_hash"] = out.apply(lambda row: compute_question_hash(row["question"], row["choices_list"]), axis=1)
    return out


def format_question_block(question: str, choices_json: str) -> str:
    return (
        "Classify this WMDP-Bio question according to the taxonomy below.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CHOICES:\n{choices_json}\n"
    )


def format_second_pass_block(row: pd.Series, first_pass: dict[str, Any]) -> str:
    return (
        "Review the first-pass inferred labels for this WMDP-Bio question.\n"
        "Either confirm them or revise them. Keep the rationale high level.\n\n"
        f"QUESTION:\n{row['question']}\n\n"
        f"CHOICES:\n{row['choices_json']}\n\n"
        "FIRST PASS:\n"
        f"{json.dumps(first_pass, ensure_ascii=False)}\n"
    )


def coerce_optional_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def clamp_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception as exc:
        raise ValueError(f"Confidence must be numeric: {value}") from exc
    return max(0.0, min(1.0, numeric))


def validate_payload(payload: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    unknown = sorted(set(payload) - EXPECTED_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys: {unknown}")

    cleaned = dict(payload)
    cleaned["risk_mechanism_primary"] = str(cleaned.get("risk_mechanism_primary", "")).strip()
    cleaned["risk_stage_primary"] = str(cleaned.get("risk_stage_primary", "")).strip()
    cleaned["proxy_relation_primary"] = str(cleaned.get("proxy_relation_primary", "")).strip()
    cleaned["risk_mechanism_secondary"] = coerce_optional_label(cleaned.get("risk_mechanism_secondary"))
    cleaned["risk_stage_secondary"] = coerce_optional_label(cleaned.get("risk_stage_secondary"))
    cleaned["risk_mechanism_confidence"] = clamp_confidence(cleaned.get("risk_mechanism_confidence", 0.0))
    cleaned["risk_stage_confidence"] = clamp_confidence(cleaned.get("risk_stage_confidence", 0.0))
    cleaned["proxy_relation_confidence"] = clamp_confidence(cleaned.get("proxy_relation_confidence", 0.0))
    cleaned["rationale_short"] = " ".join(str(cleaned.get("rationale_short", "")).split())[:500]
    cleaned["needs_review"] = bool(cleaned.get("needs_review", False))

    if cleaned["risk_mechanism_primary"] not in RISK_MECHANISMS:
        raise ValueError(f"Invalid risk_mechanism_primary: {cleaned['risk_mechanism_primary']}")
    if cleaned["risk_stage_primary"] not in RISK_STAGES:
        raise ValueError(f"Invalid risk_stage_primary: {cleaned['risk_stage_primary']}")
    if cleaned["proxy_relation_primary"] not in PROXY_RELATIONS:
        raise ValueError(f"Invalid proxy_relation_primary: {cleaned['proxy_relation_primary']}")
    if cleaned["risk_mechanism_secondary"] is not None and cleaned["risk_mechanism_secondary"] not in RISK_MECHANISMS:
        raise ValueError(f"Invalid risk_mechanism_secondary: {cleaned['risk_mechanism_secondary']}")
    if cleaned["risk_stage_secondary"] is not None and cleaned["risk_stage_secondary"] not in RISK_STAGES:
        raise ValueError(f"Invalid risk_stage_secondary: {cleaned['risk_stage_secondary']}")
    if cleaned["risk_mechanism_secondary"] == cleaned["risk_mechanism_primary"]:
        raise ValueError("Mechanism primary and secondary must differ.")
    if cleaned["risk_stage_secondary"] == cleaned["risk_stage_primary"]:
        raise ValueError("Stage primary and secondary must differ.")
    if not cleaned["rationale_short"]:
        raise ValueError("rationale_short must be non-empty.")

    cleaned["needs_review"] = (
        cleaned["needs_review"]
        or cleaned["risk_mechanism_confidence"] < thresholds["mechanism"]
        or cleaned["risk_stage_confidence"] < thresholds["stage"]
        or cleaned["proxy_relation_confidence"] < thresholds["proxy"]
    )
    return cleaned


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class OpenAIChatJudge:
    def __init__(self, api_key: str, api_base: str, model: str, temperature: float, seed: int) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed

    def complete_json(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], str]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "seed": self.seed,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.api_base}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Network error: {exc}") from exc

        parsed = json.loads(raw)
        text = parsed["choices"][0]["message"]["content"]
        return json.loads(text), text


def load_cache(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (
                record["question_hash"],
                record["judge_model"],
                record["judge_prompt_version"],
                record.get("pass_name", "first_pass"),
            )
            cache[key] = record
    return cache


def append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_failure(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    append_jsonl(path, record, lock)


def log_line(path: Path, message: str, lock: threading.Lock) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")


def dry_run_payload(row: pd.Series, thresholds: dict[str, float]) -> tuple[dict[str, Any], str]:
    placeholder = dict(DEFAULT_REVIEW_ROW)
    placeholder["rationale_short"] = (
        "Dry run placeholder based on question text only; inferred threat-model labels were not requested from a live model."
    )
    validated = validate_payload(placeholder, thresholds)
    return validated, json.dumps(placeholder, ensure_ascii=False)


def run_pass(
    row: pd.Series,
    pass_name: str,
    system_prompt: str,
    user_prompt: str,
    judge: OpenAIChatJudge | None,
    thresholds: dict[str, float],
    max_retries: int,
    dry_run: bool,
) -> tuple[dict[str, Any], str]:
    if dry_run:
        return dry_run_payload(row, thresholds)

    if judge is None:
        raise RuntimeError("No OpenAI API key available. Set the requested API key env var or use --dry-run.")

    latest_text = ""
    latest_error = ""
    prompt = user_prompt
    for attempt in range(1, max_retries + 1):
        try:
            parsed, latest_text = judge.complete_json(system_prompt, prompt)
            validated = validate_payload(parsed, thresholds)
            return validated, latest_text
        except Exception as exc:
            latest_error = str(exc)
            prompt = (
                user_prompt
                + "\n\nThe previous response was invalid. Return JSON only with valid enum values, numeric confidences in [0,1], "
                + "distinct primary vs secondary labels, and no extra keys."
            )
            time.sleep(min(2 ** (attempt - 1), 8))
    fallback = dict(DEFAULT_REVIEW_ROW)
    fallback["rationale_short"] = f"Automatic review fallback after {pass_name} failure: {latest_error[:220]}"
    validated = validate_payload(fallback, thresholds)
    return validated, latest_text


def make_output_record(
    row: pd.Series,
    payload: dict[str, Any],
    raw_json: str,
    pass_name: str,
    model: str,
    prompt_version: str,
) -> dict[str, Any]:
    return {
        "question_hash": row["question_hash"],
        "question_id": int(row["question_id"]),
        "judge_model": model,
        "judge_prompt_version": prompt_version,
        "judge_run_timestamp": datetime.now(timezone.utc).isoformat(),
        "pass_name": pass_name,
        "result": payload,
        "raw_json": raw_json,
    }


def build_row_output(
    row: pd.Series,
    first_pass: dict[str, Any],
    first_raw: str,
    model: str,
    prompt_version: str,
    second_pass: dict[str, Any] | None = None,
    second_raw: str | None = None,
) -> dict[str, Any]:
    out = {
        "question_id": int(row["question_id"]),
        "question_hash": row["question_hash"],
        "question": row["question"],
        "choices": row["choices_json"],
        "risk_mechanism_primary": first_pass["risk_mechanism_primary"],
        "risk_mechanism_secondary": first_pass["risk_mechanism_secondary"],
        "risk_stage_primary": first_pass["risk_stage_primary"],
        "risk_stage_secondary": first_pass["risk_stage_secondary"],
        "risk_stage_analysis": "test_learn"
        if first_pass["risk_stage_primary"] in {"test", "learn"}
        else first_pass["risk_stage_primary"],
        "proxy_relation_primary": first_pass["proxy_relation_primary"],
        "risk_mechanism_confidence": first_pass["risk_mechanism_confidence"],
        "risk_stage_confidence": first_pass["risk_stage_confidence"],
        "proxy_relation_confidence": first_pass["proxy_relation_confidence"],
        "rationale_short": first_pass["rationale_short"],
        "needs_review": bool(first_pass["needs_review"]),
        "judge_model": model,
        "judge_prompt_version": prompt_version,
        "judge_run_timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_raw_json": first_raw,
        "second_pass_used": False,
        "second_pass_changed_any_label": False,
        "second_pass_raw_json": None,
        "risk_mechanism_final": first_pass["risk_mechanism_primary"],
        "risk_stage_final": first_pass["risk_stage_primary"],
        "proxy_relation_final": first_pass["proxy_relation_primary"],
        "risk_stage_analysis_final": "test_learn"
        if first_pass["risk_stage_primary"] in {"test", "learn"}
        else first_pass["risk_stage_primary"],
    }
    if second_pass is not None:
        out["second_pass_used"] = True
        out["second_pass_raw_json"] = second_raw
        changed = any(
            [
                second_pass["risk_mechanism_primary"] != first_pass["risk_mechanism_primary"],
                second_pass["risk_stage_primary"] != first_pass["risk_stage_primary"],
                second_pass["proxy_relation_primary"] != first_pass["proxy_relation_primary"],
            ]
        )
        out["second_pass_changed_any_label"] = changed
        out["risk_mechanism_final"] = second_pass["risk_mechanism_primary"]
        out["risk_stage_final"] = second_pass["risk_stage_primary"]
        out["proxy_relation_final"] = second_pass["proxy_relation_primary"]
        out["risk_stage_analysis_final"] = (
            "test_learn" if second_pass["risk_stage_primary"] in {"test", "learn"} else second_pass["risk_stage_primary"]
        )
    return out


def summarize_counts(series: pd.Series) -> pd.DataFrame:
    counts = series.fillna("missing").value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    counts["pct"] = counts["count"] / counts["count"].sum() * 100
    return counts


def summarize_confidence(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {key: float("nan") for key in ["mean", "median", "p10", "p25", "p75", "p90"]}
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "p10": float(numeric.quantile(0.10)),
        "p25": float(numeric.quantile(0.25)),
        "p75": float(numeric.quantile(0.75)),
        "p90": float(numeric.quantile(0.90)),
    }


def df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                if math.isnan(value):
                    text = "nan"
                else:
                    text = f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(labels: pd.DataFrame, report_path: Path, appendix_categories_path: Path | None) -> None:
    lines: list[str] = []
    lines.append("# Inferred WMDP-Bio Threat-Model Labels Report")
    lines.append("")
    lines.append("These are inferred threat-model labels, not official WMDP metadata.")
    lines.append("")

    sections = [
        ("risk mechanism", "risk_mechanism_final"),
        ("raw risk stage", "risk_stage_final"),
        ("risk_stage_analysis", "risk_stage_analysis_final"),
        ("proxy relationship", "proxy_relation_final"),
    ]
    for title, column in sections:
        lines.append(f"## Count and percentage by {title}")
        lines.append("")
        counts = summarize_counts(labels[column])
        lines.append(df_to_markdown_table(counts))
        lines.append("")

    lines.append("## Confidence distributions")
    lines.append("")
    conf_rows = []
    for column in ["risk_mechanism_confidence", "risk_stage_confidence", "proxy_relation_confidence"]:
        stats = summarize_confidence(labels[column])
        conf_rows.append({"field": column, **stats})
    lines.append(df_to_markdown_table(pd.DataFrame(conf_rows)))
    lines.append("")

    review_count = int(labels["needs_review"].sum())
    second_pass_changes = int(labels.get("second_pass_changed_any_label", pd.Series(dtype=bool)).fillna(False).sum())
    lines.append("## Review queue")
    lines.append("")
    lines.append(f"- needs_review rows: {review_count} / {len(labels)} ({review_count / max(len(labels), 1) * 100:.2f}%)")
    lines.append(f"- second-pass label changes: {second_pass_changes}")
    lines.append("")

    lines.append("## Basic sanity checks")
    lines.append("")
    sanity = {
        "missing_values_total": int(labels.isna().sum().sum()),
        "duplicate_question_hash_values": int(labels["question_hash"].duplicated().sum()),
        "small_baskets_lt_10": int((labels["risk_stage_analysis_final"].value_counts() < 10).sum()),
    }
    lines.append(df_to_markdown_table(pd.DataFrame([sanity])))
    lines.append("")

    if appendix_categories_path is not None and appendix_categories_path.exists():
        appendix = pd.read_csv(appendix_categories_path)
        if "question_id" in appendix.columns and "category_first_pass" in appendix.columns:
            merged = labels.merge(
                appendix[["question_id", "category_first_pass"]],
                on="question_id",
                how="left",
            )
            crosstabs = [
                ("Appendix topic x risk_stage_analysis_final", "risk_stage_analysis_final"),
                ("Appendix topic x risk_mechanism_final", "risk_mechanism_final"),
                ("Appendix topic x proxy_relation_final", "proxy_relation_final"),
            ]
            for title, column in crosstabs:
                lines.append(f"## {title}")
                lines.append("")
                table = pd.crosstab(merged["category_first_pass"].fillna("missing"), merged[column].fillna("missing"))
                lines.append(df_to_markdown_table(table.reset_index()))
                lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input) if args.input else infer_default_input()
    appendix_path = Path(args.appendix_categories) if args.appendix_categories else None
    prompt_path = Path(args.prompt_file)
    output_dir = Path(args.output_dir)
    labels_dir = output_dir / "labels"
    cache_dir = output_dir / "cache"
    logs_dir = output_dir / "logs"
    reports_dir = output_dir / "reports"
    for directory in [labels_dir, cache_dir, logs_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "judge_results.jsonl"
    failure_path = logs_dir / "judge_failures.jsonl"
    log_path = logs_dir / "judge_run.log"
    report_path = reports_dir / "report_threat_model_labels.md"
    csv_path = labels_dir / "wmdp_bio_threat_model_labels_inferred_v1.csv"
    parquet_path = labels_dir / "wmdp_bio_threat_model_labels_inferred_v1.parquet"
    audited_path = labels_dir / "wmdp_bio_threat_model_labels_inferred_v1_audited.parquet"

    df = prepare_input(load_table(input_path))
    if args.start_index:
        df = df.iloc[args.start_index :].copy()
    if args.limit is not None:
        df = df.iloc[: args.limit].copy()
    df = df.reset_index(drop=True)

    thresholds = {
        "mechanism": args.review_threshold_mechanism,
        "stage": args.review_threshold_stage,
        "proxy": args.review_threshold_proxy,
    }
    system_prompt = read_prompt(prompt_path)
    api_key = os.environ.get(args.api_key_env)
    judge = None if args.dry_run else OpenAIChatJudge(api_key, args.api_base, args.model, args.temperature, args.seed) if api_key else None

    cache = load_cache(cache_path) if args.resume else {}
    lock = threading.Lock()
    rows: list[dict[str, Any]] = []
    write_mode = "resume" if args.resume else "fresh"
    log_line(log_path, f"Starting judge run in {write_mode} mode on {len(df)} rows.", lock)

    def process_row(row: pd.Series) -> dict[str, Any]:
        first_key = (row["question_hash"], args.model, args.prompt_version, "first_pass")
        second_key = (row["question_hash"], args.model, args.prompt_version, "second_pass")

        if first_key in cache:
            first_pass = cache[first_key]["result"]
            first_raw = cache[first_key]["raw_json"]
        else:
            first_pass, first_raw = run_pass(
                row=row,
                pass_name="first_pass",
                system_prompt=system_prompt,
                user_prompt=format_question_block(row["question"], row["choices_json"]),
                judge=judge,
                thresholds=thresholds,
                max_retries=args.max_retries,
                dry_run=args.dry_run,
            )
            first_record = make_output_record(row, first_pass, first_raw, "first_pass", args.model, args.prompt_version)
            append_jsonl(cache_path, first_record, lock)

        second_pass = None
        second_raw = None
        needs_second_pass = bool(first_pass["needs_review"]) or first_pass["risk_stage_primary"] == "background_or_cross_cutting"
        if args.second_pass and needs_second_pass:
            if second_key in cache:
                second_pass = cache[second_key]["result"]
                second_raw = cache[second_key]["raw_json"]
            else:
                second_pass, second_raw = run_pass(
                    row=row,
                    pass_name="second_pass",
                    system_prompt=system_prompt,
                    user_prompt=format_second_pass_block(row, first_pass),
                    judge=judge,
                    thresholds=thresholds,
                    max_retries=args.max_retries,
                    dry_run=args.dry_run,
                )
                second_record = make_output_record(row, second_pass, second_raw, "second_pass", args.model, args.prompt_version)
                append_jsonl(cache_path, second_record, lock)

        out_row = build_row_output(row, first_pass, first_raw, args.model, args.prompt_version, second_pass, second_raw)
        if out_row["needs_review"]:
            write_failure(
                failure_path,
                {
                    "question_id": out_row["question_id"],
                    "question_hash": out_row["question_hash"],
                    "judge_model": args.model,
                    "judge_prompt_version": args.prompt_version,
                    "risk_stage_primary": out_row["risk_stage_primary"],
                    "risk_mechanism_primary": out_row["risk_mechanism_primary"],
                    "proxy_relation_primary": out_row["proxy_relation_primary"],
                },
                lock,
            )
        return out_row

    if args.concurrency <= 1:
        for _, row in df.iterrows():
            rows.append(process_row(row))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(process_row, row): int(row["question_id"]) for _, row in df.iterrows()}
            for future in as_completed(futures):
                qid = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    log_line(log_path, f"question_id={qid} failed: {exc}", lock)
                    raise

    labels = pd.DataFrame(rows).sort_values("question_id").reset_index(drop=True)
    labels.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    labels.to_parquet(parquet_path, index=False)
    labels.to_parquet(audited_path, index=False)
    write_report(labels, report_path, appendix_path)
    log_line(log_path, f"Completed judge run with {len(labels)} rows.", lock)
    print(csv_path)
    print(parquet_path)
    print(audited_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
