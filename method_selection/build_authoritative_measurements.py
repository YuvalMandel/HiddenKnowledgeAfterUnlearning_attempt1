#!/usr/bin/env python3
"""
build_authoritative_measurements.py

Aggregates per-example band prediction CSVs produced by re_dr_analysis.py into
the per-(question_group_id, method, train_regime) measurement table that the
Phase-2.2 method-selection framework (phase22_lib.py) consumes.

Inputs (from method_selection_out/re_dr/{method}/{tag}/band_predictions/):
  {METHOD}_{band}_none_{eval_split}_{train_scheme}_{train_window_id}_predictions.csv
    columns: eval_row_order, gold_label, base_probe_on_base_pred,
             method_probe_on_method_pred, frozen_base_probe_on_method_pred

  checkpoints/bio_labels.csv                 → question_group_id per test row
  checkpoints/{METHOD}_bio_logit_test.csv    → tf_margin per test row (leakage proxy)

Output:
  method_selection_out/authoritative_row_measurements.csv
    columns: question_group_id, method, train_regime,
             RE_signal, frozen_recovery_mean, retrained_recovery_mean,
             recovery_gain_mean, leakage_proxy

Per-question aggregation (each question has exactly 2 rows: True + False pair):
  frozen_recovery_mean   = mean(frozen_probe_pred == gold_label) over pair
  retrained_recovery_mean= mean(method_probe_pred == gold_label) over pair
  recovery_gain_mean     = retrained - frozen
  RE_signal              = base_probe_acc_on_base - frozen_probe_acc_on_method
                         = how much accuracy was LOST from base to frozen-on-method
  leakage_proxy          = mean(|tf_margin|) for the method — high = model still confident

Usage (after all re_dr_analysis.py jobs have finished):
  python method_selection/build_authoritative_measurements.py
  python method_selection/build_authoritative_measurements.py --band late
  python method_selection/build_authoritative_measurements.py --all_bands
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


CHECKPOINT_DIR   = Path("checkpoints")
RE_DR_ROOT       = Path("method_selection_out") / "re_dr"
OUT_CSV          = Path("method_selection_out") / "authoritative_row_measurements.csv"

ALL_METHODS = [
    "GradDiff", "RMU", "RMU-LAT", "RepNoise",
    "ELM", "RR", "TAR", "PB_J",
]

BANDS = ["early", "mid", "late"]


def load_bio_labels_test(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df["split"] = df["split"].astype(str).str.lower()
    test = df[df["split"] == "test"].copy().reset_index(drop=True)
    test["eval_row_order"] = np.arange(len(test), dtype=int)
    return test[["eval_row_order", "question_group_id", "gold_label"]]


def find_prediction_csv(
    re_dr_root: Path, method: str,
    band: str, eval_split: str,
    train_scheme: str, train_window_id: str,
) -> Optional[Path]:
    tag   = f"{train_scheme}_{train_window_id}"
    fname = f"{method}_{band}_none_{eval_split}_{train_scheme}_{train_window_id}_predictions.csv"
    for candidate in [
        re_dr_root / method / tag / "band_predictions" / fname,
        re_dr_root / "multi" / tag / "band_predictions" / fname,
    ]:
        if candidate.exists():
            return candidate
    return None


def load_logit_csv(method: str, eval_split: str) -> Optional[pd.DataFrame]:
    path = CHECKPOINT_DIR / f"{method}_bio_logit_{eval_split}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["eval_row_order"] = np.arange(len(df), dtype=int)
    return df[["eval_row_order", "tf_margin"]]


def aggregate_band_predictions(
    pred_csv: Path,
    labels_test: pd.DataFrame,
) -> pd.DataFrame:
    """Join predictions with labels on eval_row_order, return per-question stats."""
    preds = pd.read_csv(pred_csv)
    merged = preds.merge(labels_test, on=["eval_row_order", "gold_label"], how="inner")
    if len(merged) == 0:
        # Try without gold_label constraint (if gold_label in predictions was set correctly)
        merged = preds.merge(
            labels_test[["eval_row_order", "question_group_id"]], on="eval_row_order", how="inner"
        )
        merged["gold_label"] = merged["gold_label_x"] if "gold_label_x" in merged.columns else preds["gold_label"]

    merged["base_correct"]    = (merged["base_probe_on_base_pred"]       == merged["gold_label"]).astype(int)
    merged["frozen_correct"]  = (merged["frozen_base_probe_on_method_pred"] == merged["gold_label"]).astype(int)
    merged["retrain_correct"] = (merged["method_probe_on_method_pred"]   == merged["gold_label"]).astype(int)

    agg = (merged
           .groupby("question_group_id", as_index=False)
           .agg(
               base_acc_mean    =("base_correct",    "mean"),
               frozen_acc_mean  =("frozen_correct",  "mean"),
               retrain_acc_mean =("retrain_correct", "mean"),
               n_rows           =("gold_label",       "count"),
           ))
    agg["RE_signal"]              = agg["base_acc_mean"] - agg["frozen_acc_mean"]
    agg["frozen_recovery_mean"]   = agg["frozen_acc_mean"]
    agg["retrained_recovery_mean"]= agg["retrain_acc_mean"]
    agg["recovery_gain_mean"]     = agg["retrained_recovery_mean"] - agg["frozen_recovery_mean"]
    return agg[["question_group_id", "RE_signal",
                "frozen_recovery_mean", "retrained_recovery_mean", "recovery_gain_mean"]]


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate RE/DR band predictions → authoritative_row_measurements.csv"
    )
    p.add_argument("--band",          type=str, default="late",
                   choices=["early", "mid", "late"],
                   help="Primary band to use for RE/recovery signals (default: late)")
    p.add_argument("--all_bands",     action="store_true",
                   help="Use mean across all 3 bands instead of a single band")
    p.add_argument("--eval_split",    type=str, default="test")
    p.add_argument("--train_scheme",  type=str, default="full")
    p.add_argument("--train_window_id", type=str, default="full")
    p.add_argument("--methods",       type=str, default=None,
                   help="Comma-separated list of methods; default: all 8")
    p.add_argument("--re_dr_root",    type=str, default=str(RE_DR_ROOT))
    p.add_argument("--labels_csv",    type=str, default=str(CHECKPOINT_DIR / "bio_labels.csv"))
    p.add_argument("--out_csv",       type=str, default=str(OUT_CSV))
    return p.parse_args()


def main():
    args  = parse_args()
    bands = BANDS if args.all_bands else [args.band]
    methods = (
        [m.strip() for m in args.methods.split(",") if m.strip()]
        if args.methods else ALL_METHODS
    )
    re_dr_root = Path(args.re_dr_root)
    labels_csv = Path(args.labels_csv)
    out_csv    = Path(args.out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading test labels from {labels_csv} …")
    labels_test = load_bio_labels_test(labels_csv)
    print(f"  {len(labels_test)} test rows, "
          f"{labels_test['question_group_id'].nunique()} unique questions")

    all_rows = []

    for method in methods:
        logit_df = load_logit_csv(method, args.eval_split)
        if logit_df is not None:
            logit_merged = labels_test.merge(logit_df, on="eval_row_order", how="left")
            leakage_by_q = (
                logit_merged
                .assign(abs_margin=logit_merged["tf_margin"].abs())
                .groupby("question_group_id", as_index=False)
                .agg(leakage_proxy=("abs_margin", "mean"))
            )
        else:
            print(f"  [warn] No logit CSV for {method} — leakage_proxy will be NaN")
            leakage_by_q = pd.DataFrame(
                {"question_group_id": labels_test["question_group_id"].unique(),
                 "leakage_proxy": float("nan")}
            )

        band_dfs = []
        for band in bands:
            pred_csv = find_prediction_csv(
                re_dr_root, method, band,
                args.eval_split, args.train_scheme, args.train_window_id,
            )
            if pred_csv is None:
                print(f"  [warn] Missing band predictions for {method}/{band} — skipping")
                continue
            agg = aggregate_band_predictions(pred_csv, labels_test)
            band_dfs.append(agg)
            print(f"  {method}/{band}: {len(agg)} questions aggregated")

        if not band_dfs:
            print(f"  [skip] No band predictions found for {method}")
            continue

        if len(band_dfs) == 1:
            combined = band_dfs[0].copy()
        else:
            # mean across bands
            combined = band_dfs[0][["question_group_id"]].copy()
            for col in ["RE_signal", "frozen_recovery_mean", "retrained_recovery_mean", "recovery_gain_mean"]:
                combined[col] = np.nanmean(
                    [df.set_index("question_group_id")[col] for df in band_dfs], axis=0
                )

        combined = combined.merge(leakage_by_q, on="question_group_id", how="left")
        combined["method"]       = method
        combined["train_regime"] = (
            "full" if args.train_scheme == "full"
            else args.train_window_id
        )
        all_rows.append(combined)

    if not all_rows:
        print("ERROR: No data produced.  Run re_dr_analysis.py for all methods first.")
        return

    result = pd.concat(all_rows, ignore_index=True)
    result = result[[
        "question_group_id", "method", "train_regime",
        "RE_signal", "frozen_recovery_mean", "retrained_recovery_mean",
        "recovery_gain_mean", "leakage_proxy",
    ]]
    result.to_csv(out_csv, index=False)

    print(f"\nWrote {len(result)} rows to {out_csv}")
    print(f"  methods     : {sorted(result['method'].unique().tolist())}")
    print(f"  questions   : {result['question_group_id'].nunique()}")
    print(f"  train_regimes: {sorted(result['train_regime'].unique().tolist())}")
    print("\nColumn stats:")
    for col in ["RE_signal", "frozen_recovery_mean", "retrained_recovery_mean",
                "recovery_gain_mean", "leakage_proxy"]:
        s = result[col].dropna()
        print(f"  {col:30s}  mean={s.mean():.3f}  std={s.std():.3f}  "
              f"min={s.min():.3f}  max={s.max():.3f}  nan={result[col].isna().sum()}")


if __name__ == "__main__":
    main()
