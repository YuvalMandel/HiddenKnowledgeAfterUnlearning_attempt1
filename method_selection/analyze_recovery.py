#!/usr/bin/env python3
"""
analyze_recovery.py  —  Method ranking by frozen/retrained recovery rates.

Adapted from analyze_frozen_vs_retrained_band_recovery_folds.py (Handover: Second).

Reads band prediction CSVs produced by re_dr_analysis.py across all 8 methods
(spread across per-method subdirs) and produces:
  - Long-form summary: frozen/retrained recovery per (method, band)
  - Wide pivot:        one row per method, columns per band
  - Band averages:     mean recovery across methods, per band
  - Best-band tables:  which band gives best frozen / retrained / gain per method

Project file layout (output of re_dr_analysis.py --save_band_prediction_csvs):
  method_selection_out/re_dr/{METHOD}/{train_scheme}_{train_window_id}/band_predictions/
    {METHOD}_{band}_{subset_mode}_{eval_split}_{train_scheme}_{train_window_id}_predictions.csv

Usage (after all re_dr_analysis.py jobs finish):
  python method_selection/analyze_recovery.py
  python method_selection/analyze_recovery.py --subset_mode none --eval_split test
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


DEFAULT_RE_DR_ROOT = "method_selection_out/re_dr"
DEFAULT_OUT_DIR    = "method_selection_out/recovery_analysis"


def parse_prediction_filename(filename: str) -> Tuple[str, str, str, str, str, str]:
    """Parse {METHOD}_{band}_{subset_mode}_{eval_split}_{train_scheme}_{train_window_id}_predictions.csv"""
    stem = filename.replace(".csv", "")
    m = re.match(
        r"(.+)_(early|mid|late)_(none|suppressed|emerged)"
        r"_(val|test\d*)_(full|traincv)_(full|traincv\d+)_predictions$",
        stem,
    )
    if not m:
        raise ValueError(f"Unrecognized prediction filename: {filename}")
    return m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)


def compute_recovery_metrics(df: pd.DataFrame) -> dict:
    required = ["gold_label", "frozen_base_probe_on_method_pred", "method_probe_on_method_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    y      = df["gold_label"].astype(int)
    frozen = df["frozen_base_probe_on_method_pred"].astype(int)
    retr   = df["method_probe_on_method_pred"].astype(int)
    n = len(df)
    frozen_count = int((frozen == y).sum())
    retr_count   = int((retr == y).sum())
    return {
        "n_eval": n,
        "frozen_recovered_count":    frozen_count,
        "frozen_recovered_rate":     frozen_count / n if n else float("nan"),
        "retrained_recovered_count": retr_count,
        "retrained_recovered_rate":  retr_count   / n if n else float("nan"),
        "recovery_gain_count":       retr_count - frozen_count,
        "recovery_gain_rate":        (retr_count - frozen_count) / n if n else float("nan"),
    }


def collect_prediction_csvs(re_dr_root: str, train_scheme: str, train_window_id: str) -> List[str]:
    """Walk re_dr/{METHOD}/{train_scheme}_{train_window_id}/band_predictions/ and collect all CSVs."""
    tag = f"{train_scheme}_{train_window_id}"
    root = Path(re_dr_root)
    found = []
    for method_dir in sorted(root.iterdir()):
        pred_dir = method_dir / tag / "band_predictions"
        if pred_dir.is_dir():
            found.extend(sorted(str(p) for p in pred_dir.glob("*_predictions.csv")))
    return found


def main():
    ap = argparse.ArgumentParser(
        description="Rank unlearning methods by frozen/retrained recovery from band prediction CSVs."
    )
    ap.add_argument("--re_dr_root", default=DEFAULT_RE_DR_ROOT,
                    help="Root dir of re_dr_analysis.py outputs (default: method_selection_out/re_dr)")
    ap.add_argument("--train_scheme",    default="full", choices=["full", "traincv"])
    ap.add_argument("--train_window_id", default="full")
    ap.add_argument("--subset_mode", default="none", choices=["none", "suppressed", "emerged"],
                    help="Which subset mode to summarise (default: none = all questions)")
    ap.add_argument("--eval_split", default="test",
                    help="Which eval split to include (default: test)")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_csvs = collect_prediction_csvs(args.re_dr_root, args.train_scheme, args.train_window_id)
    if not all_csvs:
        raise FileNotFoundError(
            f"No band prediction CSVs found under {args.re_dr_root} "
            f"for {args.train_scheme}_{args.train_window_id}. "
            "Run re_dr_analysis.py with --save_band_prediction_csvs first."
        )

    rows: List[dict] = []
    skipped = 0
    for path in all_csvs:
        fn = os.path.basename(path)
        try:
            method, band, subset_mode, eval_split, _, _ = parse_prediction_filename(fn)
        except ValueError:
            skipped += 1
            continue
        if subset_mode != args.subset_mode or eval_split != args.eval_split:
            continue
        df = pd.read_csv(path)
        metrics = compute_recovery_metrics(df)
        metrics.update({
            "method": method, "band": band,
            "subset_mode": subset_mode, "eval_split": eval_split,
            "prediction_csv": path,
        })
        rows.append(metrics)

    if not rows:
        raise ValueError(
            f"No matching CSVs for subset_mode={args.subset_mode}, eval_split={args.eval_split}. "
            f"Found {len(all_csvs)} total CSVs, skipped {skipped} with unrecognised names."
        )

    summary = pd.DataFrame(rows).sort_values(["method", "band"]).reset_index(drop=True)
    tag = f"{args.subset_mode}_{args.eval_split}"

    summary_path = os.path.join(args.out_dir, f"frozen_vs_retrained_by_band_{tag}.csv")
    summary.to_csv(summary_path, index=False)

    wide = summary.pivot(index="method", columns="band", values=[
        "n_eval",
        "frozen_recovered_count", "frozen_recovered_rate",
        "retrained_recovered_count", "retrained_recovered_rate",
        "recovery_gain_count", "recovery_gain_rate",
    ])
    wide.columns = [f"{metric}_{band}" for metric, band in wide.columns]
    wide = wide.reset_index()
    wide_path = os.path.join(args.out_dir, f"frozen_vs_retrained_band_wide_{tag}.csv")
    wide.to_csv(wide_path, index=False)

    band_avg = summary.groupby("band", as_index=False)[[
        "n_eval",
        "frozen_recovered_count", "frozen_recovered_rate",
        "retrained_recovered_count", "retrained_recovered_rate",
        "recovery_gain_count", "recovery_gain_rate",
    ]].mean(numeric_only=True)
    band_avg_path = os.path.join(args.out_dir, f"frozen_vs_retrained_band_averages_{tag}.csv")
    band_avg.to_csv(band_avg_path, index=False)

    def best_band(metric: str) -> pd.DataFrame:
        idx = summary.groupby("method")[metric].idxmax()
        return summary.loc[idx, ["method", "band", metric]].sort_values("method").reset_index(drop=True)

    best_frozen   = best_band("frozen_recovered_rate")
    best_retrained = best_band("retrained_recovered_rate")
    best_gain      = best_band("recovery_gain_rate")

    best_frozen.to_csv(   os.path.join(args.out_dir, f"best_band_frozen_{tag}.csv"),    index=False)
    best_retrained.to_csv(os.path.join(args.out_dir, f"best_band_retrained_{tag}.csv"), index=False)
    best_gain.to_csv(     os.path.join(args.out_dir, f"best_band_gain_{tag}.csv"),      index=False)

    print(f"Methods found: {sorted(summary['method'].unique().tolist())}")
    print(f"Rows processed: {len(rows)}")
    print(f"\nRetrained recovery ranking (late band):")
    late = summary[summary["band"] == "late"].sort_values("retrained_recovered_rate", ascending=False)
    for _, r in late.iterrows():
        print(f"  {r['method']:12s}  retrained={r['retrained_recovered_rate']:.3f}  "
              f"frozen={r['frozen_recovered_rate']:.3f}  gain={r['recovery_gain_rate']:.3f}")
    print(f"\nOutputs in {args.out_dir}/")
    print(f"  {os.path.basename(summary_path)}")
    print(f"  {os.path.basename(wide_path)}")
    print(f"  {os.path.basename(band_avg_path)}")
    print(f"  best_band_{{frozen,retrained,gain}}_{tag}.csv")


if __name__ == "__main__":
    main()
