#!/usr/bin/env python3
"""
rank_comparison.py — Compare predicted method ranking to actual method ranking.

For each question in the CV test fold, the phase22 classifier produced:
  y_true  = actual best method (scientific score label)
  y_pred  = predicted best method

This script asks: does the global ranking of methods (by how often they are
predicted to be best) match the true ranking (by how often they actually are best)?

Outputs Spearman ρ between predicted and actual win-count rankings per classifier,
and optionally cross-references with recovery rates from analyze_recovery.py.

Usage:
  python method_selection/rank_comparison.py --phase22_dir method_selection_out/phase22/joint
  python method_selection/rank_comparison.py --phase22_dir method_selection_out/phase22/joint \
      --recovery_csv method_selection_out/recovery_analysis/frozen_vs_retrained_by_band_none_test.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LABEL_TO_METHOD = {
    1: "PB_J", 2: "RMU", 3: "GradDiff", 4: "ELM",
    5: "RepNoise", 6: "RR", 7: "RMU_LAT", 8: "TAR",
}
ALL_LABELS = list(LABEL_TO_METHOD.keys())
ALL_METHODS = list(LABEL_TO_METHOD.values())


def load_predictions(phase22_dir: Path, classifiers: list[str], regime: str) -> dict[str, pd.DataFrame]:
    """Load per-classifier prediction CSVs for the given regime."""
    dfs = {}
    for clf in classifiers:
        path = phase22_dir / "scored" / clf / f"predictions_{regime}.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {clf}")
            continue
        df = pd.read_csv(path)
        df["method_true"] = df["y_true"].map(LABEL_TO_METHOD)
        df["method_pred"] = df["y_pred"].map(LABEL_TO_METHOD)
        dfs[clf] = df
    return dfs


def win_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count how often each method wins (y_true) and is predicted (y_pred)."""
    true_counts = df["y_true"].value_counts().reindex(ALL_LABELS, fill_value=0).rename("actual_wins")
    pred_counts = df["y_pred"].value_counts().reindex(ALL_LABELS, fill_value=0).rename("predicted_wins")
    out = pd.DataFrame({"actual_wins": true_counts, "predicted_wins": pred_counts})
    out.index = out.index.map(LABEL_TO_METHOD)
    out.index.name = "method"
    out = out.sort_values("actual_wins", ascending=False).reset_index()
    out["actual_rank"] = range(1, len(out) + 1)
    out_by_pred = out.sort_values("predicted_wins", ascending=False).reset_index(drop=True)
    pred_rank = {row["method"]: i + 1 for i, row in out_by_pred.iterrows()}
    out["predicted_rank"] = out["method"].map(pred_rank)
    return out


def spearman_on_counts(df: pd.DataFrame) -> float:
    rho, _ = spearmanr(df["actual_rank"], df["predicted_rank"])
    return float(rho)


def per_question_accuracy(df: pd.DataFrame) -> float:
    return float((df["y_true"] == df["y_pred"]).mean())


def load_recovery(recovery_csv: str) -> pd.DataFrame | None:
    if not recovery_csv or not Path(recovery_csv).exists():
        return None
    df = pd.read_csv(recovery_csv)
    if "method" not in df.columns or "retrained_recovered_rate" not in df.columns:
        return None
    late = df[df["band"] == "late"].copy() if "band" in df.columns else df.copy()
    late = late[["method", "retrained_recovered_rate", "frozen_recovered_rate"]].copy()
    late = late.sort_values("retrained_recovered_rate", ascending=False).reset_index(drop=True)
    late["recovery_rank"] = range(1, len(late) + 1)
    # Normalize method names to match LABEL_TO_METHOD values
    late["method"] = late["method"].str.replace("-", "_", regex=False)
    return late


def print_rank_table(clf: str, counts: pd.DataFrame, rho: float, acc: float) -> None:
    print(f"\n{'─'*60}")
    print(f"  Classifier: {clf}   Spearman ρ = {rho:+.3f}   per-question acc = {acc:.3f}")
    print(f"{'─'*60}")
    hdr = f"  {'Method':<12} {'ActWins':>8} {'ActRank':>8} {'PredWins':>9} {'PredRank':>9}"
    print(hdr)
    print(f"  {'-'*56}")
    for _, row in counts.iterrows():
        flag = " *" if row["actual_rank"] == row["predicted_rank"] else ""
        print(f"  {row['method']:<12} {int(row['actual_wins']):>8} {int(row['actual_rank']):>8}"
              f" {int(row['predicted_wins']):>9} {int(row['predicted_rank']):>9}{flag}")
    print()


def print_recovery_cross(recovery: pd.DataFrame, all_counts: dict[str, pd.DataFrame]) -> None:
    print(f"\n{'═'*70}")
    print("  Recovery-rank vs predicted-rank cross-reference (late band, retrained)")
    print(f"{'═'*70}")
    merged = recovery.copy()
    for clf, counts in all_counts.items():
        pr = counts.set_index("method")["predicted_rank"].rename(f"pred_rank_{clf}")
        merged = merged.merge(pr.reset_index(), on="method", how="left")
    merged = merged.sort_values("recovery_rank")

    pred_cols = [c for c in merged.columns if c.startswith("pred_rank_")]
    hdr_parts = [f"  {'Method':<12}", f"{'RecovRate':>10}", f"{'RecRank':>8}"]
    for c in pred_cols:
        clf_label = c.replace("pred_rank_", "")
        hdr_parts.append(f"{clf_label+'Rank':>10}")
    print("".join(hdr_parts))
    print(f"  {'-'*66}")
    for _, row in merged.iterrows():
        parts = [f"  {row['method']:<12}",
                 f"{row['retrained_recovered_rate']:>10.3f}",
                 f"{int(row['recovery_rank']):>8}"]
        for c in pred_cols:
            v = row.get(c)
            parts.append(f"{int(v) if pd.notna(v) else 'N/A':>10}")
        print("".join(parts))

    # Spearman ρ: recovery rank vs predicted rank
    print()
    for c in pred_cols:
        clf_label = c.replace("pred_rank_", "")
        valid = merged[["recovery_rank", c]].dropna()
        if len(valid) >= 3:
            rho, pval = spearmanr(valid["recovery_rank"], valid[c])
            print(f"  Spearman ρ(recovery_rank, {clf_label}_rank) = {rho:+.3f}  p={pval:.3f}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Rank methods by win-frequency and compare predicted vs actual.")
    ap.add_argument("--phase22_dir", default="method_selection_out/phase22/joint",
                    help="Root dir of phase22 outputs (must contain scored/{clf}/predictions_*.csv)")
    ap.add_argument("--regime", default="full",
                    help="Which training regime to use for ranking (default: full)")
    ap.add_argument("--classifiers", default="logreg,rf,gb",
                    help="Comma-separated list of classifiers to compare")
    ap.add_argument("--recovery_csv", default="method_selection_out/recovery_analysis/frozen_vs_retrained_by_band_none_test.csv",
                    help="Optional: recovery analysis CSV from analyze_recovery.py for cross-reference")
    ap.add_argument("--out_dir", default=None,
                    help="Optional: write ranking CSVs here")
    args = ap.parse_args()

    phase22_dir = Path(args.phase22_dir)
    classifiers = [c.strip() for c in args.classifiers.split(",")]

    print(f"\nRanking comparison: regime={args.regime}, classifiers={classifiers}")
    print(f"Phase22 dir: {phase22_dir}")

    pred_dfs = load_predictions(phase22_dir, classifiers, args.regime)
    if not pred_dfs:
        raise FileNotFoundError(
            f"No prediction CSVs found for regime='{args.regime}' under {phase22_dir}/scored/."
            " Run slurm_method_selection.sh first."
        )

    print(f"\nLoaded predictions for {len(pred_dfs)} classifier(s).")
    all_counts: dict[str, pd.DataFrame] = {}
    all_rhos: dict[str, float] = {}

    for clf, df in pred_dfs.items():
        n_questions = df["question_group_id"].nunique()
        print(f"  {clf}: {len(df)} rows, {n_questions} unique questions")
        counts = win_counts(df)
        rho = spearman_on_counts(counts)
        acc = per_question_accuracy(df)
        print_rank_table(clf, counts, rho, acc)
        all_counts[clf] = counts
        all_rhos[clf] = rho
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"rank_comparison_{clf}_{args.regime}.csv")
            counts.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")

    # Summary across classifiers
    print(f"\n{'═'*50}")
    print("  Summary: Spearman ρ (actual_rank vs predicted_rank)")
    print(f"{'═'*50}")
    for clf, rho in all_rhos.items():
        print(f"  {clf:<12}  ρ = {rho:+.3f}")

    # Recovery cross-reference
    recovery = load_recovery(args.recovery_csv)
    if recovery is not None:
        print_recovery_cross(recovery, all_counts)
    else:
        print(f"\n  (Recovery CSV not found at {args.recovery_csv} — skipping cross-reference)")


if __name__ == "__main__":
    main()
