from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phase22_lib import ensure_dir, resolve_default_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report for phase (2.2)")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    package_dir = Path(__file__).resolve().parent
    defaults = resolve_default_paths(package_dir)
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else defaults["phase22_out"])
    models_dir = out_dir / "models"
    reports_dir = ensure_dir(out_dir / "reports")
    datasets_dir = out_dir / "datasets"

    fold_df = pd.read_csv(models_dir / "fold_results.csv")
    preds_df = pd.read_csv(models_dir / "oof_predictions.csv")
    conf_df = pd.read_csv(models_dir / "confusion_by_fold.csv")
    regime_summary = pd.read_csv(models_dir / "regime_summary.csv")
    ds_summary = pd.read_csv(datasets_dir / "phase22_datasets_summary.csv")

    overall = {
        "n_total_runs": int(len(fold_df)),
        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "std_accuracy": float(fold_df["accuracy"].std()),
        "mean_balanced_accuracy": float(fold_df["balanced_accuracy"].mean()),
        "mean_macro_f1": float(fold_df["macro_f1"].mean()),
        "mean_top2_accuracy": float(fold_df["top2_accuracy"].mean()),
        "mean_train_questions": float(fold_df["n_train_questions"].mean()),
        "mean_test_questions": float(fold_df["n_test_questions"].mean()),
    }
    pd.DataFrame([overall]).to_csv(reports_dir / "overall_summary.csv", index=False)

    winner_counts = preds_df["y_pred"].value_counts().sort_index().rename_axis("pred_label").reset_index(name="n_questions_predicted")
    winner_counts.to_csv(reports_dir / "predicted_method_counts.csv", index=False)
    conf_df.to_csv(reports_dir / "confusion_by_fold.csv", index=False)
    regime_summary.to_csv(reports_dir / "regime_summary.csv", index=False)
    fold_df.to_csv(reports_dir / "fold_results.csv", index=False)

    md = []
    md.append("# Phase (2.2) Report\n")
    md.append("## Setup\n")
    md.append("- X: pair/confusion-structure features only\n")
    md.append("- Y: best-method label from scientific target preset\n")
    md.append("- Classifier: multiclass baseline\n")
    md.append("- Universe: val + test = 1546 measured rows = 773 question pairs\n")
    md.append("- Regimes: 7 feature-space sets (full + traincv1..6)\n")
    md.append("- CV: 5-fold GroupKFold by question_group_id, separately per regime\n")
    md.append("\n## Overall metrics across 35 runs\n")
    for k, v in overall.items():
        md.append(f"- {k}: {v}\n")
    md.append("\n## Per-regime summary\n\n")
    md.append(regime_summary.to_markdown(index=False))
    md.append("\n\n## Dataset summary\n\n")
    md.append(ds_summary.to_markdown(index=False))
    (reports_dir / "phase22_report.md").write_text("".join(md), encoding="utf-8")
    print("[report] completed", flush=True)


if __name__ == "__main__":
    main()
