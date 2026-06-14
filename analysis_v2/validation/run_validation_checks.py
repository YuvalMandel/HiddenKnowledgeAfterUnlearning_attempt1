from __future__ import annotations
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
