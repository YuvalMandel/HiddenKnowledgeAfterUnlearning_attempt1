from __future__ import annotations
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
