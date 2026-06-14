from __future__ import annotations
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
