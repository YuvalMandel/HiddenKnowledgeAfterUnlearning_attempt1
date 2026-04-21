#!/usr/bin/env python3
"""
export_bio_labels_csv.py

Convert data/wmdp_tf_pairs.csv into the CSV format expected by re_dr_analysis.py
and confidence_quadrant_analysis.py.

Output: checkpoints/bio_labels.csv
  question_group_id  — original question ID (same for True and False pair)
  split              — train / val / test
  gold_label         — 1 (True statement) or 0 (False statement)

Run once after the base stage has completed.
"""

import csv
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR       = Path("data")
WMDP_CSV_PATH  = DATA_DIR / "wmdp_tf_pairs.csv"
OUT_PATH       = CHECKPOINT_DIR / "bio_labels.csv"


def main():
    if not WMDP_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{WMDP_CSV_PATH} not found.  Run --stage base first."
        )
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    rows = []
    with open(WMDP_CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "question_group_id": r["original_id"],
                "split":             r["split"],
                "gold_label":        1 if r["label"].strip() == "True" else 0,
            })

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question_group_id", "split", "gold_label"])
        w.writeheader()
        w.writerows(rows)

    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], 0)
        by_split[r["split"]] += 1

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    for sp, n in sorted(by_split.items()):
        print(f"  {sp}: {n} rows")


if __name__ == "__main__":
    main()
