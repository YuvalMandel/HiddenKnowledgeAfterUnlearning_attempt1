Hidden-knowledge figure package with two confidence systems

What is inside
--------------
1) raw_base_threshold/
   Absolute-confidence view anchored to the untouched Base model.
   - Binary confident split: Base median abs(raw_margin)
   - Low/mid/high buckets: Base terciles of abs(raw_margin)

2) percentile_to_base/
   Relative-confidence view that preserves each method's own sample ordering.
   - Per sample, confidence percentile is computed within its own method
   - Binary confident split: percentile >= 0.5
   - Low/mid/high buckets: percentile terciles
   - confidence_score_display stores the Base-mapped confidence value for reference

Figure logic
------------
Figure 1:
  Quadrant prevalence with internal decomposition of each bar into gold=F and gold=T mass.
  This makes single-class quadrants immediately visible.

Figure 2b:
  Probe accuracy by quadrant.
  Two variants are provided:
    - early_rf
    - mid_linear

Figure 3b:
  Apples-to-apples comparison using confidence buckets, not quadrants:
    x-axis = LLM accuracy vs gold
    y-axis = probe accuracy vs gold
  Two variants are provided:
    - early_rf
    - mid_linear

Figure 4c:
  Hidden-knowledge gap on the same confidence buckets:
    gap = (probe_acc - Base_probe_acc) - (llm_acc - Base_llm_acc)
  Positive gap means the probe improved relative to Base more than the surfaced LLM did.
  Two variants are provided:
    - early_rf
    - mid_linear

Why Figures 3b / 4c use buckets instead of quadrants
----------------------------------------------------
Quadrants already encode surfaced correctness, so LLM accuracy inside a quadrant is degenerate:
  Q1/Q2 -> accuracy 1
  Q3/Q4 -> accuracy 0
To keep Figure 3b and 4c apples-to-apples (accuracy vs accuracy), the comparison is done on
confidence buckets (low/mid/high), where surfaced correctness is not built into the partition.

Threshold details
-----------------
raw_base_threshold:
  Base median abs(raw_margin) = 1.500000
  Base tercile 1              = 0.875000
  Base tercile 2              = 2.250000

percentile_to_base:
  Binary threshold            = percentile 0.5
  Bucket thresholds           = percentiles 1/3 and 2/3

Probe definitions
-----------------
early_rf:
  probe_pred = 1 if early_rf_label_prob >= 0.5 else 0

mid_linear:
  probe_pred = 1 if mid_linear_label_score >= 0.0 else 0

Surface definition
------------------
surface_pred = 1 if raw_margin > 0 else 0
surface_correct = (surface_pred == gold_label)

Reproduce
---------
Windows:
  run_generate_hidden_knowledge_dual_confidence_figures.cmd [INPUT_DIR] [OUTPUT_DIR]

Python:
  python generate_hidden_knowledge_dual_confidence_figures.py --input-dir <folder_with_csvs> --output-dir <output_folder>

Required input files
--------------------
  base_test_predictions.csv
  PB_J_test_predictions.csv
  RMU_test_predictions.csv
  TAR_test_predictions.csv
