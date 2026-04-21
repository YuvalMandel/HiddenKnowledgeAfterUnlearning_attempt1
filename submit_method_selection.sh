#!/bin/bash
# submit_method_selection.sh
#
# Submit the full method-selection pipeline as dependent SLURM jobs.
#
# Prerequisites: The main pipeline (submit_pipeline.sh) must have completed
# successfully so that the following files exist:
#   checkpoints/base_hs_{train,val,test}.npy
#   checkpoints/{METHOD}_hs_{train,val,test}.npy  (for all 8 methods)
#   checkpoints/base_bio_logit_{train,val,test}.csv
#   checkpoints/{METHOD}_bio_logit_{train,val,test}.csv
#   data/wmdp_tf_pairs.csv
#
# Pipeline stages:
#   Stage A  — export_labels    : create checkpoints/bio_labels.csv (fast, CPU)
#   Stage B  — re_dr array      : RE/DR band analysis for each of 8 methods (CPU, array 0-7)
#   Stage C  — re_dr_summary    : aggregate band predictions → authoritative_row_measurements.csv
#   Stage D  — method_selection : build labels → features → 3 classifiers → report
#
# Usage:
#   bash submit_method_selection.sh
#   bash submit_method_selection.sh --skip-export   # if bio_labels.csv already exists
#
# Outputs land in method_selection_out/:
#   re_dr/{METHOD}/full_full/               — scatter plots, band CSVs, recovery CSVs
#   authoritative_row_measurements.csv      — per-(question, method) erasure metrics
#   phase22/joint/datasets/                 — feature & label datasets
#   phase22/joint/models/                   — CV fold results, OOF predictions
#   phase22/joint/reports/                  — accuracy, confusion, method distribution

set -e

mkdir -p logs

SKIP_EXPORT=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-export) SKIP_EXPORT=true ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Stage A: export bio_labels.csv ────────────────────────────────────────────
if $SKIP_EXPORT; then
    echo "Skipping export stage (--skip-export)."
    EXPORT_DEP=""
else
    echo "Submitting export stage (creates checkpoints/bio_labels.csv) ..."
    EXPORT_JOB=$(sbatch --parsable --job-name=hk_export_labels \
        --output=logs/export_labels_%j.out \
        --error=logs/export_labels_%j.err \
        --time=00:10:00 \
        --partition=public \
        --mem=4G \
        --cpus-per-task=1 \
        --wrap="source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate unlearning && python method_selection/export_bio_labels_csv.py")
    echo "  Export job ID: ${EXPORT_JOB}"
    EXPORT_DEP="--dependency=afterok:${EXPORT_JOB}"
fi

# ── Stage B: RE/DR array (8 methods, depends on export) ───────────────────────
echo "Submitting RE/DR array (8 methods) ..."
RE_DR_JOB=$(sbatch --parsable ${EXPORT_DEP} slurm_re_dr.sh)
echo "  RE/DR array job ID: ${RE_DR_JOB}  (tasks 0–7)"

# ── Stage C: aggregate → authoritative_row_measurements.csv ───────────────────
echo "Submitting RE/DR summary (depends on all RE/DR tasks) ..."
SUMMARY_JOB=$(sbatch --parsable --dependency=afterok:${RE_DR_JOB} slurm_re_dr_summary.sh)
echo "  RE/DR summary job ID: ${SUMMARY_JOB}"

# ── Stage D: method selection (labels → features → CV → report) ───────────────
echo "Submitting method-selection pipeline (depends on summary) ..."
MS_JOB=$(sbatch --parsable --dependency=afterok:${SUMMARY_JOB} slurm_method_selection.sh)
echo "  Method-selection job ID: ${MS_JOB}"

echo ""
echo "Pipeline submitted successfully:"
if ! $SKIP_EXPORT; then
    echo "  Stage A — export labels : ${EXPORT_JOB}"
fi
echo "  Stage B — RE/DR array   : ${RE_DR_JOB}  (tasks 0–7, one per method)"
echo "  Stage C — RE/DR summary : ${SUMMARY_JOB}"
echo "  Stage D — method select : ${MS_JOB}"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo ""
echo "Key outputs:"
echo "  Scatter plots  : method_selection_out/re_dr/{METHOD}/full_full/scatter_*.png"
echo "  Recovery CSV   : method_selection_out/re_dr/{METHOD}/full_full/recovery_by_band.csv"
echo "  Measurements   : method_selection_out/authoritative_row_measurements.csv"
echo "  CV accuracy    : method_selection_out/phase22/joint/reports/overall_summary.csv"
echo "  Method counts  : method_selection_out/phase22/joint/reports/predicted_method_counts.csv"
echo "  Full report    : method_selection_out/phase22/joint/reports/phase22_report.md"
