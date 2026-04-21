#!/bin/bash
#SBATCH --job-name=hk_method_sel
#SBATCH --output=logs/method_sel_%j.out
#SBATCH --error=logs/method_sel_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=public
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
# CPU only — builds labels, features, and trains classifier CV.

echo "Job started at $(date)"
echo "Node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

OUT=method_selection_out/phase22/joint
mkdir -p logs "${OUT}"

# ── Step 1: Build scientific target labels ────────────────────────────────────
echo ""
echo "Step 1: Building scientific labels ..."
python method_selection/build_scientific_labels_phase22.py \
    --measurements_csv method_selection_out/authoritative_row_measurements.csv \
    --out_dir "${OUT}"

# ── Step 2: Build external features (joint = pair/confusion + logit/margin) ──
echo ""
echo "Step 2: Building joint features (pair + logit) ..."
python method_selection/build_features_phase22.py \
    --pairs_csv data/wmdp_tf_pairs.csv \
    --logits_dir checkpoints \
    --labels_csv "${OUT}/datasets/best_method_labels_by_question_regime.csv" \
    --feature_mode joint \
    --out_dir "${OUT}"

# ── Step 3: Train & evaluate classifiers (all three) ─────────────────────────
echo ""
echo "Step 3a: Logistic regression ..."
python method_selection/train_eval_phase22.py \
    --classifier logreg \
    --out_dir "${OUT}"

echo ""
echo "Step 3b: Random forest ..."
python method_selection/train_eval_phase22.py \
    --classifier rf \
    --out_dir "${OUT}"

echo ""
echo "Step 3c: Gradient boosting ..."
python method_selection/train_eval_phase22.py \
    --classifier gb \
    --out_dir "${OUT}"

echo ""
echo "Method-selection pipeline complete.  Outputs in ${OUT}/"
echo "  Per-classifier results : ${OUT}/reports/{logreg,rf,gb}/overall_summary.csv"
echo "  Fold-level detail      : ${OUT}/reports/{logreg,rf,gb}/fold_results_all_regimes.csv"
echo "  Scored predictions     : ${OUT}/scored/{logreg,rf,gb}/predictions_*.csv"
echo "Job finished at $(date)"
