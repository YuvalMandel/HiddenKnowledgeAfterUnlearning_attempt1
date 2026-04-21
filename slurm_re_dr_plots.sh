#!/bin/bash
#SBATCH --job-name=hk_re_dr_plots
#SBATCH --output=logs/re_dr_plots_%j.out
#SBATCH --error=logs/re_dr_plots_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=public
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
# CPU only — reads pre-computed re_dr_summary.csv files, generates plots and ranking.
# Submit AFTER all 8 re_dr array tasks finish:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm_re_dr_plots.sh

echo "Job started at $(date)"
echo "Node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

mkdir -p logs plots method_selection_out/recovery_analysis

# ── Step 1: Generate the 2 scatter plots ─────────────────────────────────────
echo ""
echo "Step 1: Generating RE/DR scatter plots ..."
python method_selection/plot_re_dr_scatter.py \
    --re_dr_root method_selection_out/re_dr \
    --train_scheme full \
    --train_window_id full \
    --subset_mode none \
    --eval_split test \
    --x_expr DR_late_minus_mid \
    --y_expr RE_mid_minus_early \
    --re_metric auc \
    --out_dir plots

# ── Step 2: Method ranking by recovery ───────────────────────────────────────
echo ""
echo "Step 2: Building method recovery ranking ..."
python method_selection/analyze_recovery.py \
    --re_dr_root method_selection_out/re_dr \
    --train_scheme full \
    --train_window_id full \
    --subset_mode none \
    --eval_split test \
    --out_dir method_selection_out/recovery_analysis

echo ""
echo "Job finished at $(date)"
echo "Scatter plots : plots/scatter_{weightDR,scoreDR}_*.png"
echo "Ranking tables: method_selection_out/recovery_analysis/"
