#!/bin/bash
# Submit the dual-confidence pipeline:
#   Step 1: job array — generate one prediction CSV per method (9 jobs in parallel)
#   Step 2: single job — generate all figures (waits for step 1 to finish)

set -e

echo "Submitting dual-confidence CSV generation (array 0-8)..."
GEN_JOB=$(sbatch --parsable slurm_dual_conf_gen.sh)
echo "  gen job array ID: ${GEN_JOB}"

echo "Submitting figure generation (depends on array ${GEN_JOB})..."
PLOT_JOB=$(sbatch --parsable --dependency=afterok:${GEN_JOB} slurm_dual_conf_plot.sh)
echo "  plot job ID: ${PLOT_JOB}"

echo ""
echo "Done. Monitor with:"
echo "  squeue -j ${GEN_JOB},${PLOT_JOB}"
echo "  tail -f logs/dual_conf_gen_${GEN_JOB}_*.out"
echo "  tail -f logs/dual_conf_plot_${PLOT_JOB}.out"
