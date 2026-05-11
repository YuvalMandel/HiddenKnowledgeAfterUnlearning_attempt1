#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "plots/predict_subset_from_base_features.py",
    "plots/quantify_ckpt_layer_trajectories.py",
    "plots/plot_suppressed_minus_forgotten_heatmap.py",
    "plots/plot_internal_external_trajectory_by_subset.py",
    "plots/base_features_vs_trajectory.py",
]


def run_script(script_rel):
    script_path = REPO / script_rel
    print(f"\n=== Running: {script_rel} ===", flush=True)
    proc = subprocess.run([sys.executable, str(script_path)], cwd=str(REPO))
    if proc.returncode != 0:
        raise RuntimeError(f"Script failed ({proc.returncode}): {script_rel}")
    print(f"=== Done: {script_rel} ===", flush=True)


def main():
    print("Starting hidden-knowledge analysis pipeline...", flush=True)
    print(f"Repo: {REPO}", flush=True)
    for script in SCRIPTS:
        run_script(script)
    print("\nAll analysis scripts completed successfully.", flush=True)


if __name__ == "__main__":
    main()
