#!/usr/bin/env python3
"""Generate 4 method-selection result plots from RE/DR and classifier data."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path("/home/yuval.mandel/LLMSecurity/HiddenKnowledgeAfterUnlearning_attempt1")
OUT_DIR = PROJECT / "plots"
RE_DR_ROOT = PROJECT / "method_selection_out" / "re_dr"
MS_ROOT = PROJECT / "method_selection_out" / "phase22" / "joint"

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
LABELS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load per-band RE/DR data (wide format: one row, columns per band) ─────────
re_bands = {b: [] for b in ["early", "mid", "late"]}
dr_bands = {b: [] for b in ["early", "mid", "late"]}
frozen_recovery = []
retrained_recovery = []

for m in METHODS:
    p = RE_DR_ROOT / m / "full_full" / "re_dr_summary.csv"
    if p.exists():
        r = pd.read_csv(p).iloc[0]
        for band in ["early", "mid", "late"]:
            re_col = f"RE_auc_{band}"
            dr_col = f"DR_weight_{band}"
            re_bands[band].append(float(r[re_col]) if re_col in r.index else 0.0)
            dr_bands[band].append(float(r[dr_col]) if dr_col in r.index else 0.0)
        # Recovery from same row
        frz = float(r["frozen_rate_late"]) if "frozen_rate_late" in r.index else 0.5
        ret = float(r["retrain_rate_late"]) if "retrain_rate_late" in r.index else 0.5
        frozen_recovery.append(frz)
        retrained_recovery.append(ret)
    else:
        print(f"WARNING: {p} not found")
        for band in ["early", "mid", "late"]:
            re_bands[band].append(0.0)
            dr_bands[band].append(0.0)
        frozen_recovery.append(0.5)
        retrained_recovery.append(0.5)

# ── Load classifier accuracy data ────────────────────────────────────────────
clf_data = {}
for clf in ["logreg", "rf", "gb"]:
    p = MS_ROOT / "reports" / clf / "overall_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        clf_data[clf] = {
            "accuracy": float(df["mean_accuracy"].iloc[0]),
            "balanced_accuracy": float(df["mean_balanced_accuracy"].iloc[0]),
            "top2_accuracy": float(df["mean_top2_accuracy"].iloc[0]) if "mean_top2_accuracy" in df.columns else 0.0,
        }
    else:
        print(f"WARNING: {p} not found")
        clf_data[clf] = {"accuracy": 0.0, "balanced_accuracy": 0.0, "top2_accuracy": 0.0}

print("RE bands:", {k: [f'{v:.3f}' for v in vs] for k, vs in re_bands.items()})
print("DR bands:", {k: [f'{v:.3f}' for v in vs] for k, vs in dr_bands.items()})
print("Frozen:", [f'{v:.3f}' for v in frozen_recovery])
print("Retrained:", [f'{v:.3f}' for v in retrained_recovery])
print("Classifiers:", clf_data)

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: RE AUC by band — grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
n = len(METHODS)
band_colors = {"early": "#aec6e8", "mid": "#6baed6", "late": "#2171b5"}
x = np.arange(n)
width = 0.25
for i, band in enumerate(["early", "mid", "late"]):
    ax.bar(x + (i - 1) * width, re_bands[band], width,
           label=band.capitalize(), color=band_colors[band], edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Representation Erasure AUC", fontsize=11)
ax.set_title("RE Signal by Layer Band across Unlearning Methods", fontsize=13)
ax.legend(title="Band", fontsize=10)
ymax = max(max(re_bands[b]) for b in ["early", "mid", "late"])
ax.set_ylim(-0.05, ymax * 1.15)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot1_re_by_band.png", dpi=150)
plt.close()
print("Saved plot1_re_by_band.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Scatter — DR_late vs RE_late
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
for i, (m, lab) in enumerate(zip(METHODS, LABELS)):
    ax.scatter(re_bands["late"][i], dr_bands["late"][i],
               color=COLORS[i], s=120, zorder=5)
    ax.annotate(lab, (re_bands["late"][i], dr_bands["late"][i]),
                textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("RE AUC (late band)", fontsize=12)
ax.set_ylabel("DR weight distance (late band)", fontsize=12)
ax.set_title("Deletion Readout vs Representation Erasure\n(late band, 8 methods)", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot2_dr_vs_re_scatter.png", dpi=150)
plt.close()
print("Saved plot2_dr_vs_re_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Frozen vs Retrained recovery rate (late band)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(n)
w = 0.35
ax.bar(x - w/2, frozen_recovery, w, label="Frozen probe (no retrain)", color="#fc8d59", edgecolor="white")
ax.bar(x + w/2, retrained_recovery, w, label="Retrained probe", color="#d7191c", edgecolor="white")
ax.axhline(0.5, color="gray", linewidth=1, linestyle="--", label="Chance (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Recovery accuracy (late band)", fontsize=11)
ax.set_title("Knowledge Recovery: Frozen vs Retrained Probe (Late Band)", fontsize=13)
ax.legend(fontsize=10)
all_vals = frozen_recovery + retrained_recovery
ax.set_ylim(min(all_vals) * 0.9, max(all_vals) * 1.1)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot3_recovery_late_band.png", dpi=150)
plt.close()
print("Saved plot3_recovery_late_band.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: Method-selection classifier accuracy
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
clf_names = ["LogReg", "Random\nForest", "Gradient\nBoosting"]
acc  = [clf_data[c]["accuracy"]          for c in ["logreg", "rf", "gb"]]
bacc = [clf_data[c]["balanced_accuracy"] for c in ["logreg", "rf", "gb"]]
top2 = [clf_data[c]["top2_accuracy"]     for c in ["logreg", "rf", "gb"]]
x = np.arange(3)
w = 0.25
ax.bar(x - w,  acc,  w, label="Top-1 accuracy",    color="#2166ac", edgecolor="white")
ax.bar(x,      bacc, w, label="Balanced accuracy",  color="#4dac26", edgecolor="white")
ax.bar(x + w,  top2, w, label="Top-2 accuracy",     color="#b2df8a", edgecolor="white")
ax.axhline(1/8, color="red",    linewidth=1.5, linestyle="--", label="Random top-1 (12.5%)")
ax.axhline(2/8, color="orange", linewidth=1.0, linestyle=":",  label="Random top-2 (25%)")
ax.set_xticks(x)
ax.set_xticklabels(clf_names, fontsize=11)
ax.set_ylabel("Cross-validation accuracy", fontsize=11)
ax.set_title("Method-Selection Classifier Performance\n(5-fold GroupKFold, 8 classes)", fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, max(top2) * 1.3 + 0.05)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot4_classifier_accuracy.png", dpi=150)
plt.close()
print("Saved plot4_classifier_accuracy.png")

print("\nAll 4 plots saved to", OUT_DIR)
