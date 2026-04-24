#!/usr/bin/env python3
"""
plot_suppressed_emerged.py — Suppressed vs Emerged analysis.

Filters the test set to questions where the probe succeeds BOTH in the base model
and in the unlearned model, then examines generation (logit-based) outcomes:

  - "Suppressed" : base logit correct, method logit wrong
                   → knowledge is behaviorally suppressed but representationally intact
  - "Emerged"    : base logit wrong, method logit correct
                   → rare reversal; unlearning improved generation for this question
  - "Consistent" : both logit correct
                   → unlearning did not change output for this question
  - "Both-wrong" : both logit wrong
                   → generation was never reliable here (probe succeeds independently)

The "both-probe-correct" filter selects questions where hidden knowledge is robustly
detectable regardless of unlearning, making any generation difference truly attributable
to the unlearning procedure (behavioral suppression) rather than initial model noise.

Usage:
    python method_selection/plot_suppressed_emerged.py
    python method_selection/plot_suppressed_emerged.py --out_dir plots
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CHECKPOINTS_DIR = Path("checkpoints")
DATA_DIR = Path("data")

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
METHOD_DISPLAY = {
    "GradDiff": "GradDiff", "RMU": "RMU", "RMU-LAT": "RMU-LAT",
    "RepNoise": "RepNoise", "ELM": "ELM", "RR": "RR",
    "TAR": "TAR", "PB_J": "PB&J",
}
CSV_NAME_MAP = {m: m for m in METHODS}
CSV_NAME_MAP["PB_J"] = "PB_J"   # file is PB_J_bio_logit_test.csv

QUAD_COLORS = {
    "Suppressed":  "#e41a1c",
    "Both-wrong":  "#aaaaaa",
    "Consistent":  "#377eb8",
    "Emerged":     "#4daf4a",
}


def load_y_test() -> np.ndarray:
    df = pd.read_csv(CHECKPOINTS_DIR / "bio_labels.csv")
    return df[df["split"] == "test"]["gold_label"].values


def load_logit_margin(name: str) -> np.ndarray:
    """Load per-question tf_margin. Positive = model predicts True."""
    path = CHECKPOINTS_DIR / f"{name}_bio_logit_test.csv"
    df = pd.read_csv(path)
    return df[df["split"] == "test"]["tf_margin"].values


def load_base_per_layer_probes(clf_name: str = "LR") -> dict:
    with open(CHECKPOINTS_DIR / "base_probes.pkl", "rb") as f:
        bp = pickle.load(f)
    return bp["per_layer"][clf_name]


def load_method_per_layer_probes(method: str, clf_name: str = "LR") -> dict:
    path = CHECKPOINTS_DIR / f"{method}_probes.pkl"
    with open(path, "rb") as f:
        mp = pickle.load(f)
    return mp["per_layer"][clf_name]


def best_layer_scores(probes: dict, hs: np.ndarray, y: np.ndarray
                       ) -> tuple[np.ndarray, float, int]:
    """Return (proba_scores, auc, best_layer) using the layer with highest AUC."""
    best_auc, best_scores, best_layer = -1.0, None, -1
    for layer_idx, clf in probes.items():
        X = hs[:, layer_idx, :]
        try:
            scores = clf.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, scores)
            if auc > best_auc:
                best_auc, best_scores, best_layer = auc, scores, layer_idx
        except Exception:
            pass
    return best_scores, best_auc, best_layer


def probe_correct(scores: np.ndarray, y: np.ndarray, threshold: float = 0.5
                   ) -> np.ndarray:
    """Boolean array: probe prediction matches gold label."""
    pred = (scores >= threshold).astype(int)
    return pred == y.astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="plots")
    ap.add_argument("--clf", default="LR")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading labels and base probes...")
    y_test = load_y_test()
    base_probes = load_base_per_layer_probes(args.clf)

    # Base probe on base hidden states
    base_hs = np.load(CHECKPOINTS_DIR / "base_hs_test.npy")
    base_scores, base_auc, base_layer = best_layer_scores(base_probes, base_hs, y_test)
    base_probe_ok = probe_correct(base_scores, y_test, args.threshold)
    print(f"Base probe: AUC={base_auc:.3f}, acc={base_probe_ok.mean():.3f}, best_layer={base_layer}")

    # Base logit predictions
    base_logit_ok = ((load_logit_margin("base") > 0) == y_test.astype(bool))

    rows = []
    for method in METHODS:
        hs_path = CHECKPOINTS_DIR / f"{method}_hs_test.npy"
        probe_path = CHECKPOINTS_DIR / f"{method}_probes.pkl"
        logit_path = CHECKPOINTS_DIR / f"{method}_bio_logit_test.csv"

        if not (hs_path.exists() and probe_path.exists() and logit_path.exists()):
            print(f"  SKIP {method}: missing files")
            continue

        print(f"\nProcessing {method}...")
        method_hs = np.load(hs_path)
        method_probes = load_method_per_layer_probes(method, args.clf)
        method_scores, method_auc, method_layer = best_layer_scores(
            method_probes, method_hs, y_test)
        method_probe_ok = probe_correct(method_scores, y_test, args.threshold)
        method_logit_ok = ((load_logit_margin(method) > 0) == y_test.astype(bool))

        # Filter: probe correct in BOTH base and method
        both_ok = base_probe_ok & method_probe_ok
        n_both = int(both_ok.sum())
        pct_both = 100.0 * n_both / len(y_test)

        sub_base_logit  = base_logit_ok[both_ok]
        sub_method_logit = method_logit_ok[both_ok]

        n_suppressed = int((sub_base_logit & ~sub_method_logit).sum())
        n_emerged    = int((~sub_base_logit & sub_method_logit).sum())
        n_consistent = int((sub_base_logit  & sub_method_logit).sum())
        n_both_wrong = int((~sub_base_logit & ~sub_method_logit).sum())

        sub_y = y_test[both_ok]
        sub_b_scores = base_scores[both_ok]
        sub_m_scores = method_scores[both_ok]

        def safe_auc(y, s):
            return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")

        print(f"  Method probe AUC={method_auc:.3f} (layer {method_layer})")
        print(f"  Both-probe-correct: {n_both}/{len(y_test)} ({pct_both:.1f}%)")
        print(f"  Suppressed={n_suppressed}, Emerged={n_emerged}, "
              f"Consistent={n_consistent}, Both-wrong={n_both_wrong}")

        rows.append({
            "method":                method,
            "method_probe_auc":      method_auc,
            "n_both_probe_correct":  n_both,
            "pct_both_probe_correct": pct_both,
            "n_suppressed":          n_suppressed,
            "n_emerged":             n_emerged,
            "n_consistent":          n_consistent,
            "n_both_wrong":          n_both_wrong,
            "sub_auc_base_probe":    safe_auc(sub_y, sub_b_scores),
            "sub_auc_method_probe":  safe_auc(sub_y, sub_m_scores),
        })

    df = pd.DataFrame(rows)
    csv_out = os.path.join(args.out_dir, "suppressed_emerged_analysis.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")
    print(df[["method", "n_both_probe_correct", "n_suppressed",
              "n_emerged", "n_consistent", "n_both_wrong"]].to_string(index=False))

    # ── Stacked bar plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    methods_disp = [METHOD_DISPLAY.get(r["method"], r["method"]) for _, r in df.iterrows()]
    x = np.arange(len(df))
    width = 0.55

    ns  = df["n_suppressed"].values
    ne  = df["n_emerged"].values
    nc  = df["n_consistent"].values
    nbw = df["n_both_wrong"].values

    # Left: absolute counts
    ax = axes[0]
    ax.bar(x, ns,             width, label="Suppressed",  color=QUAD_COLORS["Suppressed"],  alpha=0.88)
    ax.bar(x, nbw, width, bottom=ns,         label="Both-wrong",  color=QUAD_COLORS["Both-wrong"],  alpha=0.88)
    ax.bar(x, nc,  width, bottom=ns+nbw,     label="Consistent",  color=QUAD_COLORS["Consistent"],  alpha=0.88)
    ax.bar(x, ne,  width, bottom=ns+nbw+nc,  label="Emerged",     color=QUAD_COLORS["Emerged"],     alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(methods_disp, fontsize=7, rotation=20, ha="right")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Question count", fontsize=8)
    ax.set_title("Absolute counts", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: percentages within doubly-correct set
    ax2 = axes[1]
    total = ns + ne + nc + nbw
    total = np.where(total == 0, 1, total)
    pct_s  = 100 * ns  / total
    pct_bw = 100 * nbw / total
    pct_c  = 100 * nc  / total
    pct_e  = 100 * ne  / total

    ax2.bar(x, pct_s,  width, label="Suppressed",  color=QUAD_COLORS["Suppressed"],  alpha=0.88)
    ax2.bar(x, pct_bw, width, bottom=pct_s,              label="Both-wrong",  color=QUAD_COLORS["Both-wrong"],  alpha=0.88)
    ax2.bar(x, pct_c,  width, bottom=pct_s+pct_bw,       label="Consistent",  color=QUAD_COLORS["Consistent"],  alpha=0.88)
    ax2.bar(x, pct_e,  width, bottom=pct_s+pct_bw+pct_c, label="Emerged",     color=QUAD_COLORS["Emerged"],     alpha=0.88)

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods_disp, fontsize=7, rotation=20, ha="right")
    ax2.tick_params(axis="y", labelsize=7)
    ax2.set_ylabel("% within doubly-correct set", fontsize=8)
    ax2.set_title("Normalized (%)", fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Behavioral Outcomes for Questions Where Probe Succeeds in Both Base and Unlearned Model\n"
        "(filtered to doubly-detectable questions; logit-based generation proxy)",
        fontsize=9,
    )
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        out = os.path.join(args.out_dir, f"suppressed_emerged.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
