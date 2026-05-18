#!/usr/bin/env python3
"""
Task F: Method-specific mechanism comparison summary.

Assembles per-method metrics from Tasks C, D, E and the binary classifiers:
  - suppressed / forgotten / retained counts and proportion suppressed
  - suppressed-minus-forgotten K_int traj diff + effect size + sig (Task C)
  - suppressed-minus-retained external drop diff (Task E)
  - classifier ROC-AUC: suppressed vs forgotten (binary_metrics.csv)
  - classifier ROC-AUC: suppressed vs retained (binary_metrics.csv)

Outputs (plots/method_mechanism_summary/):
  method_mechanism_summary.csv
  method_mechanism_summary.md
  method_ranking_plot.pdf / .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "plots" / "method_mechanism_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAJ_BY_Q   = REPO / "plots" / "ckpt_layer_trajectory_metrics" / "trajectory_metrics_by_method_subset.csv"
PUB_TABLE   = REPO / "plots" / "ckpt_layer_trajectory_metrics" / "suppressed_vs_forgotten_publication_table.csv"
MECH_TABLE  = REPO / "plots" / "retained_vs_suppressed_mechanism" / "retained_vs_suppressed_mechanism.csv"
BIN_METRICS = REPO / "plots" / "base_feature_prediction" / "binary_metrics.csv"

METHOD_ORDER = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
N_FILTERED   = 312


def build_summary():
    traj = pd.read_csv(TRAJ_BY_Q)
    pub  = pd.read_csv(PUB_TABLE)
    mech = pd.read_csv(MECH_TABLE)
    bm   = pd.read_csv(BIN_METRICS)

    counts = (
        traj[(traj["layer_band"] == "full") & (traj["subset"].isin(["retained", "suppressed", "forgotten", "lucky"]))]
        .groupby(["method", "subset"])["question_idx"].nunique()
        .unstack("subset", fill_value=0)
        .reset_index()
    )
    for col in ["retained", "suppressed", "forgotten", "lucky"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["prop_suppressed"] = (counts["suppressed"] / N_FILTERED).round(3)

    sf_kint = pub[["method", "diff", "cohens_d", "cliffs_delta", "sig", "q_value"]].rename(columns={
        "diff":         "kint_traj_diff_sf",
        "cohens_d":     "cohens_d_sf",
        "cliffs_delta": "cliffs_delta_sf",
        "sig":          "kint_sig",
        "q_value":      "kint_q",
    })

    ext_drop = mech[mech["metric"] == "ext_drop"][[
        "method", "diff_supp_minus_ret", "sig"
    ]].rename(columns={
        "diff_supp_minus_ret": "ext_drop_diff_sr",
        "sig":                 "ext_drop_sig",
    })

    sf_auc = (bm[bm["task"] == "suppressed_vs_forgotten"]
              [["method", "roc_auc"]].rename(columns={"roc_auc": "clf_auc_sf"}))
    sr_auc = (bm[bm["task"] == "suppressed_vs_retained"]
              [["method", "roc_auc"]].rename(columns={"roc_auc": "clf_auc_sr"}))

    df = counts[["method", "retained", "suppressed", "forgotten", "lucky", "prop_suppressed"]]
    for right in [sf_kint, ext_drop, sf_auc, sr_auc]:
        df = df.merge(right, on="method", how="left")

    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values("method").reset_index(drop=True)

    return df


def write_md(df):
    lines = [
        "# Method Mechanism Summary",
        "",
        "## Counts and suppressed proportion",
        "",
        "| Method | retained | suppressed | forgotten | lucky | prop_supp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['method']} | {r['retained']} | {r['suppressed']} | "
                     f"{r['forgotten']} | {r['lucky']} | {r['prop_suppressed']:.3f} |")

    lines += [
        "",
        "## Suppressed vs Forgotten K_int trajectory gap (full-layer probe)",
        "",
        "| Method | K_int traj diff | Cohen d | Cliff d | sig |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['method']} | {r['kint_traj_diff_sf']:.4f} | {r['cohens_d_sf']:.3f} | "
            f"{r['cliffs_delta_sf']:.3f} | {r['kint_sig']} |"
        )

    lines += [
        "",
        "## Retained vs Suppressed external drop difference",
        "",
        "| Method | Ext drop diff (supp-ret) | sig |",
        "|---|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['method']} | {r['ext_drop_diff_sr']:.4f} | {r['ext_drop_sig']} |")

    lines += [
        "",
        "## Classifier ROC-AUC (base features)",
        "",
        "| Method | Supp vs Forg AUC | Supp vs Ret AUC |",
        "|---|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['method']} | {r['clf_auc_sf']:.3f} | {r['clf_auc_sr']:.3f} |")

    (OUT_DIR / "method_mechanism_summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_ranking(df):
    df_plot = df.sort_values("prop_suppressed", ascending=True).reset_index(drop=True)
    methods = df_plot["method"].tolist()
    y = np.arange(len(methods))

    panels = [
        ("prop_suppressed",   "Proportion suppressed",         "#d62728"),
        ("kint_traj_diff_sf", "Supp-Forg K_int traj diff",     "#1f77b4"),
        ("ext_drop_diff_sr",  "Supp-Ret ext drop diff",        "#ff7f0e"),
        ("clf_auc_sf",        "Clf AUC: supp vs forg",         "#2ca02c"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), sharey=True)
    fig.suptitle("Method comparison — hidden-knowledge strength\n"
                 "(sorted by proportion suppressed)", fontsize=11)

    for ax, (col, title, color) in zip(axes, panels):
        vals = df_plot[col].to_numpy(dtype=float)

        if col == "clf_auc_sf":
            ax.axvline(0.5, color="black", lw=0.8, ls="--", zorder=0)

        ax.barh(y, vals, color=color, alpha=0.75, edgecolor="white")

        sig_col = {"kint_traj_diff_sf": "kint_sig",
                   "ext_drop_diff_sr": "ext_drop_sig"}.get(col)
        if sig_col:
            for i, (v, s) in enumerate(zip(vals, df_plot[sig_col])):
                if s and s not in ("ns", ""):
                    ax.text(v + 0.003, i, s, va="center", ha="left",
                            fontsize=8, fontweight="bold", color="black")

        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    stem = OUT_DIR / "method_ranking_plot"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved: {stem}.pdf / .png")


def main():
    df = build_summary()
    df.to_csv(OUT_DIR / "method_mechanism_summary.csv", index=False)
    print("Saved method_mechanism_summary.csv")

    write_md(df)
    print("Saved method_mechanism_summary.md")

    plot_ranking(df)

    print("\n-- Method Mechanism Summary --")
    pd.set_option("display.float_format", "{:.3f}".format)
    print(df[["method", "suppressed", "forgotten", "prop_suppressed",
              "kint_traj_diff_sf", "ext_drop_diff_sr",
              "clf_auc_sf", "clf_auc_sr"]].to_string(index=False))


if __name__ == "__main__":
    main()
