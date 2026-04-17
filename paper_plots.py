"""
Generate paper-quality plots for the Hidden Knowledge After Unlearning paper.
Run from repo root on the Newton server: python paper_plots.py
Output: latex/imgs/*.pdf  (and *.png for preview)
"""

import csv
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

DATA  = Path("data")
OUT   = Path("latex/imgs")
OUT.mkdir(parents=True, exist_ok=True)

# ── colour / label conventions ────────────────────────────────────────────────

METHOD_ORDER = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]
METHOD_COLORS = {
    "GradDiff": "#e41a1c", "RMU": "#377eb8", "RMU-LAT": "#4daf4a",
    "RepNoise": "#984ea3", "ELM": "#ff7f00", "RR": "#a65628",
    "TAR": "#f781bf", "PB&J": "#999999",
    "Base": "#2ca02c",
}
TEX_LABELS = {
    "GradDiff": "GradDiff", "RMU": "RMU", "RMU-LAT": "RMU-LAT",
    "RepNoise": "RepNoise", "ELM": "ELM", "RR": "RR",
    "TAR": "TAR", "PB&J": "PB\\&J",
    "Base": "Base (8B-I)", "base": "Base (8B-I)", "Llama3-8B": "Llama-3-8B",
}

def load_csv(name):
    rows = {}
    p = DATA / name
    if not p.exists():
        print(f"  [WARN] missing {p}"); return rows
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["method"]] = row
    return rows

def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {stem}.pdf / .png")

# ── Plot 1: Gap plot — GenAcc vs Probe AUC ────────────────────────────────────

def plot_gap():
    gen  = load_csv("summary_table1_gen_logit.csv")
    prob = load_csv("summary_table3_method_probes.csv")

    models  = ["Base"] + METHOD_ORDER
    gen_acc = []
    pba_auc = []

    for m in models:
        gkey = m
        pkey = m if m != "Base" else None
        g = gen.get(gkey, {})
        # use gen_valid_acc (excludes gibberish)
        ga = float(g.get("gen_valid_acc", g.get("gen_acc", 0.5)))
        gen_acc.append(ga)
        if pkey is None:
            # base model's own probe AUC comes from per-layer data
            pba_auc.append(0.703)   # from summary_table2_base_probes Base row pl_lr_auc
        else:
            p = prob.get(pkey, {})
            auc = float(p.get("mb_lr_auc", p.get("pl_lr_auc", 0.5)))
            pba_auc.append(auc)

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))

    bars1 = ax.bar(x - w/2, gen_acc,  w, label="GenAcc (valid)", color="#4878CF", alpha=0.85)
    bars2 = ax.bar(x + w/2, pba_auc, w, label="Probe AUC (ML-LR)", color="#D65F5F", alpha=0.85)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([TEX_LABELS.get(m, m) for m in models], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Behavioral vs. Representational Unlearning Gap (WMDP-bio)")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    save(fig, "gap_gen_vs_probe_auc")

# ── Plot 2: AUC delta heatmap (Base − Method, per layer, LR) ─────────────────

def plot_auc_delta_heatmap():
    rows = {}
    p = DATA / "kfold_per_layer_aggregated.csv"
    if not p.exists(): print(f"  [WARN] missing {p}"); return
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["clf"] != "LR": continue
            model = row["model"]
            layer = int(row["layer"])
            auc   = float(row["mean_auc"])
            rows.setdefault(model, {})[layer] = auc

    base_auc = rows.get("base", {})
    n_layers = 33
    methods  = [m for m in METHOD_ORDER if m in rows]

    delta = np.zeros((len(methods), n_layers))
    for mi, m in enumerate(methods):
        for l in range(n_layers):
            b = base_auc.get(l, np.nan)
            v = rows[m].get(l, np.nan)
            delta[mi, l] = b - v   # positive = base is better (method erased more)

    fig, ax = plt.subplots(figsize=(13, 4))
    norm = TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.15)
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", norm=norm,
                   origin="upper", interpolation="nearest")
    ax.set_xticks(range(0, n_layers, 4))
    ax.set_xticklabels(range(0, n_layers, 4))
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([TEX_LABELS.get(m, m) for m in methods])
    ax.set_xlabel("Layer")
    ax.set_title("AUC Delta: Base $-$ Method (LR probe, WMDP-bio)\nRed = method reduced AUC")
    fig.colorbar(im, ax=ax, label="ΔAUC")
    # highlight mid-band
    for x_val in [11.5, 22.5]:
        ax.axvline(x_val, color="black", linewidth=1.2, linestyle="--")
    fig.tight_layout()
    save(fig, "auc_delta_heatmap_base_minus_method")

# ── Plot 3: Cross-probe accuracy heatmap ─────────────────────────────────────

def plot_cross_probe_heatmap():
    b2m = load_csv("summary_table2_base_probes.csv")   # base→method
    m2b = load_csv("summary_table5_cross_probes.csv")  # method→base

    methods = METHOD_ORDER
    # direction labels
    row_labels = [f"Base→{TEX_LABELS.get(m,m)}" for m in methods] + \
                 [f"{TEX_LABELS.get(m,m)}→Base" for m in methods]

    accs = []
    for m in methods:
        accs.append(float(b2m.get(m, {}).get("pl_lr_acc", 0.5)))
    for m in methods:
        accs.append(float(m2b.get(m, {}).get("pl_lr_acc", 0.5)))

    fig, ax = plt.subplots(figsize=(5, 7))
    colors = ["#c0392b" if a < 0.53 else "#2980b9" if a > 0.60 else "#f39c12"
              for a in accs]
    bars = ax.barh(range(len(accs)), accs, color=colors, alpha=0.85)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.9)
    ax.set_yticks(range(len(accs)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Transfer Accuracy")
    ax.set_xlim(0.44, 0.72)
    ax.set_title("Cross-Probe Transfer (Best-Layer LR)")
    ax.invert_yaxis()
    # annotate values
    for i, (bar, a) in enumerate(zip(bars, accs)):
        ax.text(a + 0.002, i, f"{a:.3f}", va="center", fontsize=7)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    save(fig, "cross_probe_transfer_bar")

# ── Plot 4: Bio vs Cyber probe AUC comparison ─────────────────────────────────

def plot_bio_cyber_comparison():
    bio  = load_csv("summary_table3_method_probes.csv")
    cyb  = load_csv("summary_table4c_cyber_method_probes.csv")

    methods   = METHOD_ORDER
    bio_aucs  = [float(bio.get(m, {}).get("mb_lr_auc", 0.5)) for m in methods]
    cyb_aucs  = [float(cyb.get(m, {}).get("mb_lr_auc", 0.5)) for m in methods]

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, bio_aucs, w, label="Bio (forget set)", color="#e74c3c", alpha=0.85)
    ax.bar(x + w/2, cyb_aucs, w, label="Cyber (retain set)", color="#3498db", alpha=0.85)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([TEX_LABELS.get(m, m) for m in methods], rotation=30, ha="right")
    ax.set_ylabel("Probe AUC (ML-LR, layers 12–22)")
    ax.set_ylim(0.45, 0.85)
    ax.set_title("Bio vs. Cyber Probe AUC After Unlearning")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    save(fig, "bio_vs_cyber_probe_auc")

# ── Plot 5: Generation breakdown (True / False / Gibberish) ──────────────────

def plot_gen_breakdown():
    gen = load_csv("summary_table1_gen_logit.csv")
    models = ["Base"] + METHOD_ORDER
    labels = [TEX_LABELS.get(m, m) for m in models]

    correct_T, correct_F, wrong_T, wrong_F, gib = [], [], [], [], []
    for m in models:
        g = gen.get(m, {})
        gen_acc   = float(g.get("gen_acc", 0))
        gen_true  = float(g.get("gen_true", 0))  # fraction of True that are correct
        gen_false = float(g.get("gen_false", 0))
        gibberish = float(g.get("gibberish", 0))
        valid     = 1 - gibberish
        # Estimate counts (dataset is balanced 50/50)
        ct = 0.5 * gen_true * valid   # correct True
        cf = 0.5 * gen_false * valid  # correct False
        wt = 0.5 * (1 - gen_false) * valid  # wrong (said True when False)
        wf = 0.5 * (1 - gen_true) * valid   # wrong (said False when True)
        correct_T.append(ct); correct_F.append(cf)
        wrong_T.append(wt);   wrong_F.append(wf)
        gib.append(gibberish)

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(models))
    for vals, label, color in [
        (correct_T, "Correct True",  "#27ae60"),
        (correct_F, "Correct False", "#2ecc71"),
        (wrong_T,   "Wrong (True→F)","#e74c3c"),
        (wrong_F,   "Wrong (F→True)","#c0392b"),
        (gib,       "Gibberish",     "#bdc3c7"),
    ]:
        ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.9)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Generation Outcome Breakdown (WMDP-bio)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    save(fig, "gen_breakdown_stacked")

# ── Plot 6: Per-layer AUC lines (all methods, LR) ─────────────────────────────

def plot_per_layer_lines():
    rows = {}
    p = DATA / "kfold_per_layer_aggregated.csv"
    if not p.exists(): print(f"  [WARN] missing {p}"); return
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["clf"] != "LR": continue
            model = row["model"]
            layer = int(row["layer"])
            auc   = float(row["mean_auc"])
            std   = float(row["std_auc"])
            rows.setdefault(model, {}).setdefault("auc", {})[layer] = auc
            rows.setdefault(model, {}).setdefault("std", {})[layer] = std

    fig, ax = plt.subplots(figsize=(10, 5))
    n_layers = 33
    xs = list(range(n_layers))

    # base first
    if "base" in rows:
        aucs = [rows["base"]["auc"].get(l, np.nan) for l in xs]
        stds = [rows["base"]["std"].get(l, 0) for l in xs]
        ax.plot(xs, aucs, color="black", linewidth=2.2, label="Base (8B-I)", zorder=5)
        ax.fill_between(xs,
                        [a - s for a, s in zip(aucs, stds)],
                        [a + s for a, s in zip(aucs, stds)],
                        alpha=0.12, color="black")

    for m in METHOD_ORDER:
        if m not in rows: continue
        aucs = [rows[m]["auc"].get(l, np.nan) for l in xs]
        stds = [rows[m]["std"].get(l, 0) for l in xs]
        c = METHOD_COLORS.get(m, "grey")
        ax.plot(xs, aucs, color=c, linewidth=1.4, label=TEX_LABELS.get(m, m))
        ax.fill_between(xs,
                        [a - s for a, s in zip(aucs, stds)],
                        [a + s for a, s in zip(aucs, stds)],
                        alpha=0.07, color=c)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axvspan(12, 22, alpha=0.06, color="blue", label="Mid band (12–22)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("LR Probe AUC (5-fold CV)")
    ax.set_xlim(0, 32)
    ax.set_ylim(0.48, 0.82)
    ax.set_title("Per-Layer LR Probe AUC on WMDP-bio")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    save(fig, "per_layer_auc_lines_all_methods")

# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating paper plots...")
    plot_gap()
    plot_auc_delta_heatmap()
    plot_cross_probe_heatmap()
    plot_bio_cyber_comparison()
    plot_gen_breakdown()
    plot_per_layer_lines()
    print(f"Done. Output in {OUT}/")
