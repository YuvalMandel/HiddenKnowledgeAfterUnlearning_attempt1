"""
Checkpoint trajectory line plots per method — Q*, best_layer, v6 KFold.

One PDF/PNG per method. 8 lines (4 subsets × 2 metrics):
  Color encodes subset: retained=green, suppressed=red, forgotten=grey, lucky=blue
  Line style: K_ext = solid, K_int = dotted

X-axis: base, ck1 … ck8.
At base, all Q* questions have K_int = K_ext = 1 by definition.
Subsets are labelled at ck8. Shaded band = ±1 SEM.

Output: plots/v6_trajectory/<method>.pdf and .png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR  = Path(__file__).parent.parent / "inside_out_out"
SAVE_DIR = Path(__file__).parent / "v6_trajectory"
SAVE_DIR.mkdir(exist_ok=True)

KNOWS = 0.5

METHODS = [
    ("GradDiff", "GradDiff"),
    ("PB_J",     "PB&J"),
    ("RMU",      "RMU"),
    ("RMU-LAT",  "RMU-LAT"),
    ("RepNoise", "RepNoise"),
    ("ELM",      "ELM"),
    ("RR",       "RR"),
    ("TAR",      "TAR"),
]
CKPTS     = [f"ck{i}" for i in range(1, 9)]
CKPT_TICKS = ["base"] + CKPTS

SUBSET_COLORS = {
    "retained":  "#2ca02c",
    "suppressed":"#d62728",
    "forgotten": "#888888",
    "lucky":     "#1f77b4",
}


def load_cv_bl(model_id: str) -> pd.DataFrame:
    p = OUT_DIR / model_id / "k_scores.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    return df[
        (df["split_type"] == "cv") & (df["domain"] == "bio") &
        (df["clf"] == "LR") & (df["probe_type"] == "own") &
        (df["layer_config"] == "best_layer")
    ][["question_idx", "k_internal", "k_external"]].copy()


# ── Q* from base ──────────────────────────────────────────────────────────────
base_cv = load_cv_bl("base")
qstar   = set(base_cv.loc[
    (base_cv["k_internal"] == 1.0) & (base_cv["k_external"] == 1.0),
    "question_idx"
])
print(f"Q* size: {len(qstar)}")


def mean_sem(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else (float(a.mean()), 0.0)


# ── One figure per method ─────────────────────────────────────────────────────
for mname, label in METHODS:
    print(f"  {label} ...", end=" ", flush=True)

    # Subset labels at ck8
    ck8 = load_cv_bl(f"{mname}_ck8")
    if ck8.empty:
        print("SKIP (missing ck8)"); continue
    ck8_q = ck8[ck8["question_idx"].isin(qstar)].set_index("question_idx")

    subset_map = {}
    for qi, row in ck8_q.iterrows():
        ki, ke = row["k_internal"], row["k_external"]
        if   ki > KNOWS and ke > KNOWS:   subset_map[qi] = "retained"
        elif ki > KNOWS and ke <= KNOWS:  subset_map[qi] = "suppressed"
        elif ki <= KNOWS and ke <= KNOWS: subset_map[qi] = "forgotten"
        else:                             subset_map[qi] = "lucky"

    subset_qs = {s: {qi for qi, sub in subset_map.items() if sub == s}
                 for s in SUBSET_COLORS}

    # Build trajectory dict: subset → metric → list of (mean, sem) per ck
    # Index 0 = base (K_int = K_ext = 1 for all Q* by definition)
    SUBSETS = list(SUBSET_COLORS.keys())
    METRICS = ["k_internal", "k_external"]
    traj = {(s, m): [(1.0, 0.0)] for s in SUBSETS for m in METRICS}

    for ck in CKPTS:
        df_ck = load_cv_bl(f"{mname}_{ck}")
        if df_ck.empty:
            for key in traj: traj[key].append((float("nan"), float("nan")))
            continue
        df_ck = df_ck.set_index("question_idx")
        for s in SUBSETS:
            for m in METRICS:
                vals = [float(df_ck.loc[qi, m])
                        for qi in subset_qs[s] if qi in df_ck.index]
                traj[(s, m)].append(mean_sem(vals))

    x = np.arange(len(CKPT_TICKS))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for s in SUBSETS:
        color = SUBSET_COLORS[s]
        n_s   = len(subset_qs[s])
        if n_s == 0:
            continue
        for m, ls, metric_lbl in [
            ("k_external", "-",  "ext"),
            ("k_internal", ":", "int"),
        ]:
            means = np.array([v[0] for v in traj[(s, m)]])
            sems  = np.array([v[1] for v in traj[(s, m)]])
            lbl   = rf"{s.capitalize()} $K_\mathrm{{{metric_lbl}}}$ (n={n_s})"
            ax.plot(x, means * 100, color=color, linestyle=ls, linewidth=2.0,
                    marker="o", markersize=3.5, label=lbl, zorder=4)
            ax.fill_between(x, (means - sems) * 100, (means + sems) * 100,
                            color=color, alpha=0.10, zorder=2)

    ax.axhline(50, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(CKPT_TICKS, fontsize=8.5)
    ax.set_ylabel(r"Mean pairwise $K$ score (%)", fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.0, loc="lower left", framealpha=0.88,
              ncol=2, columnspacing=0.8, handlelength=2.0)

    fig.tight_layout()
    stem = mname.replace("-", "_")
    fig.savefig(SAVE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(SAVE_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("done")

print(f"\nSaved to: {SAVE_DIR}/")
