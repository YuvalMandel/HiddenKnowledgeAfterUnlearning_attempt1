"""
Internal vs external suppression lag — best_layer probe, v6 KFold, Q*.

For each Q* question in each method, finds:
  ck_ext: first checkpoint where K_ext transitions ≤ 0.5 (external suppression)
  ck_int: first checkpoint where K_int transitions ≤ 0.5 (internal erasure)
  9 = "never within ck1–ck8"

Plots paired box/violin: ck_ext vs ck_int distributions per method,
broken down by final subset (suppressed / forgotten).
Outputs: v6_suppression_lag.csv, v6_suppression_lag.pdf/.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR  = Path(__file__).parent.parent / "inside_out_out"
SAVE_DIR = Path(__file__).parent
KNOWS    = 0.5
NEVER    = 9   # sentinel: did not drop within ck1–ck8

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
CKPTS = [f"ck{i}" for i in range(1, 9)]


def load_cv_bl(model_id: str) -> pd.DataFrame:
    p = OUT_DIR / model_id / "k_scores.parquet"
    if not p.exists():
        print(f"  Missing: {p}"); return pd.DataFrame()
    df = pd.read_parquet(p)
    return df[
        (df["split_type"] == "cv") & (df["domain"] == "bio") &
        (df["clf"] == "LR") & (df["probe_type"] == "own") &
        (df["layer_config"] == "best_layer")
    ][["model_id", "question_idx", "k_internal", "k_external"]].copy()


# ── Q* from base ──────────────────────────────────────────────────────────────
base_cv = load_cv_bl("base")
qstar   = set(base_cv.loc[
    (base_cv["k_internal"] == 1.0) & (base_cv["k_external"] == 1.0),
    "question_idx"
])
print(f"Q* size: {len(qstar)}")

# ── Per-question trajectory + lag ─────────────────────────────────────────────
all_rows = []

for mname, label in METHODS:
    # Subset labels at ck8
    ck8_df = load_cv_bl(f"{mname}_ck8")
    if ck8_df.empty: continue
    ck8_q  = ck8_df[ck8_df["question_idx"].isin(qstar)].set_index("question_idx")
    subset_map = {}
    for qi, row in ck8_q.iterrows():
        ki, ke = row["k_internal"], row["k_external"]
        if   ki > KNOWS and ke > KNOWS:  subset_map[qi] = "retained"
        elif ki > KNOWS and ke <= KNOWS: subset_map[qi] = "suppressed"
        elif ki <= KNOWS and ke > KNOWS: subset_map[qi] = "lucky"
        else:                            subset_map[qi] = "forgotten"

    # Collect K_int and K_ext at each checkpoint per question
    ki_traj = {qi: [] for qi in subset_map}
    ke_traj = {qi: [] for qi in subset_map}

    for ck_i, ck in enumerate(CKPTS, start=1):
        ck_df = load_cv_bl(f"{mname}_{ck}")
        if ck_df.empty: continue
        ck_q = ck_df[ck_df["question_idx"].isin(subset_map)].set_index("question_idx")
        for qi in subset_map:
            if qi in ck_q.index:
                ki_traj[qi].append((ck_i, float(ck_q.loc[qi, "k_internal"])))
                ke_traj[qi].append((ck_i, float(ck_q.loc[qi, "k_external"])))

    for qi, subset in subset_map.items():
        # First ck where K_ext drops ≤ KNOWS
        ck_ext_drop = NEVER
        for ck_i, ke in ke_traj.get(qi, []):
            if ke <= KNOWS:
                ck_ext_drop = ck_i; break

        # First ck where K_int drops ≤ KNOWS
        ck_int_drop = NEVER
        for ck_i, ki in ki_traj.get(qi, []):
            if ki <= KNOWS:
                ck_int_drop = ck_i; break

        all_rows.append({
            "method":      label,
            "question_idx": qi,
            "subset":      subset,
            "ck_ext_drop": ck_ext_drop,
            "ck_int_drop": ck_int_drop,
            "lag":         ck_int_drop - ck_ext_drop,  # positive = K_int lags behind K_ext
        })

df_lag = pd.DataFrame(all_rows)
df_lag.to_csv(SAVE_DIR / "v6_suppression_lag.csv", index=False)
print(f"Saved: v6_suppression_lag.csv")

# ── Plot: grouped horizontal box plots per method ─────────────────────────────
# Three series per method:
#   (1) suppressed  — K_ext first drop  [red]
#   (2) forgotten   — K_ext first drop  [dark grey]
#   (3) forgotten   — K_int first drop  [light grey]
# K_int first drop for suppressed is annotated inline (mostly "never").

methods_order = [l for _, l in METHODS if l in df_lag["method"].unique()]
n_m   = len(methods_order)
STEP  = 1.0       # vertical spacing between methods
OFFS  = [+0.27, 0.0, -0.27]   # y offsets for the 3 series within a method row
BOX_H = 0.18      # half-height of each box

SERIES = [
    ("suppressed", "ck_ext_drop", "#d62728", r"$K_\mathrm{ext}$ drop · Suppressed"),
    ("forgotten",  "ck_ext_drop", "#555555", r"$K_\mathrm{ext}$ drop · Forgotten"),
    ("forgotten",  "ck_int_drop", "#aaaaaa", r"$K_\mathrm{int}$ drop · Forgotten"),
]

fig, ax = plt.subplots(figsize=(8, 0.85 * n_m + 1.6))

for mi, meth in enumerate(methods_order):
    y_base = (n_m - 1 - mi) * STEP
    sub    = df_lag[df_lag["method"] == meth]

    for (subset, col, color, _), off in zip(SERIES, OFFS):
        vals = sub.loc[sub["subset"] == subset, col].values
        if len(vals) == 0: continue
        y_c  = y_base + off

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.broken_barh([(q1, q3 - q1)], (y_c - BOX_H, 2 * BOX_H),
                       facecolors=color, alpha=0.55, zorder=3)
        ax.plot([med, med], [y_c - BOX_H, y_c + BOX_H],
                color="black", linewidth=1.5, zorder=4)
        # whiskers to 5th / 95th percentile
        p5, p95 = np.percentile(vals, [5, 95])
        ax.plot([p5, q1],   [y_c, y_c], color=color, linewidth=1.2, zorder=3)
        ax.plot([q3, p95],  [y_c, y_c], color=color, linewidth=1.2, zorder=3)

    # Annotate % K_int "never" for suppressed
    supp_ki = sub.loc[sub["subset"] == "suppressed", "ck_int_drop"].values
    if len(supp_ki):
        pct_never = 100 * (supp_ki == NEVER).mean()
        ax.text(NEVER + 0.08, y_base + OFFS[0],
                f"{pct_never:.0f}% K_int never↓",
                va="center", ha="left", fontsize=6.5, color="#d62728", style="italic")

ax.set_yticks([( n_m - 1 - mi) * STEP for mi in range(n_m)])
ax.set_yticklabels(methods_order, fontsize=9)
ax.set_xlim(0.5, NEVER + 1.6)
ax.set_xticks(range(1, NEVER + 1))
ax.set_xticklabels([f"ck{i}" for i in range(1, NEVER)] + ["never"], fontsize=8)
ax.axvline(NEVER, color="black", linewidth=0.8, linestyle="--", alpha=0.35)
ax.set_xlabel("First checkpoint where K ≤ 0.5  (box = IQR, line = median, whiskers = 5–95th pct)",
              fontsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.45, zorder=0)
ax.set_axisbelow(True)

legend_patches = [
    plt.Rectangle((0,0), 1, 1, color=color, alpha=0.7, label=lbl)
    for (_, _, color, lbl) in SERIES
]
ax.legend(handles=legend_patches, fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig(SAVE_DIR / "v6_suppression_lag.pdf", bbox_inches="tight")
fig.savefig(SAVE_DIR / "v6_suppression_lag.png", dpi=150, bbox_inches="tight")
print(f"Saved: v6_suppression_lag.pdf / .png")
plt.close(fig)

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"\n{'Method':<12} {'Subset':<12} {'n':>5}  {'med_ext':>7} {'med_int':>7}  {'never_ext%':>10} {'never_int%':>10}")
print("-" * 72)
for meth in methods_order:
    for subset in ["suppressed", "forgotten"]:
        sub = df_lag[(df_lag["method"] == meth) & (df_lag["subset"] == subset)]
        if sub.empty: continue
        n = len(sub)
        med_e = int(np.median(sub["ck_ext_drop"]))
        med_i = int(np.median(sub["ck_int_drop"]))
        pct_e = 100 * (sub["ck_ext_drop"] == NEVER).mean()
        pct_i = 100 * (sub["ck_int_drop"] == NEVER).mean()
        print(f"{meth:<12} {subset:<12} {n:>5}  {med_e:>7}  {med_i:>7}  {pct_e:>9.1f}% {pct_i:>9.1f}%")
