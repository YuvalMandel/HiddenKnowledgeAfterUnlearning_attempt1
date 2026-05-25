"""
Suppressed vs Forgotten K_int trajectory gap — best_layer probe, v6 KFold, Q*.

For each method: compare mean K_int over ck1–ck8 between suppressed and forgotten
Q* questions (classified at ck8). Reports Mann-Whitney U, BH-corrected q-values,
bootstrap 95% CI on mean difference, Cohen's d, Cliff's delta.
Outputs: v6_sf_gap.csv, v6_sf_gap_forest.pdf/.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

OUT_DIR  = Path(__file__).parent.parent / "inside_out_out"
SAVE_DIR = Path(__file__).parent
KNOWS    = 0.5
N_BOOT   = 5000
SEED     = 42

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


def bootstrap_ci(a, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    diffs = [
        rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
        for _ in range(n_boot)
    ]
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na + nb <= 2: return float("nan")
    pooled = ((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na + nb - 2)
    return float((a.mean() - b.mean()) / np.sqrt(max(pooled, 1e-12)))


def cliffs_delta(a, b):
    greater = sum(ai > bi for ai in a for bi in b)
    less    = sum(ai < bi for ai in a for bi in b)
    return float((greater - less) / max(len(a) * len(b), 1))


def bh_correct(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int); ranks[order] = np.arange(1, n+1)
    q = np.minimum(1.0, np.array(pvals) * n / ranks)
    # monotone enforcement
    q_mono = q.copy()
    for i in range(n-2, -1, -1):
        q_mono[order[i]] = min(q_mono[order[i]], q_mono[order[i+1]])
    return q_mono.tolist()


# ── Q* from base ──────────────────────────────────────────────────────────────
base_cv = load_cv_bl("base")
qstar   = set(base_cv.loc[
    (base_cv["k_internal"] == 1.0) & (base_cv["k_external"] == 1.0),
    "question_idx"
])
print(f"Q* size: {len(qstar)}")

# ── Build per-question K_int trajectory for each method ───────────────────────
rows = []
pvals = []

for mname, label in METHODS:
    # Subset labels from ck8
    ck8_df = load_cv_bl(f"{mname}_ck8")
    if ck8_df.empty: continue
    ck8_q = ck8_df[ck8_df["question_idx"].isin(qstar)].set_index("question_idx")
    labels_map = {}
    for qi, row in ck8_q.iterrows():
        ki, ke = row["k_internal"], row["k_external"]
        if   ki > KNOWS and ke > KNOWS:  labels_map[qi] = "retained"
        elif ki > KNOWS and ke <= KNOWS: labels_map[qi] = "suppressed"
        elif ki <= KNOWS and ke > KNOWS: labels_map[qi] = "lucky"
        else:                            labels_map[qi] = "forgotten"

    # K_int trajectory: mean over ck1–ck8 per question
    traj = {qi: [] for qi in qstar if qi in labels_map}
    for ck in CKPTS:
        ck_df = load_cv_bl(f"{mname}_{ck}")
        if ck_df.empty: continue
        ck_q = ck_df[ck_df["question_idx"].isin(traj)].set_index("question_idx")
        for qi in traj:
            if qi in ck_q.index:
                traj[qi].append(float(ck_q.loc[qi, "k_internal"]))

    mean_traj = {qi: np.mean(v) for qi, v in traj.items() if v}

    supp = np.array([mean_traj[qi] for qi, l in labels_map.items()
                     if l == "suppressed" and qi in mean_traj])
    forg = np.array([mean_traj[qi] for qi, l in labels_map.items()
                     if l == "forgotten" and qi in mean_traj])

    n_s, n_f = len(supp), len(forg)
    print(f"  {label:<12}  suppressed={n_s}  forgotten={n_f}")
    if n_s < 2 or n_f < 2:
        rows.append({"method": label, "n_suppressed": n_s, "n_forgotten": n_f,
                     "mean_supp": float(supp.mean()) if n_s else float("nan"),
                     "mean_forg": float(forg.mean()) if n_f else float("nan"),
                     "mean_diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                     "p_mwu": float("nan"), "cohens_d": float("nan"), "cliffs_delta": float("nan")})
        pvals.append(1.0); continue

    _, p = mannwhitneyu(supp, forg, alternative="greater")
    diff, ci_lo, ci_hi = bootstrap_ci(supp, forg)
    d  = cohens_d(supp, forg)
    cd = cliffs_delta(supp, forg)
    rows.append({"method": label, "n_suppressed": n_s, "n_forgotten": n_f,
                 "mean_supp": float(supp.mean()), "mean_forg": float(forg.mean()),
                 "mean_diff": diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
                 "p_mwu": float(p), "cohens_d": d, "cliffs_delta": cd})
    pvals.append(float(p))

# BH correction
q_vals = bh_correct(pvals)
for i, r in enumerate(rows): r["q_bh"] = q_vals[i]

df_sf = pd.DataFrame(rows)
df_sf.to_csv(SAVE_DIR / "v6_sf_gap.csv", index=False)
print(f"\nSaved: v6_sf_gap.csv")
print(df_sf[["method","n_suppressed","n_forgotten","mean_diff","ci_lo","ci_hi","q_bh","cohens_d"]].to_string(index=False))

# ── Forest plot ───────────────────────────────────────────────────────────────
df_plot = df_sf.dropna(subset=["mean_diff"]).copy()
n = len(df_plot)
y = np.arange(n)

fig, ax = plt.subplots(figsize=(7, 0.55 * n + 1.2))
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

for i, (_, r) in enumerate(df_plot.iterrows()):
    sig = r.get("q_bh", 1.0) < 0.05
    color = "#d62728" if sig else "#888888"
    ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=color, linewidth=2, zorder=3)
    ax.scatter(r["mean_diff"], i, color=color, s=50, zorder=4,
               marker="D" if sig else "o")
    ax.text(r["ci_hi"] + 0.005, i,
            f"d={r['cohens_d']:.2f}, q={r['q_bh']:.3f}",
            va="center", fontsize=7.5, color=color)

ax.set_yticks(y)
ax.set_yticklabels(df_plot["method"].tolist(), fontsize=9)
ax.set_xlabel("Mean K_int (suppressed − forgotten), trajectory mean ck1–ck8", fontsize=8.5)
ax.set_xlim(left=min(df_plot["ci_lo"].min() - 0.05, -0.02))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
fig.tight_layout()
fig.savefig(SAVE_DIR / "v6_sf_gap_forest.pdf", bbox_inches="tight")
fig.savefig(SAVE_DIR / "v6_sf_gap_forest.png", dpi=150, bbox_inches="tight")
print(f"Saved: v6_sf_gap_forest.pdf / .png")
plt.close(fig)
