"""
Checkpoint trajectory line plots per method — full Q*, best_layer, v6 KFold.

One PDF/PNG per method. 2 lines over all Q* questions (no subset split):
  K_ext  solid line
  K_int  dotted line
Shaded band = ±1 SEM. X-axis: base → ck1 … ck8.

Output: plots/v6_trajectory_full/<method>.pdf and .png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR  = Path(__file__).parent.parent / "inside_out_out"
SAVE_DIR = Path(__file__).parent / "v6_trajectory_full"
SAVE_DIR.mkdir(exist_ok=True)

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
CKPTS      = [f"ck{i}" for i in range(1, 9)]
CKPT_TICKS = ["base"] + CKPTS

INT_COLOR = "#2166ac"
EXT_COLOR = "#d6604d"


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


def mean_sem(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


# ── Q* from base ──────────────────────────────────────────────────────────────
base_cv = load_cv_bl("base")
qstar   = set(base_cv.loc[
    (base_cv["k_internal"] == 1.0) & (base_cv["k_external"] == 1.0),
    "question_idx"
])
print(f"Q* size: {len(qstar)}")

# ── One figure per method ─────────────────────────────────────────────────────
for mname, label in METHODS:
    print(f"  {label} ...", end=" ", flush=True)

    traj_ki = [(1.0, 0.0)]   # base: all Q* have K_int=1
    traj_ke = [(1.0, 0.0)]   # base: all Q* have K_ext=1

    for ck in CKPTS:
        df_ck = load_cv_bl(f"{mname}_{ck}")
        if df_ck.empty:
            traj_ki.append((float("nan"), float("nan")))
            traj_ke.append((float("nan"), float("nan")))
            continue
        df_ck = df_ck.set_index("question_idx")
        q_idx = [qi for qi in qstar if qi in df_ck.index]
        traj_ki.append(mean_sem([float(df_ck.loc[qi, "k_internal"]) for qi in q_idx]))
        traj_ke.append(mean_sem([float(df_ck.loc[qi, "k_external"]) for qi in q_idx]))

    x      = np.arange(len(CKPT_TICKS))
    ki_m   = np.array([v[0] for v in traj_ki])
    ki_s   = np.array([v[1] for v in traj_ki])
    ke_m   = np.array([v[0] for v in traj_ke])
    ke_s   = np.array([v[1] for v in traj_ke])

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    ax.plot(x, ke_m * 100, color=EXT_COLOR, linestyle="-", linewidth=2.2,
            marker="o", markersize=4.5, label=r"$K_\mathrm{ext}$", zorder=4)
    ax.fill_between(x, (ke_m - ke_s) * 100, (ke_m + ke_s) * 100,
                    color=EXT_COLOR, alpha=0.15, zorder=2)

    ax.plot(x, ki_m * 100, color=INT_COLOR, linestyle=":", linewidth=2.2,
            marker="o", markersize=4.5, label=r"$K_\mathrm{int}$", zorder=4)
    ax.fill_between(x, (ki_m - ki_s) * 100, (ki_m + ki_s) * 100,
                    color=INT_COLOR, alpha=0.15, zorder=2)

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
    ax.legend(fontsize=9, loc="lower left", framealpha=0.88)

    fig.tight_layout()
    stem = mname.replace("-", "_")
    fig.savefig(SAVE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(SAVE_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("done")

print(f"\nSaved to: {SAVE_DIR}/")
