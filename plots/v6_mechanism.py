"""
Tab:mechanism — K_int/K_ext mechanism comparison, v6 pipeline.

For each question in Q* (best_layer, 701), compute:
  k_int_traj  = mean K_int over ck1–ck8 (best_layer probe)
  ext_drop    = 1.0 - K_ext_ck8  (K_ext_base=1 for all Q*)
  int_ext_gap = K_int_ck8 - K_ext_ck8

Aggregate means by subset (retained/suppressed/forgotten), pooled across all 8 methods.
Outputs: v6_mechanism.csv, v6_mechanism_agg.csv, prints LaTeX row.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR  = Path(__file__).parent.parent / "inside_out_out"
SAVE_DIR = Path(__file__).parent
KNOWS    = 0.5

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


# Q* from base
base_cv = load_cv_bl("base")
qstar = set(base_cv.loc[
    (base_cv["k_internal"] == 1.0) & (base_cv["k_external"] == 1.0),
    "question_idx"
])
print(f"Q* size: {len(qstar)}")

rows = []
for mname, label in METHODS:
    # ck8 subset labels
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

    k_int_ck8 = ck8_q["k_internal"].to_dict()
    k_ext_ck8 = ck8_q["k_external"].to_dict()

    # K_int trajectory: mean over ck1–ck8
    traj = {qi: [] for qi in labels_map}
    for ck in CKPTS:
        ck_df = load_cv_bl(f"{mname}_{ck}")
        if ck_df.empty: continue
        ck_q = ck_df[ck_df["question_idx"].isin(traj)].set_index("question_idx")
        for qi in traj:
            if qi in ck_q.index:
                traj[qi].append(float(ck_q.loc[qi, "k_internal"]))

    for qi, subset in labels_map.items():
        mean_traj = np.mean(traj[qi]) if traj[qi] else float("nan")
        ki_ck8 = float(k_int_ck8.get(qi, float("nan")))
        ke_ck8 = float(k_ext_ck8.get(qi, float("nan")))
        rows.append({
            "method":       label,
            "question_idx": qi,
            "subset":       subset,
            "k_int_traj":   mean_traj,
            "ext_drop":     1.0 - ke_ck8,
            "int_ext_gap":  ki_ck8 - ke_ck8,
            "k_int_ck8":    ki_ck8,
            "k_ext_ck8":    ke_ck8,
        })

df_all = pd.DataFrame(rows)
df_all.to_csv(SAVE_DIR / "v6_mechanism.csv", index=False)
print(f"Saved v6_mechanism.csv  ({len(df_all)} rows)")

# Aggregate across all methods by subset
SUBSETS = ["retained", "suppressed", "forgotten", "lucky"]
agg = df_all.groupby("subset")[["k_int_traj", "ext_drop", "int_ext_gap"]].mean()
agg["n"] = df_all.groupby("subset")["question_idx"].count()
agg = agg.reindex(SUBSETS).reset_index()
agg.to_csv(SAVE_DIR / "v6_mechanism_agg.csv", index=False)
print(f"\nSaved v6_mechanism_agg.csv")
print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# LaTeX table rows
print("\n% Tab:mechanism LaTeX rows (best_layer probe):")
for _, r in agg.iterrows():
    sign_gap = "+" if r["int_ext_gap"] >= 0 else ""
    print(f"{r['subset'].capitalize():<10} & {r['k_int_traj']:.2f} & "
          f"{r['ext_drop']:.2f} & ${sign_gap}{r['int_ext_gap']:.2f}$ \\\\")
