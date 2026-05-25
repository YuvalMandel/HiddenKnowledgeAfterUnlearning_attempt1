"""
Tab:regression — base features predict K_int/K_ext trajectory targets, v6 pipeline.

Base features (8, same as old pipeline) computed for Q*=701 questions.
Features 1-3,7,8 from inside_out_ext/base_bio_*.npy.
Features 4-6 (per-layer K_int profile) from old all_k_scores.parquet base layer rows.

Targets (best_layer probe, v6):
  k_int_traj_auc  — mean K_int over ck1-ck8
  k_int_ck8       — K_int at ck8
  ext_drop        — 1.0 - K_ext_ck8
  int_ext_gap     — K_int_ck8 - K_ext_ck8

Models: Ridge, Lasso, ElasticNet, RF, HGB
CV: RepeatedKFold(5 splits, 10 repeats)
Outputs: v6_trajectory_regression.csv, v6_regression_feature_importance_{target}.pdf/.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_predict, cross_val_score, KFold

REPO     = Path(__file__).parent.parent
OUT_DIR  = REPO / "inside_out_out"
EXT_DIR  = REPO / "inside_out_ext"
SAVE_DIR = Path(__file__).parent
KNOWS    = 0.5
SEED     = 42
N_OPT    = 4

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

FEATURE_COLS = [
    "min_pairwise_sigmoid_margin",
    "max_distractor_confidence",
    "correct_option_confidence",
    "earliest_layer_kint1",
    "mean_kint_layers",
    "std_kint_layers",
    "min_probe_probability_margin",
    "rank_alignment",
]
FEATURE_SHORT = [
    "min_ext_margin", "max_distractor", "correct_conf",
    "earliest_kint1", "mean_kint", "std_kint",
    "min_int_margin", "rank_align",
]
TARGETS = {
    "k_int_traj_auc":  "K_int traj AUC (best-layer, ck1-ck8)",
    "k_int_ck8":       "K_int @ ck8 (best-layer)",
    "ext_drop":        "Ext drop (1 - K_ext_ck8)",
    "int_ext_gap":     "K_int - K_ext gap @ ck8",
}

CV    = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
MODELS = {
    "Ridge":      Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Lasso":      Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.01, max_iter=5000))]),
    "ElasticNet": Pipeline([("sc", StandardScaler()), ("m", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))]),
    "RF":         Pipeline([("sc", StandardScaler()), ("m", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=4))]),
    "HGB":        Pipeline([("sc", StandardScaler()), ("m", HistGradientBoostingRegressor(max_iter=200, random_state=SEED))]),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def load_cv_bl(model_id):
    p = OUT_DIR / model_id / "k_scores.parquet"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_parquet(p)
    return df[
        (df["split_type"] == "cv") & (df["domain"] == "bio") &
        (df["clf"] == "LR") & (df["probe_type"] == "own") &
        (df["layer_config"] == "best_layer")
    ][["model_id", "question_idx", "k_internal", "k_external"]].copy()


# ── Q* ────────────────────────────────────────────────────────────────────────
base_cv = load_cv_bl("base")
qstar   = sorted(base_cv.loc[
    (base_cv["k_internal"] == 1.0) & (base_cv["k_external"] == 1.0),
    "question_idx"
].tolist())
filt_te = np.array(qstar, dtype=int)
print(f"Q* size: {len(filt_te)}")

# ── Correct answer indices ────────────────────────────────────────────────────
tf = pd.read_csv(REPO / "data" / "wmdp_tf_pairs.csv")
correct_idx = (tf[["original_id", "correct_idx"]]
               .drop_duplicates("original_id")
               .sort_values("original_id")["correct_idx"].astype(int).to_numpy())

# ── Base features ─────────────────────────────────────────────────────────────
print("Computing base features...")
ext_base = np.load(EXT_DIR / "base_bio_ext.npy")       # (1273, 4)
int_base = np.load(EXT_DIR / "base_bio_int_proba.npy") # (1273, 4)

f1, f2, f3, f7, f8 = [], [], [], [], []
for qi in filt_te:
    c  = int(correct_idx[qi])
    ws = [j for j in range(N_OPT) if j != c]
    ec = float(ext_base[qi, c])
    ew = np.array([float(ext_base[qi, j]) for j in ws])
    pair_sig = sigmoid(ec - ew)
    f1.append(float(pair_sig.min()))
    f2.append(float(sigmoid(ew).max()))
    f3.append(float(sigmoid(ec)))
    ic = float(int_base[qi, c])
    iw = np.array([float(int_base[qi, j]) for j in ws])
    int_marg = ic - iw
    ext_marg = ec - ew
    f7.append(float(int_marg.min()))
    rx = pd.Series(int_marg).rank(method="average").to_numpy()
    ry = pd.Series(ext_marg).rank(method="average").to_numpy()
    f8.append(float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else 0.0)

# Per-layer features from old all_k_scores.parquet
old_parquet = REPO / "plots" / "all_k_scores.parquet"
f4, f5, f6 = [np.nan] * len(filt_te), [np.nan] * len(filt_te), [np.nan] * len(filt_te)
if old_parquet.exists():
    old = pd.read_parquet(old_parquet)
    bl = old[
        (old["domain"] == "bio") & (old["clf"] == "LR") &
        (old["split_type"] == "cv") & (old["model_id"] == "base") &
        (old["layer_config"].astype(str).str.startswith("layer_"))
    ][["question_idx", "layer_config", "k_internal"]].copy()
    bl["layer"] = bl["layer_config"].str.replace("layer_", "", regex=False).astype(int)
    ql = bl.groupby(["question_idx", "layer"], as_index=False)["k_internal"].mean()
    pivot = ql.pivot(index="question_idx", columns="layer", values="k_internal").sort_index(axis=1)
    layers = pivot.columns.to_numpy()
    for i, qi in enumerate(filt_te):
        if qi not in pivot.index: continue
        arr = pivot.loc[qi].to_numpy(dtype=np.float64)
        valid = np.isfinite(arr)
        if not valid.any(): continue
        a, l = arr[valid], layers[valid]
        f5[i] = float(a.mean())
        f6[i] = float(a.std())
        idx = np.where(a >= 0.999999)[0]
        f4[i] = float(l[idx[0]]) if len(idx) else np.nan
    print(f"  Per-layer features loaded from {old_parquet.name}")
else:
    print(f"  WARNING: {old_parquet} not found; features 4-6 will be NaN")

feat_df = pd.DataFrame({
    "question_idx":                 filt_te.tolist(),
    "min_pairwise_sigmoid_margin":  f1,
    "max_distractor_confidence":    f2,
    "correct_option_confidence":    f3,
    "earliest_layer_kint1":         f4,
    "mean_kint_layers":             f5,
    "std_kint_layers":              f6,
    "min_probe_probability_margin": f7,
    "rank_alignment":               f8,
})

# ── Trajectory targets from v6 ────────────────────────────────────────────────
print("Loading trajectory targets...")
targ_rows = []
for mname, label in METHODS:
    ck8_df = load_cv_bl(f"{mname}_ck8")
    if ck8_df.empty: continue
    ck8_q = ck8_df[ck8_df["question_idx"].isin(set(filt_te))].set_index("question_idx")

    traj = {qi: [] for qi in filt_te if qi in ck8_q.index}
    for ck in CKPTS:
        ck_df = load_cv_bl(f"{mname}_{ck}")
        if ck_df.empty: continue
        ck_q = ck_df[ck_df["question_idx"].isin(traj)].set_index("question_idx")
        for qi in traj:
            if qi in ck_q.index:
                traj[qi].append(float(ck_q.loc[qi, "k_internal"]))

    for qi in traj:
        if not traj[qi]: continue
        row = ck8_q.loc[qi]
        ki_ck8 = float(row["k_internal"])
        ke_ck8 = float(row["k_external"])
        targ_rows.append({
            "method":        label,
            "question_idx":  qi,
            "k_int_traj_auc": float(np.mean(traj[qi])),
            "k_int_ck8":     ki_ck8,
            "ext_drop":      1.0 - ke_ck8,
            "int_ext_gap":   ki_ck8 - ke_ck8,
        })

targ_df = pd.DataFrame(targ_rows)
data = feat_df.merge(targ_df, on="question_idx", how="inner")
data = data.dropna(subset=FEATURE_COLS + list(TARGETS.keys()))
print(f"  Dataset: {len(data)} rows, {data['question_idx'].nunique()} questions")

X = data[FEATURE_COLS].to_numpy(dtype=float)
result_rows = []

for t_key, t_label in TARGETS.items():
    print(f"\nTarget: {t_key}")
    y = data[t_key].to_numpy(dtype=float)
    best_rho, best_name = -1, ""
    for mname, model in MODELS.items():
        r2s = cross_val_score(model, X, y, cv=CV, scoring="r2", n_jobs=1)
        y_pred = cross_val_predict(model, X, y,
                                   cv=KFold(n_splits=5, shuffle=True, random_state=SEED))
        rho, _ = spearmanr(y, y_pred)
        print(f"  {mname:12s}  R2={r2s.mean():.3f}±{r2s.std():.3f}  rho={rho:.3f}")
        result_rows.append({
            "target": t_key, "model": mname,
            "r2_mean": round(float(r2s.mean()), 4),
            "r2_std":  round(float(r2s.std()),  4),
            "spearman_rho": round(float(rho), 4),
        })
        if abs(rho) > best_rho:
            best_rho, best_name, y_best = abs(rho), mname, y_pred

    # Feature importance plot for best model
    best_model = MODELS[best_name]
    best_model.fit(X, y)
    m = best_model.named_steps["m"]
    if hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
    elif hasattr(m, "coef_"):
        sc = best_model.named_steps["sc"]
        imp = np.abs(m.coef_ * sc.scale_)
    else:
        imp = np.zeros(len(FEATURE_COLS))
    imp = imp / (imp.sum() + 1e-12)
    order = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = ["#d62728" if imp[i] == imp.max() else "#1f77b4" for i in order]
    ax.barh(np.arange(len(order)), imp[order], color=colors, alpha=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([FEATURE_SHORT[i] for i in order], fontsize=9)
    ax.set_title(f"{t_key} | best={best_name}, rho={best_rho:.3f}", fontsize=9)
    ax.set_xlabel("Normalized importance", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    stem = SAVE_DIR / f"v6_regression_feature_importance_{t_key}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

results = pd.DataFrame(result_rows)
results.to_csv(SAVE_DIR / "v6_trajectory_regression.csv", index=False)
print(f"\nSaved v6_trajectory_regression.csv")

# Print best-per-target summary (for paper table)
print("\n% Tab:regression summary (best model per target):")
for t_key in TARGETS:
    sub = results[results["target"] == t_key]
    best = sub.loc[sub["r2_mean"].idxmax()]
    print(f"  {t_key:<20}  best={best['model']:<6}  R2={best['r2_mean']:.3f}±{best['r2_std']:.3f}  rho={best['spearman_rho']:.3f}")
