"""Does rho=0.73 survive (a) v6-sourced features, (b) leakage-free grouped CV?

tab:regression currently reports HGB R2=0.515, rho=0.730 for k_int_traj_auc.
Three things are worth separating:

  1. PROVENANCE. Features 4-6 (per-layer base K_int) came from the pre-v6
     plots/all_k_scores.parquet while the targets come from v6. Both cover the
     same 573 questions, so n is unchanged, but the sources differ.
  2. LEAKAGE. The design matrix has one row per (question, method) -- 8 rows per
     question with IDENTICAL features and different targets. RepeatedKFold
     splits rows, so a question's other 7 rows sit in the training fold. A
     flexible model can look up the answer.
  3. COVERAGE. Features 4-6 exist only for 573 questions, cutting Q* 701 -> 320.
     Dropping those three features restores the full 701.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import (GroupKFold, KFold, RepeatedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
OUT, EXT = REPO / "inside_out_out", REPO / "inside_out_ext"
SEED, N_OPT = 42, 4
METHODS = ["GradDiff", "PB_J", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR"]
CKPTS = [f"ck{i}" for i in range(1, 9)]
BASE3 = ["min_pairwise_sigmoid_margin", "max_distractor_confidence",
         "correct_option_confidence"]
LAYER3 = ["earliest_layer_kint1", "mean_kint_layers", "std_kint_layers"]
REST2 = ["min_probe_probability_margin", "rank_alignment"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def load_cv_bl(mid):
    p = OUT / mid / "k_scores.parquet"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_parquet(p)
    return d[(d.split_type == "cv") & (d.domain == "bio") & (d.clf == "LR")
             & (d.probe_type == "own") & (d.layer_config == "best_layer")][
        ["question_idx", "k_internal", "k_external"]]


base_cv = load_cv_bl("base")
qstar = np.array(sorted(base_cv.loc[(base_cv.k_internal == 1.0)
                                    & (base_cv.k_external == 1.0),
                                    "question_idx"].tolist()), dtype=int)

tf = pd.read_csv(REPO / "data" / "wmdp_tf_pairs.csv", keep_default_na=False)
correct_idx = (tf[["original_id", "correct_idx"]].drop_duplicates("original_id")
               .sort_values("original_id")["correct_idx"].astype(int).to_numpy())

ext_b = np.load(EXT / "base_bio_ext.npy")
int_b = np.load(EXT / "base_bio_int_proba.npy")
rows = []
for qi in qstar:
    c = int(correct_idx[qi])
    ws = [j for j in range(N_OPT) if j != c]
    ec, ew = float(ext_b[qi, c]), np.array([float(ext_b[qi, j]) for j in ws])
    ic, iw = float(int_b[qi, c]), np.array([float(int_b[qi, j]) for j in ws])
    rx = pd.Series(ic - iw).rank().to_numpy()
    ry = pd.Series(ec - ew).rank().to_numpy()
    rows.append(dict(
        question_idx=qi,
        min_pairwise_sigmoid_margin=float(sigmoid(ec - ew).min()),
        max_distractor_confidence=float(sigmoid(ew).max()),
        correct_option_confidence=float(sigmoid(ec)),
        min_probe_probability_margin=float((ic - iw).min()),
        rank_alignment=(float(np.corrcoef(rx, ry)[0, 1])
                        if rx.std() > 0 and ry.std() > 0 else 0.0)))
feat = pd.DataFrame(rows)


def layer_feats(src):
    """earliest layer reaching K_int=1, mean and std of per-layer K_int."""
    if src == "old":
        d = pd.read_parquet(REPO / "plots" / "all_k_scores.parquet")
        d = d[(d.domain == "bio") & (d.clf == "LR") & (d.split_type == "cv")
              & (d.model_id == "base")
              & (d.layer_config.astype(str).str.startswith("layer_"))]
    else:
        d = pd.read_parquet(OUT / "base" / "k_scores_layer.parquet")
        d = d[d.layer_config.astype(str).str.startswith("layer_")]
    d = d[["question_idx", "layer_config", "k_internal"]].copy()
    d["layer"] = d.layer_config.str.replace("layer_", "", regex=False).astype(int)
    piv = (d.groupby(["question_idx", "layer"], as_index=False).k_internal.mean()
           .pivot(index="question_idx", columns="layer", values="k_internal")
           .sort_index(axis=1))
    out = []
    for qi, r in piv.iterrows():
        a = r.to_numpy(dtype=float)
        v = np.isfinite(a)
        if not v.any():
            continue
        aa, ll = a[v], piv.columns.to_numpy()[v]
        hit = np.where(aa >= 0.999999)[0]
        out.append(dict(question_idx=int(qi),
                        earliest_layer_kint1=float(ll[hit[0]]) if len(hit) else np.nan,
                        mean_kint_layers=float(aa.mean()),
                        std_kint_layers=float(aa.std())))
    return pd.DataFrame(out)


trows = []
for m in METHODS:
    ck8 = load_cv_bl(f"{m}_ck8")
    if ck8.empty:
        continue
    ck8q = ck8[ck8.question_idx.isin(set(qstar))].set_index("question_idx")
    traj = {qi: [] for qi in qstar if qi in ck8q.index}
    for ck in CKPTS:
        d = load_cv_bl(f"{m}_{ck}")
        if d.empty:
            continue
        dq = d[d.question_idx.isin(traj)].set_index("question_idx")
        for qi in traj:
            if qi in dq.index:
                traj[qi].append(float(dq.loc[qi, "k_internal"]))
    for qi, v in traj.items():
        if v:
            trows.append(dict(method=m, question_idx=qi,
                              k_int_traj_auc=float(np.mean(v))))
targ = pd.DataFrame(trows)

HGB = Pipeline([("sc", StandardScaler()),
                ("m", HistGradientBoostingRegressor(max_iter=200,
                                                    random_state=SEED))])


def run(cols, layer_src, grouped, tag):
    f = feat.copy()
    if layer_src:
        f = f.merge(layer_feats(layer_src), on="question_idx", how="inner")
    d = f.merge(targ, on="question_idx", how="inner").dropna(subset=cols +
                                                             ["k_int_traj_auc"])
    X = d[cols].to_numpy(float)
    y = d.k_int_traj_auc.to_numpy(float)
    g = d.question_idx.to_numpy()
    # match the paper exactly: R2 from RepeatedKFold(5x10) mean of fold scores,
    # rho from a single KFold(5) cross_val_predict. Grouped variants swap both
    # for GroupKFold on question_idx.
    if grouped:
        cv_score = GroupKFold(n_splits=5)
        cv_pred = GroupKFold(n_splits=5)
        kw = dict(groups=g)
    else:
        cv_score = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
        cv_pred = KFold(n_splits=5, shuffle=True, random_state=SEED)
        kw = {}
    r2 = cross_val_score(HGB, X, y, cv=cv_score, scoring="r2", **kw).mean()
    pred = cross_val_predict(HGB, X, y, cv=cv_pred, **kw)
    rho = spearmanr(y, pred).statistic
    print(f"{tag:<46}{d.question_idx.nunique():>6}{len(d):>8}"
          f"{r2:>9.3f}{rho:>8.3f}")
    return r2, rho


print(f"{'configuration':<46}{'quest':>6}{'rows':>8}{'R2':>9}{'rho':>8}")
print("-" * 77)
run(BASE3 + LAYER3 + REST2, "old", False, "A. published: old feats, row-wise CV")
run(BASE3 + LAYER3 + REST2, "v6", False, "B. v6 feats, row-wise CV")
run(BASE3 + LAYER3 + REST2, "v6", True, "C. v6 feats, GROUPED CV (no leakage)")
run(BASE3 + REST2, None, False, "D. 5 feats, full Q*, row-wise CV")
run(BASE3 + REST2, None, True, "E. 5 feats, full Q*, GROUPED CV")
print("-" * 77)
print("paper reports: R2=0.515  rho=0.730")
