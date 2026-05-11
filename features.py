import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(r"c:\Users\zivev\HiddenKnowledgeAfterUnlearning_attempt1")
EXT_DIR = REPO / "inside_out_ext"
PARQUET = REPO / "plots" / "all_k_scores.parquet"

SEED = 42
N_OPTIONS = 4

def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

# split mapping
te = np.random.default_rng(SEED).permutation(1273)[700:]
orig_to_pos = {int(qi): i for i, qi in enumerate(te)}

# labels (local file; avoids datasets dependency)
tf = pd.read_csv(REPO / "data" / "wmdp_tf_pairs.csv")
q = tf[["original_id", "correct_idx"]].drop_duplicates("original_id")
correct_idx = (
    q.sort_values("original_id")["correct_idx"]
    .astype(int)
    .to_numpy()
)

# parquet
df = pd.read_parquet(PARQUET)

# pre-filter: base, multi, single, k_int=1 & k_ext=1
base_single = df[
    (df["domain"] == "bio") &
    (df["clf"] == "LR") &
    (df["layer_config"] == "multi") &
    (df["split_type"] == "single") &
    (df["model_id"] == "base")
][["question_idx", "k_internal", "k_external"]]

filt_mask = np.zeros(len(te), dtype=bool)
for _, row in base_single.iterrows():
    qi = int(row["question_idx"])
    pos = orig_to_pos.get(qi)
    if pos is not None and float(row["k_internal"]) == 1.0 and float(row["k_external"]) == 1.0:
        filt_mask[pos] = True

sel_pos = np.where(filt_mask)[0]
sel_qi = te[sel_pos]
print(f"Filtered questions: {len(sel_qi)} / {len(te)}")

# arrays
ext_base = np.load(EXT_DIR / "base_bio_ext.npy")
int_proba_base = np.load(EXT_DIR / "base_bio_int_proba.npy")

# features 1,2,3,7,8
f1 = np.zeros(len(sel_qi))
f2 = np.zeros(len(sel_qi))
f3 = np.zeros(len(sel_qi))
f7 = np.zeros(len(sel_qi))
f8 = np.zeros(len(sel_qi))

for i, qi in enumerate(sel_qi):
    c = int(correct_idx[qi])
    wrongs = [j for j in range(N_OPTIONS) if j != c]

    ext_c = float(ext_base[qi, c])
    ext_w = np.array([float(ext_base[qi, j]) for j in wrongs])
    ext_pair = sigmoid(ext_c - ext_w)

    f1[i] = ext_pair.min()
    f2[i] = sigmoid(ext_w).max()
    f3[i] = sigmoid(ext_c)

    int_c = float(int_proba_base[qi, c])
    int_w = np.array([float(int_proba_base[qi, j]) for j in wrongs])
    int_marg = int_c - int_w
    ext_marg = ext_c - ext_w

    f7[i] = int_marg.min()

    # rank alignment (Spearman-like via rank corr on 3 values)
    rx = pd.Series(int_marg).rank(method="average").to_numpy()
    ry = pd.Series(ext_marg).rank(method="average").to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        f8[i] = 0.0
    else:
        f8[i] = np.corrcoef(rx, ry)[0, 1]

# features 4,5,6 from base per-layer cv
base_layer_cv = df[
    (df["domain"] == "bio") &
    (df["clf"] == "LR") &
    (df["split_type"] == "cv") &
    (df["model_id"] == "base") &
    (df["layer_config"].astype(str).str.startswith("layer_"))
][["question_idx", "layer_config", "k_internal"]].copy()

base_layer_cv["layer"] = (
    base_layer_cv["layer_config"]
    .astype(str)
    .str.replace("layer_", "", regex=False)
    .astype(int)
)
ql = base_layer_cv.groupby(["question_idx", "layer"], as_index=False)["k_internal"].mean()
pivot = ql.pivot(index="question_idx", columns="layer", values="k_internal").sort_index(axis=1)
layers = pivot.columns.to_numpy()

f4 = np.full(len(sel_qi), np.nan)
f5 = np.full(len(sel_qi), np.nan)
f6 = np.full(len(sel_qi), np.nan)

for i, qi in enumerate(sel_qi):
    if qi not in pivot.index:
        continue
    arr = pivot.loc[qi].to_numpy(dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        continue
    a = arr[valid]
    l = layers[valid]

    f5[i] = a.mean()
    f6[i] = a.std()
    idx = np.where(a >= 0.999999)[0]
    if len(idx):
        f4[i] = l[idx[0]]

features = {
    "1_min_pairwise_sigmoid_margin": f1,
    "2_max_distractor_confidence": f2,
    "3_correct_abs_confidence": f3,
    "4_earliest_layer_kint_eq_1": f4,
    "5_mean_kint_across_layers": f5,
    "6_std_kint_across_layers": f6,
    "7_min_probe_prob_margin": f7,
    "8_rank_alignment_corr": f8,
}

print("\nMean ± Std on filtered questions:")
for name, v in features.items():
    vv = v[np.isfinite(v)]
    print(f"{name:35s} n={len(vv):3d}  mean={vv.mean():.6f}  std={vv.std():.6f}")