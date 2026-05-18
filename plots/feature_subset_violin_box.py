#!/usr/bin/env python3
"""
Plot base-feature distributions by subset for each unlearning method.

For each method at ck8:
  - classify filtered questions (n=312) into retained/suppressed/forgotten/lucky
  - plot per-feature distributions across subsets (box + jitter)
  - save both PNG and PDF

Run from repo root:
  python plots/feature_subset_violin_box.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent.parent
EXT_DIR = REPO / "inside_out_ext"
PARQUET = REPO / "plots" / "all_k_scores.parquet"
OUT_DIR = REPO / "plots" / "feature_subset_violin_box"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_SIZE = 500
VAL_SIZE = 200
N_OPTIONS = 4
KNOWS_THRESHOLD = 0.5

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
ORDER = ["retained", "suppressed", "forgotten", "lucky"]
SUBSET_COLORS = {
    "retained": "#2ca02c",
    "suppressed": "#d62728",
    "forgotten": "#7f7f7f",
    "lucky": "#ff7f0e",
}

FEATURE_ORDER = [
    "1_min_pairwise_sigmoid_margin",
    "2_max_distractor_confidence",
    "3_correct_abs_confidence",
    "4_earliest_layer_kint_eq_1",
    "5_mean_kint_across_layers",
    "6_std_kint_across_layers",
    "7_min_probe_prob_margin",
    "8_rank_alignment_corr",
]
FEATURE_LABELS = {
    "1_min_pairwise_sigmoid_margin": "Min pairwise sigmoid margin",
    "2_max_distractor_confidence": "Max distractor confidence",
    "3_correct_abs_confidence": "Correct option confidence",
    "4_earliest_layer_kint_eq_1": "Earliest layer K_int=1",
    "5_mean_kint_across_layers": "Mean K_int across layers",
    "6_std_kint_across_layers": "Std K_int across layers",
    "7_min_probe_prob_margin": "Min probe prob margin",
    "8_rank_alignment_corr": "Rank alignment corr",
}


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def load_correct_idx():
    tf = pd.read_csv(REPO / "data" / "wmdp_tf_pairs.csv")
    q = tf[["original_id", "correct_idx"]].drop_duplicates("original_id")
    return q.sort_values("original_id")["correct_idx"].astype(int).to_numpy()


def compute_k_ext(ext, correct_idx, te):
    k = np.zeros(len(te), np.float32)
    for i, qi in enumerate(te):
        c = int(correct_idx[qi])
        ws = [ext[qi, j] for j in range(N_OPTIONS) if j != c]
        k[i] = sum(float(ext[qi, c] > w) for w in ws) / len(ws)
    return k


def compute_k_int_from_proba(proba, correct_idx, te):
    k = np.zeros(len(te), np.float32)
    for i, qi in enumerate(te):
        c = int(correct_idx[qi])
        ws = [proba[qi, j] for j in range(N_OPTIONS) if j != c]
        k[i] = sum(float(proba[qi, c] > w) for w in ws) / len(ws)
    return k


def compute_base_features(correct_idx, te, filt_te, df):
    ext_base = np.load(EXT_DIR / "base_bio_ext.npy")
    int_proba_base = np.load(EXT_DIR / "base_bio_int_proba.npy")

    f1 = np.zeros(len(filt_te), dtype=np.float64)
    f2 = np.zeros(len(filt_te), dtype=np.float64)
    f3 = np.zeros(len(filt_te), dtype=np.float64)
    f7 = np.zeros(len(filt_te), dtype=np.float64)
    f8 = np.zeros(len(filt_te), dtype=np.float64)

    for i, qi in enumerate(filt_te):
        c = int(correct_idx[qi])
        wrongs = [j for j in range(N_OPTIONS) if j != c]

        ext_c = float(ext_base[qi, c])
        ext_w = np.array([float(ext_base[qi, j]) for j in wrongs], dtype=np.float64)
        ext_pair = sigmoid(ext_c - ext_w)

        f1[i] = float(ext_pair.min())
        f2[i] = float(sigmoid(ext_w).max())
        f3[i] = float(sigmoid(ext_c))

        int_c = float(int_proba_base[qi, c])
        int_w = np.array([float(int_proba_base[qi, j]) for j in wrongs], dtype=np.float64)
        int_marg = int_c - int_w
        ext_marg = ext_c - ext_w
        f7[i] = float(int_marg.min())

        rx = pd.Series(int_marg).rank(method="average").to_numpy()
        ry = pd.Series(ext_marg).rank(method="average").to_numpy()
        if rx.std() == 0 or ry.std() == 0:
            f8[i] = 0.0
        else:
            f8[i] = float(np.corrcoef(rx, ry)[0, 1])

    base_layer_cv = df[
        (df["domain"] == "bio")
        & (df["clf"] == "LR")
        & (df["split_type"] == "cv")
        & (df["model_id"] == "base")
        & (df["layer_config"].astype(str).str.startswith("layer_"))
    ][["question_idx", "layer_config", "k_internal"]].copy()
    base_layer_cv["layer"] = (
        base_layer_cv["layer_config"].astype(str).str.replace("layer_", "", regex=False).astype(int)
    )

    ql = base_layer_cv.groupby(["question_idx", "layer"], as_index=False)["k_internal"].mean()
    pivot = ql.pivot(index="question_idx", columns="layer", values="k_internal").sort_index(axis=1)
    layers = pivot.columns.to_numpy()

    f4 = np.full(len(filt_te), np.nan, dtype=np.float64)
    f5 = np.full(len(filt_te), np.nan, dtype=np.float64)
    f6 = np.full(len(filt_te), np.nan, dtype=np.float64)

    for i, qi in enumerate(filt_te):
        if qi not in pivot.index:
            continue
        arr = pivot.loc[qi].to_numpy(dtype=np.float64)
        valid = np.isfinite(arr)
        if not valid.any():
            continue
        a = arr[valid]
        l = layers[valid]
        f5[i] = float(a.mean())
        f6[i] = float(a.std())
        idx = np.where(a >= 0.999999)[0]
        if len(idx):
            f4[i] = float(l[idx[0]])

    return pd.DataFrame(
        {
            "question_idx": filt_te.astype(int),
            "1_min_pairwise_sigmoid_margin": f1,
            "2_max_distractor_confidence": f2,
            "3_correct_abs_confidence": f3,
            "4_earliest_layer_kint_eq_1": f4,
            "5_mean_kint_across_layers": f5,
            "6_std_kint_across_layers": f6,
            "7_min_probe_prob_margin": f7,
            "8_rank_alignment_corr": f8,
        }
    )


def subset_labels_for_method(method, correct_idx, te, filt_te, filt_orig_to_pos, parquet_multi):
    fname = method.replace("/", "_").replace("&", "_")
    ext_path = EXT_DIR / f"{fname}_ck8_bio_ext.npy"
    method_ext = np.load(ext_path)
    mk_ext = compute_k_ext(method_ext, correct_idx, filt_te)

    model_id = f"{fname}_ck8"
    mk_int_series = parquet_multi[parquet_multi["model_id"] == model_id][["question_idx", "k_internal"]]
    if not mk_int_series.empty:
        mk_int = np.full(len(filt_te), np.nan, dtype=np.float32)
        for _, row in mk_int_series.iterrows():
            pos = filt_orig_to_pos.get(int(row["question_idx"]))
            if pos is not None:
                mk_int[pos] = float(row["k_internal"])
        k_int_source = "multi-layer LR"
    else:
        int_path = EXT_DIR / f"{fname}_ck8_bio_int_proba.npy"
        proba = np.load(int_path)
        mk_int = compute_k_int_from_proba(proba, correct_idx, filt_te)
        k_int_source = "layer-26 LR"

    mk_i = mk_int > KNOWS_THRESHOLD
    mk_e = mk_ext > KNOWS_THRESHOLD
    subsets = np.full(len(filt_te), "forgotten", dtype=object)
    subsets[mk_i & mk_e] = "retained"
    subsets[mk_i & ~mk_e] = "suppressed"
    subsets[~mk_i & mk_e] = "lucky"
    return subsets, k_int_source


def plot_method(method, data, k_int_source):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    rng = np.random.default_rng(SEED)

    for idx, feature in enumerate(FEATURE_ORDER):
        ax = axes[idx // 4, idx % 4]
        col_data = []
        for s in ORDER:
            vals = data.loc[data["subset"] == s, feature].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            col_data.append(vals)

        bp = ax.boxplot(
            col_data,
            labels=ORDER,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
        )
        for box, s in zip(bp["boxes"], ORDER):
            box.set_facecolor(SUBSET_COLORS[s])
            box.set_alpha(0.45)
            box.set_edgecolor("#333333")

        for x_pos, vals, s in zip(range(1, len(ORDER) + 1), col_data, ORDER):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full(len(vals), x_pos) + jitter,
                vals,
                s=8,
                color=SUBSET_COLORS[s],
                alpha=0.4,
                linewidths=0,
                rasterized=True,
            )

        ax.set_title(FEATURE_LABELS[feature], fontsize=10)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    counts = {s: int((data["subset"] == s).sum()) for s in ORDER}
    fig.suptitle(
        f"{method} (ck8) — Base features by subset (n={len(data)})\n"
        f"retained={counts['retained']}, suppressed={counts['suppressed']}, "
        f"forgotten={counts['forgotten']}, lucky={counts['lucky']} | K_int: {k_int_source}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)

    fname = method.replace("/", "_").replace("&", "_")
    out_png = OUT_DIR / f"feature_subset_violin_box_{fname}.png"
    out_pdf = OUT_DIR / f"feature_subset_violin_box_{fname}.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png.name}, {out_pdf.name}")


def main():
    correct_idx = load_correct_idx()
    te = np.random.default_rng(SEED).permutation(1273)[TRAIN_SIZE + VAL_SIZE :]
    orig_to_pos = {int(qi): i for i, qi in enumerate(te)}

    df = pd.read_parquet(PARQUET)
    parquet_multi = df[
        (df["domain"] == "bio")
        & (df["clf"] == "LR")
        & (df["layer_config"] == "full")
        & (df["split_type"] == "single")
    ]

    base_single = parquet_multi[parquet_multi["model_id"] == "base"][["question_idx", "k_internal", "k_external"]]
    filt_mask = np.zeros(len(te), dtype=bool)
    for _, row in base_single.iterrows():
        pos = orig_to_pos.get(int(row["question_idx"]))
        if pos is not None and float(row["k_internal"]) == 1.0 and float(row["k_external"]) == 1.0:
            filt_mask[pos] = True
    filt_te = te[filt_mask]
    filt_orig_to_pos = {int(qi): i for i, qi in enumerate(filt_te)}
    print(f"Filtered questions: {len(filt_te)} / {len(te)}")

    base_features = compute_base_features(correct_idx, te, filt_te, df)

    for method in METHODS:
        fname = method.replace("/", "_").replace("&", "_")
        ext_path = EXT_DIR / f"{fname}_ck8_bio_ext.npy"
        if not ext_path.exists():
            print(f"Skip {method}: missing ext file")
            continue

        subsets, k_int_source = subset_labels_for_method(
            method, correct_idx, te, filt_te, filt_orig_to_pos, parquet_multi
        )
        d = base_features.copy()
        d["subset"] = subsets
        plot_method(method, d, k_int_source)

    print("Done.")


if __name__ == "__main__":
    main()
