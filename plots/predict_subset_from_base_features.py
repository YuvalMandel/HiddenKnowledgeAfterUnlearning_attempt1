#!/usr/bin/env python3
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hk_utils import (
    REPO,
    EXT_DIR,
    METHODS,
    SUBSET_ORDER,
    SEED,
    N_OPTIONS,
    method_fname,
    sigmoid,
    load_correct_idx,
    load_split_indices,
    load_scores_df,
    compute_prefilter_mask,
    get_parquet_multi_single,
    get_method_ck8_labels,
)


OUT_DIR = REPO / "plots" / "base_feature_prediction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_MAP = {
    "1_min_pairwise_sigmoid_margin": "min_pairwise_sigmoid_margin",
    "2_max_distractor_confidence": "max_distractor_confidence",
    "3_correct_abs_confidence": "correct_option_confidence",
    "4_earliest_layer_kint_eq_1": "earliest_layer_kint1",
    "5_mean_kint_across_layers": "mean_kint_layers",
    "6_std_kint_across_layers": "std_kint_layers",
    "7_min_probe_prob_margin": "min_probe_probability_margin",
    "8_rank_alignment_corr": "rank_alignment",
}

ALL_FEATURES = list(FEATURE_MAP.values())
EXTERNAL_FEATURES = [
    "min_pairwise_sigmoid_margin",
    "max_distractor_confidence",
    "correct_option_confidence",
]
INTERNAL_FEATURES = [
    "earliest_layer_kint1",
    "mean_kint_layers",
    "std_kint_layers",
    "min_probe_probability_margin",
]
ALIGNMENT_FEATURES = ["rank_alignment"]


def choose_cv(y, requested_splits=5):
    min_count = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(requested_splits, min_count))
    if min_count < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def make_pipeline(multinomial=True):
    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=5000,
        random_state=SEED,
    )
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ALL_FEATURES,
            )
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def compute_base_features(correct_idx, filt_te, df):
    ext_base = np.load(EXT_DIR / "base_bio_ext.npy")
    int_proba_base = np.load(EXT_DIR / "base_bio_int_proba.npy")

    f1, f2, f3, f7, f8 = [], [], [], [], []
    for qi in filt_te:
        c = int(correct_idx[qi])
        wrongs = [j for j in range(N_OPTIONS) if j != c]

        ext_c = float(ext_base[qi, c])
        ext_w = np.array([float(ext_base[qi, j]) for j in wrongs], dtype=np.float64)
        ext_pair = sigmoid(ext_c - ext_w)
        f1.append(float(ext_pair.min()))
        f2.append(float(sigmoid(ext_w).max()))
        f3.append(float(sigmoid(ext_c)))

        int_c = float(int_proba_base[qi, c])
        int_w = np.array([float(int_proba_base[qi, j]) for j in wrongs], dtype=np.float64)
        int_marg = int_c - int_w
        ext_marg = ext_c - ext_w
        f7.append(float(int_marg.min()))

        rx = pd.Series(int_marg).rank(method="average").to_numpy()
        ry = pd.Series(ext_marg).rank(method="average").to_numpy()
        if rx.std() == 0 or ry.std() == 0:
            f8.append(0.0)
        else:
            f8.append(float(np.corrcoef(rx, ry)[0, 1]))

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

    f4, f5, f6 = [], [], []
    for qi in filt_te:
        if qi not in pivot.index:
            f4.append(np.nan)
            f5.append(np.nan)
            f6.append(np.nan)
            continue
        arr = pivot.loc[qi].to_numpy(dtype=np.float64)
        valid = np.isfinite(arr)
        if not valid.any():
            f4.append(np.nan)
            f5.append(np.nan)
            f6.append(np.nan)
            continue
        a = arr[valid]
        l = layers[valid]
        f5.append(float(a.mean()))
        f6.append(float(a.std()))
        idx = np.where(a >= 0.999999)[0]
        f4.append(float(l[idx[0]]) if len(idx) else np.nan)

    data = pd.DataFrame(
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
    return data.rename(columns=FEATURE_MAP)


def fit_eval_multinomial(X, y, method):
    y = np.asarray(y, dtype=str)
    cv = choose_cv(y)
    if cv is None:
        return None, None, "too few samples per class"
    pipe = make_pipeline(multinomial=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
    labels = SUBSET_ORDER
    cm = confusion_matrix(y, y_pred, labels=labels)
    metrics = {
        "method": method,
        "n": len(y),
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        **{f"f1_{c}": f1_score(y, y_pred, labels=[c], average="macro", zero_division=0) for c in labels},
    }
    return metrics, cm, None


def fit_eval_binary(X, y_bin):
    y_bin = np.asarray(y_bin, dtype=int)
    cv = choose_cv(y_bin)
    if cv is None:
        return None, None, None
    pipe = make_pipeline(multinomial=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(pipe, X, y_bin, cv=cv, method="predict")
        y_prob = cross_val_predict(pipe, X, y_bin, cv=cv, method="predict_proba")[:, 1]
    roc = roc_auc_score(y_bin, y_prob) if len(np.unique(y_bin)) == 2 else np.nan
    pr = average_precision_score(y_bin, y_prob) if len(np.unique(y_bin)) == 2 else np.nan
    metrics = {
        "roc_auc": roc,
        "pr_auc": pr,
        "balanced_accuracy": balanced_accuracy_score(y_bin, y_pred),
        "f1": f1_score(y_bin, y_pred, zero_division=0),
    }
    return metrics, y_prob, y_pred


def plot_confusion(cm, labels, title, out_stem):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def fit_coefs_table(X, y, task_name, method_name):
    pipe = make_pipeline(multinomial=(len(np.unique(y)) > 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)
    coefs = pipe.named_steps["clf"].coef_
    rows = []
    classes = pipe.named_steps["clf"].classes_
    for ci, cname in enumerate(classes):
        row_idx = ci if coefs.ndim == 2 and ci < coefs.shape[0] else 0
        for fi, fname in enumerate(ALL_FEATURES):
            rows.append(
                {
                    "method": method_name,
                    "task": task_name,
                    "class_or_positive": str(cname),
                    "feature": fname,
                    "coef": float(coefs[row_idx, fi] if coefs.ndim == 2 else coefs[fi]),
                }
            )
    return pd.DataFrame(rows)


def main():
    correct_idx = load_correct_idx()
    te, orig_to_pos = load_split_indices()
    df = load_scores_df()
    parquet_multi = get_parquet_multi_single(df)
    filt_mask = compute_prefilter_mask(te, orig_to_pos, df)
    filt_te = te[filt_mask]
    filt_orig_to_pos = {int(qi): i for i, qi in enumerate(filt_te)}

    base_features = compute_base_features(correct_idx, filt_te, df)

    multinomial_rows, binary_rows, coef_tables, warnings_rows = [], [], [], []
    roc_points = []

    all_methods_data = []
    for method in METHODS:
        labels, _, k_src = get_method_ck8_labels(method, correct_idx, filt_te, filt_orig_to_pos, parquet_multi)
        if labels is None:
            warnings_rows.append({"method": method, "warning": "missing ck8 files"})
            continue
        data = base_features.copy()
        data["method"] = method
        data["subset"] = labels
        data["k_int_source"] = k_src
        all_methods_data.append(data)

        X = data[ALL_FEATURES]
        y = data["subset"].values

        m_metrics, cm, warn = fit_eval_multinomial(X, y, method)
        if warn:
            warnings_rows.append({"method": method, "warning": f"multinomial skipped: {warn}"})
        else:
            multinomial_rows.append(m_metrics)
            plot_confusion(
                cm,
                SUBSET_ORDER,
                f"{method} multinomial confusion",
                OUT_DIR / f"confusion_matrix_{method_fname(method)}",
            )
            coef_tables.append(fit_coefs_table(X, y, "multinomial", method))

        binary_tasks = [
            ("suppressed_vs_forgotten", {"suppressed": 1, "forgotten": 0}),
            ("suppressed_vs_retained", {"suppressed": 1, "retained": 0}),
            ("internally_survives_vs_erased", {"retained": 1, "suppressed": 1, "forgotten": 0, "lucky": 0}),
        ]
        for task_name, mapping in binary_tasks:
            dsub = data[data["subset"].isin(mapping.keys())].copy()
            y_bin = dsub["subset"].map(mapping).astype(int).values
            X_bin = dsub[ALL_FEATURES]
            res, y_prob, _ = fit_eval_binary(X_bin, y_bin)
            if res is None:
                warnings_rows.append({"method": method, "warning": f"{task_name} skipped: too few samples"})
                continue
            row = {"method": method, "task": task_name, "n": len(dsub), **res}
            binary_rows.append(row)
            coef_tables.append(fit_coefs_table(X_bin, y_bin, task_name, method))
            if task_name == "suppressed_vs_forgotten":
                roc_points.append((method, y_bin, y_prob))

        groups = [
            ("external_only", EXTERNAL_FEATURES),
            ("internal_only", INTERNAL_FEATURES),
            ("alignment_only", ALIGNMENT_FEATURES),
            ("all_features", ALL_FEATURES),
        ]
        dsub = data[data["subset"].isin(["suppressed", "forgotten"])].copy()
        y_bin = dsub["subset"].map({"suppressed": 1, "forgotten": 0}).astype(int).values
        for gname, cols in groups:
            cv = choose_cv(y_bin)
            if cv is None:
                continue
            clf = LogisticRegression(
                solver="lbfgs", class_weight="balanced", max_iter=5000, random_state=SEED
            )
            pre = ColumnTransformer(
                [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cols)]
            )
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            y_pred = cross_val_predict(pipe, dsub[ALL_FEATURES], y_bin, cv=cv, method="predict")
            if "ablation_rows" not in locals():
                ablation_rows = []
            ablation_rows.append(
                {
                    "method": method,
                    "task": "suppressed_vs_forgotten",
                    "feature_group": gname,
                    "n": len(dsub),
                    "balanced_accuracy": balanced_accuracy_score(y_bin, y_pred),
                    "f1": f1_score(y_bin, y_pred, zero_division=0),
                }
            )

    if not all_methods_data:
        raise RuntimeError("No method data available.")
    all_df = pd.concat(all_methods_data, ignore_index=True)
    all_df.to_csv(OUT_DIR / "pooled_features.csv", index=False)

    # Leave-one-method-out (LOMO) on 4-way task
    lomo_rows = []
    for held in METHODS:
        tr = all_df[all_df["method"] != held]
        te_df = all_df[all_df["method"] == held]
        if len(te_df) == 0:
            continue
        pipe = make_pipeline(multinomial=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(tr[ALL_FEATURES], tr["subset"])
            pred = pipe.predict(te_df[ALL_FEATURES])
        lomo_rows.append(
            {
                "held_out_method": held,
                "n_test": len(te_df),
                "balanced_accuracy": balanced_accuracy_score(te_df["subset"], pred),
                "macro_f1": f1_score(te_df["subset"], pred, average="macro", zero_division=0),
            }
        )

    # Save tables
    pd.DataFrame(multinomial_rows).to_csv(OUT_DIR / "multinomial_metrics.csv", index=False)
    pd.DataFrame(binary_rows).to_csv(OUT_DIR / "binary_metrics.csv", index=False)
    pd.DataFrame(locals().get("ablation_rows", [])).to_csv(OUT_DIR / "ablation_metrics.csv", index=False)
    pd.DataFrame(lomo_rows).to_csv(OUT_DIR / "leave_one_method_out_metrics.csv", index=False)
    pd.concat(coef_tables, ignore_index=True).to_csv(OUT_DIR / "feature_coefficients.csv", index=False)
    pd.DataFrame(warnings_rows).to_csv(OUT_DIR / "warnings.csv", index=False)

    # Plot ROC for suppressed vs forgotten
    fig, ax = plt.subplots(figsize=(6, 5))
    from sklearn.metrics import roc_curve

    for method, y_true, y_prob in roc_points:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{method} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Suppressed vs Forgotten ROC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "binary_roc_suppressed_vs_forgotten.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT_DIR / "binary_roc_suppressed_vs_forgotten.pdf", bbox_inches="tight")
    plt.close(fig)

    # Feature importance from pooled suppressed-vs-forgotten
    pooled = all_df[all_df["subset"].isin(["suppressed", "forgotten"])].copy()
    y_pool = pooled["subset"].map({"suppressed": 1, "forgotten": 0}).astype(int).values
    pipe = make_pipeline(multinomial=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(pooled[ALL_FEATURES], y_pool)
    coef = pipe.named_steps["clf"].coef_[0]
    imp = pd.DataFrame({"feature": ALL_FEATURES, "coef": coef, "abs_coef": np.abs(coef)}).sort_values(
        "abs_coef", ascending=False
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp["feature"], imp["coef"])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Feature coefficients: suppressed vs forgotten (pooled)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance_suppressed_vs_forgotten.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT_DIR / "feature_importance_suppressed_vs_forgotten.pdf", bbox_inches="tight")
    plt.close(fig)

    # Ablation + LOMO bars
    abl_df = pd.DataFrame(locals().get("ablation_rows", []))
    if not abl_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        g = abl_df.groupby("feature_group")["balanced_accuracy"].mean().reindex(
            ["external_only", "internal_only", "alignment_only", "all_features"]
        )
        ax.bar(g.index, g.values)
        ax.set_ylim(0, 1)
        ax.set_title("Ablation balanced accuracy (suppressed vs forgotten)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "ablation_bar.png", dpi=180, bbox_inches="tight")
        fig.savefig(OUT_DIR / "ablation_bar.pdf", bbox_inches="tight")
        plt.close(fig)

    lomo_df = pd.DataFrame(lomo_rows)
    if not lomo_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(lomo_df["held_out_method"], lomo_df["balanced_accuracy"])
        ax.set_ylim(0, 1)
        ax.set_title("Leave-one-method-out balanced accuracy")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "leave_one_method_out_bar.png", dpi=180, bbox_inches="tight")
        fig.savefig(OUT_DIR / "leave_one_method_out_bar.pdf", bbox_inches="tight")
        plt.close(fig)

    print("Saved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
