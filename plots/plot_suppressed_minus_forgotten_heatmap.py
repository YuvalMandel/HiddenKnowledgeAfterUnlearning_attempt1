#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hk_utils import (
    REPO,
    METHODS,
    method_fname,
    load_correct_idx,
    load_split_indices,
    load_scores_df,
    compute_prefilter_mask,
    get_parquet_multi_single,
    get_cv_layer_df,
    get_method_ck8_labels,
)


OUT_DIR = REPO / "plots" / "suppressed_minus_forgotten_heatmap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ckpt_label(model_id, method):
    if model_id == "base":
        return "base"
    mf = method_fname(method)
    if model_id.startswith(f"{mf}_ck"):
        return model_id.replace(f"{mf}_", "")
    return None


def build_grid(cv_sub, method, qidx):
    mf = method_fname(method)
    ids = {"base"} | {f"{mf}_ck{i}" for i in range(1, 9)}
    sub = cv_sub[(cv_sub["model_id"].isin(ids)) & (cv_sub["question_idx"].isin(qidx))].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["checkpoint"] = sub["model_id"].map(lambda x: ckpt_label(x, method))
    sub = sub[sub["checkpoint"].notna()]
    fold_mean = sub.groupby(["checkpoint", "layer_idx", "fold"], as_index=False)["k_internal"].mean()
    ck_layer = fold_mean.groupby(["checkpoint", "layer_idx"], as_index=False)["k_internal"].mean()
    piv = ck_layer.pivot(index="checkpoint", columns="layer_idx", values="k_internal")
    order = ["base"] + [f"ck{i}" for i in range(1, 9)]
    present = [o for o in order if o in piv.index]
    return piv.loc[present]


def main():
    correct_idx = load_correct_idx()
    te, orig_to_pos = load_split_indices()
    df = load_scores_df()
    cv = get_cv_layer_df(df)
    cv = cv[cv["layer_idx"] > 0].copy()
    parquet_multi = get_parquet_multi_single(df)
    filt_mask = compute_prefilter_mask(te, orig_to_pos, df)
    filt_te = te[filt_mask]
    filt_orig_to_pos = {int(qi): i for i, qi in enumerate(filt_te)}

    for method in METHODS:
        labels, _, _ = get_method_ck8_labels(method, correct_idx, filt_te, filt_orig_to_pos, parquet_multi)
        if labels is None:
            continue
        sup = set(int(qi) for qi, lab in zip(filt_te, labels) if lab == "suppressed")
        fog = set(int(qi) for qi, lab in zip(filt_te, labels) if lab == "forgotten")
        if len(sup) == 0 or len(fog) == 0:
            continue
        g_sup = build_grid(cv, method, sup)
        g_fog = build_grid(cv, method, fog)
        if g_sup.empty or g_fog.empty:
            continue
        common_idx = [i for i in g_sup.index if i in g_fog.index]
        common_cols = [c for c in g_sup.columns if c in g_fog.columns]
        if not common_idx or not common_cols:
            continue
        diff = g_sup.loc[common_idx, common_cols] - g_fog.loc[common_idx, common_cols]

        vmax = float(np.nanmax(np.abs(diff.values)))
        vmax = max(vmax, 1e-6)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        im = ax.imshow(diff.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(len(common_cols)))
        ax.set_xticklabels(common_cols, fontsize=6, rotation=90)
        ax.set_yticks(range(len(common_idx)))
        ax.set_yticklabels(common_idx, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Checkpoint")
        ax.set_title(f"{method}: suppressed - forgotten mean K_internal")
        fig.colorbar(im, ax=ax, shrink=0.85, label="K_internal difference")
        fig.tight_layout()

        stem = OUT_DIR / f"{method_fname(method)}"
        fig.savefig(f"{stem}.png", dpi=180, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    print("Saved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
