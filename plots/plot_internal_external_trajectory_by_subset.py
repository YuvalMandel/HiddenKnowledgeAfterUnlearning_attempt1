#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hk_utils import (
    REPO,
    EXT_DIR,
    METHODS,
    SUBSET_ORDER,
    method_fname,
    compute_k_ext,
    load_correct_idx,
    load_split_indices,
    load_scores_df,
    compute_prefilter_mask,
    get_parquet_multi_single,
    get_cv_layer_df,
    get_method_ck8_labels,
)


OUT_DIR = REPO / "plots" / "internal_external_trajectory_by_subset"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MID_LAYERS = set(range(8, 25))


def ckpt_label(model_id, method):
    if model_id == "base":
        return "base"
    mf = method_fname(method)
    if model_id.startswith(f"{mf}_ck"):
        return model_id.replace(f"{mf}_", "")
    return None


def mean_internal_mid_by_ckpt(cv_sub, method, subset_qidx):
    mf = method_fname(method)
    ids = {"base"} | {f"{mf}_ck{i}" for i in range(1, 9)}
    sub = cv_sub[
        (cv_sub["model_id"].isin(ids))
        & (cv_sub["question_idx"].isin(subset_qidx))
        & (cv_sub["layer_idx"].isin(MID_LAYERS))
    ].copy()
    if sub.empty:
        return {}
    sub["checkpoint"] = sub["model_id"].map(lambda x: ckpt_label(x, method))
    fold_mean = sub.groupby(["checkpoint", "question_idx", "fold"], as_index=False)["k_internal"].mean()
    qmean = fold_mean.groupby(["checkpoint", "question_idx"], as_index=False)["k_internal"].mean()
    cmean = qmean.groupby("checkpoint")["k_internal"].mean().to_dict()
    return cmean


def mean_external_by_ckpt(correct_idx, method, subset_qidx):
    ext_vals = {}
    ext_vals["base"] = compute_k_ext(np.load(EXT_DIR / "base_bio_ext.npy"), correct_idx, np.array(sorted(subset_qidx)))
    mf = method_fname(method)
    for ck in range(1, 9):
        path = EXT_DIR / f"{mf}_ck{ck}_bio_ext.npy"
        if not path.exists():
            continue
        arr = np.load(path)
        ext_vals[f"ck{ck}"] = compute_k_ext(arr, correct_idx, np.array(sorted(subset_qidx)))
    return {k: float(np.mean(v)) for k, v in ext_vals.items()}


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

    ck_order = ["base"] + [f"ck{i}" for i in range(1, 9)]
    x = np.arange(len(ck_order))

    for method in METHODS:
        labels, _, _ = get_method_ck8_labels(method, correct_idx, filt_te, filt_orig_to_pos, parquet_multi)
        if labels is None:
            continue

        subset_q = {s: set(int(qi) for qi, lab in zip(filt_te, labels) if lab == s) for s in SUBSET_ORDER}
        fig, axes = plt.subplots(4, 1, figsize=(8.5, 10), sharex=True)

        for i, subset in enumerate(SUBSET_ORDER):
            ax = axes[i]
            qidx = subset_q[subset]
            if not qidx:
                ax.text(0.5, 0.5, f"{subset}: no data", transform=ax.transAxes, ha="center", va="center")
                continue
            int_traj = mean_internal_mid_by_ckpt(cv, method, qidx)
            ext_traj = mean_external_by_ckpt(correct_idx, method, qidx)
            y_int = [int_traj.get(c, np.nan) for c in ck_order]
            y_ext = [ext_traj.get(c, np.nan) for c in ck_order]
            ax.plot(x, y_int, marker="o", label="Internal K (mid layers)")
            ax.plot(x, y_ext, marker="s", label="External K")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3, linestyle=":")
            ax.set_ylabel("K")
            ax.set_title(f"{subset} (n={len(qidx)})", fontsize=10)
            ax.legend(loc="lower left", fontsize=8)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(ck_order, rotation=0)
        axes[-1].set_xlabel("Checkpoint")
        fig.suptitle(f"{method}: internal (mid-layer) vs external trajectory by ck8 subset", fontsize=12)
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        stem = OUT_DIR / f"{method_fname(method)}"
        fig.savefig(f"{stem}.png", dpi=180, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    print("Saved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
