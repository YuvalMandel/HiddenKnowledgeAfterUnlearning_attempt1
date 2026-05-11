#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from hk_utils import (
    REPO,
    METHODS,
    SUBSET_ORDER,
    method_fname,
    load_correct_idx,
    load_split_indices,
    load_scores_df,
    compute_prefilter_mask,
    get_parquet_multi_single,
    get_cv_layer_df,
    get_method_ck8_labels,
)


OUT_DIR = REPO / "plots" / "ckpt_layer_trajectory_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = {
    "early": list(range(1, 8)),
    "mid": list(range(8, 25)),
    "late": list(range(25, 33)),
    "core_mid": list(range(10, 21)),
}


def bootstrap_ci(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs.append(float(np.mean(aa) - np.mean(bb)))
    diffs = np.asarray(diffs)
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def checkpoint_label(model_id, method):
    if model_id == "base":
        return "base"
    mf = method_fname(method)
    if model_id.startswith(f"{mf}_ck"):
        return model_id.replace(f"{mf}_", "")
    return None


def ordered_ckpts_present(labels):
    order = ["base"] + [f"ck{i}" for i in range(1, 9)]
    return [x for x in order if x in labels]


def compute_question_trajectory_table(cv_df, method, subset_map):
    mf = method_fname(method)
    allowed_ids = {"base"} | {f"{mf}_ck{i}" for i in range(1, 9)}
    sub = cv_df[cv_df["model_id"].isin(allowed_ids)].copy()
    sub["checkpoint"] = sub["model_id"].map(lambda x: checkpoint_label(x, method))
    sub = sub[sub["checkpoint"].notna()]
    sub["subset"] = sub["question_idx"].map(subset_map)
    sub = sub[sub["subset"].isin(SUBSET_ORDER)]
    if sub.empty:
        return pd.DataFrame()
    fold_mean = (
        sub.groupby(["question_idx", "subset", "checkpoint", "layer_idx", "fold"], as_index=False)["k_internal"]
        .mean()
    )
    return (
        fold_mean.groupby(["question_idx", "subset", "checkpoint", "layer_idx"], as_index=False)["k_internal"]
        .mean()
    )


def main():
    correct_idx = load_correct_idx()
    te, orig_to_pos = load_split_indices()
    df = load_scores_df()
    cv_all = get_cv_layer_df(df)
    cv_all = cv_all[cv_all["layer_idx"] > 0].copy()
    parquet_multi = get_parquet_multi_single(df)
    filt_mask = compute_prefilter_mask(te, orig_to_pos, df)
    filt_te = te[filt_mask]
    filt_orig_to_pos = {int(qi): i for i, qi in enumerate(filt_te)}

    all_rows = []
    by_q_rows = []
    stat_rows = []

    for method in METHODS:
        labels, _, _ = get_method_ck8_labels(method, correct_idx, filt_te, filt_orig_to_pos, parquet_multi)
        if labels is None:
            continue
        subset_map = {int(qi): lab for qi, lab in zip(filt_te, labels)}
        qtraj = compute_question_trajectory_table(cv_all, method, subset_map)
        if qtraj.empty:
            continue

        ckpt_order = ["base"] + [f"ck{i}" for i in range(1, 9)]
        for subset in SUBSET_ORDER:
            ss = qtraj[qtraj["subset"] == subset].copy()
            if ss.empty:
                continue
            for band_name, band_layers in BANDS.items():
                sb = ss[ss["layer_idx"].isin(band_layers)].copy()
                if sb.empty:
                    continue
                qck = sb.groupby(["question_idx", "checkpoint"], as_index=False)["k_internal"].mean()
                qpivot = qck.pivot(index="question_idx", columns="checkpoint", values="k_internal")
                for c in ckpt_order:
                    if c not in qpivot.columns:
                        qpivot[c] = np.nan
                qpivot = qpivot[ckpt_order]

                auc = np.nanmean(qpivot.values, axis=1)
                base_vals = qpivot["base"].to_numpy()
                ck8_vals = qpivot["ck8"].to_numpy() if "ck8" in qpivot.columns else np.full(len(qpivot), np.nan)
                delta = ck8_vals - base_vals
                slope = delta / 8.0
                min_ckpt = np.nanmin(qpivot.values, axis=1)

                by_q_rows.extend(
                    [
                        {
                            "method": method,
                            "subset": subset,
                            "layer_band": band_name,
                            "question_idx": int(qi),
                            "auc": float(a),
                            "base": float(b) if np.isfinite(b) else np.nan,
                            "ck8": float(c8) if np.isfinite(c8) else np.nan,
                            "delta_ck8_minus_base": float(d) if np.isfinite(d) else np.nan,
                            "slope_per_ckpt": float(s) if np.isfinite(s) else np.nan,
                            "min_checkpoint_value": float(mn) if np.isfinite(mn) else np.nan,
                        }
                        for qi, a, b, c8, d, s, mn in zip(
                            qpivot.index.to_numpy(), auc, base_vals, ck8_vals, delta, slope, min_ckpt
                        )
                    ]
                )

                all_rows.append(
                    {
                        "method": method,
                        "subset": subset,
                        "layer_band": band_name,
                        "n_questions": int(qpivot.shape[0]),
                        "mean_auc": float(np.nanmean(auc)),
                        "mean_base": float(np.nanmean(base_vals)),
                        "mean_ck8": float(np.nanmean(ck8_vals)),
                        "mean_delta_ck8_minus_base": float(np.nanmean(delta)),
                        "mean_slope_per_ckpt": float(np.nanmean(slope)),
                        "mean_min_checkpoint_value": float(np.nanmean(min_ckpt)),
                    }
                )

            # mid-minus-late at ck8 per subset
            mid = ss[ss["layer_idx"].isin(BANDS["mid"])].groupby("question_idx")["k_internal"].mean()
            late = ss[ss["layer_idx"].isin(BANDS["late"])].groupby("question_idx")["k_internal"].mean()
            idx = sorted(set(mid.index).intersection(set(late.index)))
            if idx:
                mm = (mid.loc[idx] - late.loc[idx]).to_numpy()
                all_rows.append(
                    {
                        "method": method,
                        "subset": subset,
                        "layer_band": "mid_minus_late",
                        "n_questions": int(len(idx)),
                        "mean_auc": np.nan,
                        "mean_base": np.nan,
                        "mean_ck8": np.nan,
                        "mean_delta_ck8_minus_base": np.nan,
                        "mean_slope_per_ckpt": np.nan,
                        "mean_min_checkpoint_value": np.nan,
                        "mean_mid_minus_late": float(np.nanmean(mm)),
                    }
                )

        # suppressed vs forgotten stats
        by_q_df = pd.DataFrame([r for r in by_q_rows if r["method"] == method])
        for band in ["mid", "core_mid"]:
            sup = by_q_df[(by_q_df["subset"] == "suppressed") & (by_q_df["layer_band"] == band)]["auc"].dropna().to_numpy()
            fog = by_q_df[(by_q_df["subset"] == "forgotten") & (by_q_df["layer_band"] == band)]["auc"].dropna().to_numpy()
            if len(sup) < 2 or len(fog) < 2:
                continue
            u, p = mannwhitneyu(sup, fog, alternative="greater")
            mean_diff, lo, hi = bootstrap_ci(sup, fog)
            stat_rows.append(
                {
                    "method": method,
                    "metric": f"{band}_auc",
                    "n_suppressed": len(sup),
                    "n_forgotten": len(fog),
                    "mannwhitney_u": float(u),
                    "p_value_greater": float(p),
                    "mean_diff_supp_minus_forg": float(mean_diff),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                }
            )

    summary_df = pd.DataFrame(all_rows)
    by_q_df = pd.DataFrame(by_q_rows)
    stats_df = pd.DataFrame(stat_rows)

    summary_df.to_csv(OUT_DIR / "trajectory_metrics.csv", index=False)
    by_q_df.to_csv(OUT_DIR / "trajectory_metrics_by_method_subset.csv", index=False)
    stats_df.to_csv(OUT_DIR / "suppressed_vs_forgotten_stats.csv", index=False)

    lines = []
    lines.append("# Trajectory Metrics Summary")
    lines.append("")
    if not summary_df.empty:
        lines.append(f"- rows: {len(summary_df)}")
        mid = summary_df[(summary_df["subset"] == "suppressed") & (summary_df["layer_band"] == "mid")]
        fog = summary_df[(summary_df["subset"] == "forgotten") & (summary_df["layer_band"] == "mid")]
        if not mid.empty and not fog.empty:
            lines.append(
                f"- mean mid-layer AUC (suppressed): {mid['mean_auc'].mean():.4f} | "
                f"(forgotten): {fog['mean_auc'].mean():.4f}"
            )
    (OUT_DIR / "trajectory_metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")

    stats_md = ["# Suppressed vs Forgotten Stats", ""]
    if stats_df.empty:
        stats_md.append("- no valid comparisons")
    else:
        for _, r in stats_df.iterrows():
            stats_md.append(
                f"- {r['method']} {r['metric']}: diff={r['mean_diff_supp_minus_forg']:.4f}, "
                f"p={r['p_value_greater']:.3g}, CI=[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
            )
    (OUT_DIR / "suppressed_vs_forgotten_stats.md").write_text("\n".join(stats_md), encoding="utf-8")
    print("Saved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
