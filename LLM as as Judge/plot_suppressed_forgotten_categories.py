from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PAIR_DATASET = ROOT.parent / "plots" / "pair_level_suppressed_forgotten" / "pair_level_dataset.csv"
CATEGORY_DATASET = ROOT / "wmdp_bio_inferred_categories_first_pass.csv"
OUT_DIR = ROOT / "figures"


def load_joined() -> pd.DataFrame:
    pair = pd.read_csv(PAIR_DATASET)
    cat = pd.read_csv(CATEGORY_DATASET)

    q = (
        pair.loc[pair["checkpoint"].eq("ck8"), ["method", "question_idx", "subset_ck8"]]
        .drop_duplicates()
        .copy()
    )
    if q.duplicated(["method", "question_idx"]).any():
        raise ValueError("Non-unique (method, question_idx) assignments at ck8.")

    cat = cat[["question_id", "category_first_pass"]].copy()
    merged = q.merge(cat, left_on="question_idx", right_on="question_id", how="left")
    return merged[merged["subset_ck8"].isin(["suppressed", "forgotten"])].copy()


def summarize_overall(sf: pd.DataFrame) -> pd.DataFrame:
    counts = sf.groupby(["subset_ck8", "category_first_pass"]).size().rename("count").reset_index()
    totals = counts.groupby("subset_ck8")["count"].sum().rename("subset_total").reset_index()
    counts = counts.merge(totals, on="subset_ck8", how="left")
    counts["pct_within_subset"] = counts["count"] / counts["subset_total"]

    p = counts.pivot(index="category_first_pass", columns="subset_ck8", values="pct_within_subset").fillna(0)
    c = counts.pivot(index="category_first_pass", columns="subset_ck8", values="count").fillna(0)

    out = pd.DataFrame(index=sorted(p.index))
    out["suppressed_count"] = c.get("suppressed", 0)
    out["forgotten_count"] = c.get("forgotten", 0)
    out["suppressed_pct"] = p.get("suppressed", 0)
    out["forgotten_pct"] = p.get("forgotten", 0)
    out["pct_point_diff_supp_minus_forg"] = (out["suppressed_pct"] - out["forgotten_pct"]) * 100
    out = out.reset_index().rename(columns={"index": "category_first_pass"})

    supp_total = int(out["suppressed_count"].sum())
    forg_total = int(out["forgotten_count"].sum())
    pvals = []
    for _, r in out.iterrows():
        s = int(r["suppressed_count"])
        f = int(r["forgotten_count"])
        table = [[s, supp_total - s], [f, forg_total - f]]
        _, pval, _, _ = chi2_contingency(table, correction=False)
        pvals.append(pval)
    out["p_value"] = pvals

    m = len(out)
    out = out.sort_values("p_value").reset_index(drop=True)
    out["rank"] = np.arange(1, m + 1)
    out["q_value_bh"] = (out["p_value"] * m / out["rank"]).clip(upper=1.0)
    out["q_value_bh"] = out["q_value_bh"][::-1].cummin()[::-1]
    return out.sort_values("pct_point_diff_supp_minus_forg", ascending=False).reset_index(drop=True)


def summarize_by_method(sf: pd.DataFrame) -> pd.DataFrame:
    counts = sf.groupby(["method", "subset_ck8", "category_first_pass"]).size().rename("count").reset_index()
    totals = sf.groupby(["method", "subset_ck8"]).size().rename("subset_total").reset_index()
    out = counts.merge(totals, on=["method", "subset_ck8"], how="left")
    out["pct_within_method_subset"] = out["count"] / out["subset_total"]
    return out


def plot_overall_bars(overall: pd.DataFrame, output_path: Path) -> None:
    d = overall.sort_values("pct_point_diff_supp_minus_forg")
    y = np.arange(len(d))
    vals = d["pct_point_diff_supp_minus_forg"].to_numpy()
    colors = np.where(vals >= 0, "#2E8B57", "#B22222")
    labels = d["category_first_pass"].to_list()
    sig = d["q_value_bh"].to_numpy() < 0.05

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Suppressed - Forgotten (percentage points)")
    ax.set_title("Category Enrichment in Suppressed vs Forgotten")

    for i, (v, is_sig) in enumerate(zip(vals, sig)):
        mark = " *" if is_sig else ""
        x = v + (0.35 if v >= 0 else -0.35)
        ha = "left" if v >= 0 else "right"
        ax.text(x, i, f"{v:+.2f}pp{mark}", va="center", ha=ha, fontsize=9)

    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_method_heatmap(by_method: pd.DataFrame, output_path: Path) -> None:
    p = by_method.pivot_table(
        index=["method", "category_first_pass"],
        columns="subset_ck8",
        values="pct_within_method_subset",
        fill_value=0,
    ).reset_index()
    p["pp_diff"] = (p.get("suppressed", 0) - p.get("forgotten", 0)) * 100
    mat = p.pivot(index="method", columns="category_first_pass", values="pp_diff").fillna(0)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    vmax = float(np.nanmax(np.abs(mat.to_numpy())))
    vmax = max(vmax, 1e-6)
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title("Per-Method Category Shift (Suppressed - Forgotten, pp)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if abs(val) >= 8:
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Percentage-point difference")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_method_consistency(by_method: pd.DataFrame, output_path: Path) -> None:
    p = by_method.pivot_table(
        index=["method", "category_first_pass"],
        columns="subset_ck8",
        values="pct_within_method_subset",
        fill_value=0,
    ).reset_index()
    p["pp_diff"] = (p.get("suppressed", 0) - p.get("forgotten", 0)) * 100

    summary = (
        p.groupby("category_first_pass")["pp_diff"]
        .agg(mean_pp_diff="mean", std_pp_diff="std", min_pp_diff="min", max_pp_diff="max")
        .reset_index()
        .sort_values("mean_pp_diff", ascending=False)
    )

    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.errorbar(
        summary["mean_pp_diff"],
        y,
        xerr=summary["std_pp_diff"].fillna(0),
        fmt="o",
        color="#1f4e79",
        ecolor="#7aa6d1",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["category_first_pass"])
    ax.set_xlabel("Mean (Suppressed - Forgotten) percentage points across methods")
    ax.set_title("Category Shift Consistency Across Methods (mean +/- SD)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sf = load_joined()
    overall = summarize_overall(sf)
    by_method = summarize_by_method(sf)

    overall.to_csv(ROOT / "suppressed_vs_forgotten_by_category_overall_with_stats.csv", index=False)
    by_method.to_csv(ROOT / "suppressed_vs_forgotten_by_category_by_method.csv", index=False)

    plot_overall_bars(overall, OUT_DIR / "suppressed_vs_forgotten_overall_category_shift.png")
    plot_method_heatmap(by_method, OUT_DIR / "suppressed_vs_forgotten_method_category_heatmap.png")
    plot_method_consistency(by_method, OUT_DIR / "suppressed_vs_forgotten_category_consistency.png")

    print("Wrote:")
    print(ROOT / "suppressed_vs_forgotten_by_category_overall_with_stats.csv")
    print(ROOT / "suppressed_vs_forgotten_by_category_by_method.csv")
    print(OUT_DIR / "suppressed_vs_forgotten_overall_category_shift.png")
    print(OUT_DIR / "suppressed_vs_forgotten_method_category_heatmap.png")
    print(OUT_DIR / "suppressed_vs_forgotten_category_consistency.png")


if __name__ == "__main__":
    main()
