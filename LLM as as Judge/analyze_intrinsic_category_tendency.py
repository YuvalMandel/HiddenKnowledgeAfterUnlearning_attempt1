from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parent
PAIR_DATASET = ROOT.parent / "plots" / "pair_level_suppressed_forgotten" / "pair_level_dataset.csv"
CATEGORY_DATASET = ROOT / "wmdp_bio_inferred_categories_first_pass.csv"
OUT_DIR = ROOT / "intrinsic_question_level_outputs"


def load_question_method_labels() -> pd.DataFrame:
    pair = pd.read_csv(PAIR_DATASET)
    cat = pd.read_csv(CATEGORY_DATASET)[["question_id", "category_first_pass"]].copy()

    q = (
        pair.loc[pair["checkpoint"].eq("ck8"), ["method", "question_idx", "subset_ck8"]]
        .drop_duplicates()
        .copy()
    )
    q = q[q["subset_ck8"].isin(["suppressed", "forgotten"])].copy()
    merged = q.merge(cat, left_on="question_idx", right_on="question_id", how="left")
    return merged


def build_question_level(df: pd.DataFrame) -> pd.DataFrame:
    question_level = (
        df.groupby(["question_idx", "category_first_pass"])["subset_ck8"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["suppressed", "forgotten"]:
        if col not in question_level.columns:
            question_level[col] = 0
    question_level["n_method_votes"] = question_level["suppressed"] + question_level["forgotten"]
    question_level["suppressed_rate"] = question_level["suppressed"] / question_level["n_method_votes"]
    question_level["forgotten_rate"] = question_level["forgotten"] / question_level["n_method_votes"]
    question_level["net_supp_minus_forg_rate"] = (
        question_level["suppressed_rate"] - question_level["forgotten_rate"]
    )
    return question_level


def pooled_intrinsic_summary(question_level: pd.DataFrame) -> pd.DataFrame:
    def ci95(s: pd.Series) -> tuple[float, float]:
        n = len(s)
        m = float(s.mean())
        if n <= 1:
            return m, m
        se = float(s.std(ddof=1) / np.sqrt(n))
        margin = 1.96 * se
        return m - margin, m + margin

    rows = []
    for cat, g in question_level.groupby("category_first_pass"):
        lo, hi = ci95(g["suppressed_rate"])
        net_lo, net_hi = ci95(g["net_supp_minus_forg_rate"])
        rows.append(
            {
                "category_first_pass": cat,
                "n_questions": int(len(g)),
                "mean_suppressed_rate": float(g["suppressed_rate"].mean()),
                "mean_forgotten_rate": float(g["forgotten_rate"].mean()),
                "mean_net_supp_minus_forg_rate": float(g["net_supp_minus_forg_rate"].mean()),
                "suppressed_rate_ci95_low": lo,
                "suppressed_rate_ci95_high": hi,
                "net_rate_ci95_low": net_lo,
                "net_rate_ci95_high": net_hi,
                "total_suppressed_votes": int(g["suppressed"].sum()),
                "total_forgotten_votes": int(g["forgotten"].sum()),
                "total_votes": int(g["n_method_votes"].sum()),
            }
        )

    out = pd.DataFrame(rows).sort_values("mean_net_supp_minus_forg_rate", ascending=False)
    out["suppressed_vote_share"] = out["total_suppressed_votes"] / out["total_votes"]
    out["forgotten_vote_share"] = out["total_forgotten_votes"] / out["total_votes"]
    return out


def per_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["method", "category_first_pass", "subset_ck8"])
        .size()
        .rename("count")
        .reset_index()
    )
    p = g.pivot_table(
        index=["method", "category_first_pass"], columns="subset_ck8", values="count", fill_value=0
    ).reset_index()
    for col in ["suppressed", "forgotten"]:
        if col not in p.columns:
            p[col] = 0
    p["total"] = p["suppressed"] + p["forgotten"]
    p["suppressed_share"] = p["suppressed"] / p["total"]
    p["forgotten_share"] = p["forgotten"] / p["total"]
    return p


def per_method_vs_baseline(per_method: pd.DataFrame) -> pd.DataFrame:
    baseline = per_method.groupby("method")["suppressed_share"].mean().rename("method_baseline").reset_index()
    out = per_method.merge(baseline, on="method", how="left")
    out["suppressed_share_minus_method_baseline"] = out["suppressed_share"] - out["method_baseline"]
    return out


def plot_intrinsic_bar(summary: pd.DataFrame, path: Path) -> None:
    d = summary.copy().sort_values("mean_net_supp_minus_forg_rate")
    y = np.arange(len(d))
    x = d["mean_net_supp_minus_forg_rate"].to_numpy() * 100
    lo = (d["mean_net_supp_minus_forg_rate"] - d["net_rate_ci95_low"]).to_numpy() * 100
    hi = (d["net_rate_ci95_high"] - d["mean_net_supp_minus_forg_rate"]).to_numpy() * 100
    xerr = np.vstack([lo, hi])
    colors = np.where(x >= 0, "#1f77b4", "#d62728")

    fig, ax = plt.subplots(figsize=(10.2, 5))
    ax.barh(y, x, color=colors, alpha=0.9)
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="black", elinewidth=1.1, capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["category_first_pass"])
    ax.set_xlabel("Intrinsic tendency: mean (suppressed_rate - forgotten_rate), percentage points")
    ax.set_title("Question-Level Intrinsic Category Tendency")
    for i, v in enumerate(x):
        ax.text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.1f}pp", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=230)
    plt.close(fig)


def plot_vote_share_bars(summary: pd.DataFrame, path: Path) -> None:
    d = summary.copy().sort_values("suppressed_vote_share", ascending=False)
    x = np.arange(len(d))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - w / 2, d["suppressed_vote_share"] * 100, w, label="suppressed", color="#2E8B57")
    ax.bar(x + w / 2, d["forgotten_vote_share"] * 100, w, label="forgotten", color="#B22222")
    ax.set_xticks(x)
    ax.set_xticklabels(d["category_first_pass"], rotation=28, ha="right")
    ax.set_ylabel("Vote share (%) among suppressed+forgotten")
    ax.set_title("Pooled Method Votes by Category")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=230)
    plt.close(fig)


def plot_method_heatmap(per_method_delta: pd.DataFrame, path: Path) -> None:
    mat = per_method_delta.pivot(
        index="method", columns="category_first_pass", values="suppressed_share_minus_method_baseline"
    ).fillna(0)
    arr = mat.to_numpy() * 100
    vmax = float(np.max(np.abs(arr)))
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    im = ax.imshow(arr, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=28, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title("Per-Method Category Effect (Suppressed Share - Method Baseline), pp")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if abs(val) >= 8:
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8)

    cb = fig.colorbar(im, ax=ax, shrink=0.92)
    cb.set_label("percentage points")
    fig.tight_layout()
    fig.savefig(path, dpi=230)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_question_method_labels()
    question_level = build_question_level(df)
    pooled = pooled_intrinsic_summary(question_level)
    per_method = per_method_summary(df)
    per_method_delta = per_method_vs_baseline(per_method)

    q_path = OUT_DIR / "question_level_suppressed_forgotten_rates.csv"
    pooled_path = OUT_DIR / "pooled_intrinsic_category_summary.csv"
    pm_path = OUT_DIR / "per_method_category_suppressed_forgotten_summary.csv"
    pmd_path = OUT_DIR / "per_method_category_effect_vs_baseline.csv"

    question_level.to_csv(q_path, index=False)
    pooled.to_csv(pooled_path, index=False)
    per_method.to_csv(pm_path, index=False)
    per_method_delta.to_csv(pmd_path, index=False)

    plot_intrinsic_bar(pooled, OUT_DIR / "intrinsic_category_tendency_bar_ci.png")
    plot_vote_share_bars(pooled, OUT_DIR / "pooled_category_vote_share_grouped_bars.png")
    plot_method_heatmap(per_method_delta, OUT_DIR / "per_method_category_effect_heatmap.png")

    print("Wrote:")
    print(q_path)
    print(pooled_path)
    print(pm_path)
    print(pmd_path)
    print(OUT_DIR / "intrinsic_category_tendency_bar_ci.png")
    print(OUT_DIR / "pooled_category_vote_share_grouped_bars.png")
    print(OUT_DIR / "per_method_category_effect_heatmap.png")


if __name__ == "__main__":
    main()
