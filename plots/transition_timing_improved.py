#!/usr/bin/env python3
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
SUBSETS = ["retained", "suppressed", "forgotten", "lucky"]
SUBSET_COLORS = {
    "retained": "#2ca02c",
    "suppressed": "#d62728",
    "forgotten": "#7f7f7f",
    "lucky": "#1f77b4",
}
TICKS = list(range(10))
TICK_LABELS = ["base", "ck1", "ck2", "ck3", "ck4", "ck5", "ck6", "ck7", "ck8", "none"]


def load_transition_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    # fallback search
    cands = sorted(path.parent.glob("*transition*timing*question*.csv"))
    if cands:
        warnings.warn(f"Input not found at {path}; using fallback {cands[0]}")
        return pd.read_csv(cands[0])
    raise FileNotFoundError(f"Could not find transition timing input near: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "final_subset": "subset",
        "method_name": "method",
    }
    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    required = ["method", "question_idx", "subset", "t_ext_drop", "t_int_drop", "suppression_lag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}; found: {sorted(df.columns.tolist())}")

    if "ext_censored" not in df.columns:
        df["ext_censored"] = df["t_ext_drop"] >= 9
    if "int_censored" not in df.columns:
        df["int_censored"] = df["t_int_drop"] >= 9

    df = df.copy()
    df["method"] = df["method"].astype(str)
    df["subset"] = df["subset"].astype(str)
    for c in ["t_ext_drop", "t_int_drop", "suppression_lag"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t_ext_drop", "t_int_drop", "suppression_lag"])
    return df


def style_axes(ax):
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.set_xticklabels(TICK_LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(TICK_LABELS, fontsize=8)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.grid(alpha=0.15, linewidth=0.5)


def make_bubble_scatter(df: pd.DataFrame, outdir: Path):
    agg = (
        df.groupby(["method", "subset", "t_ext_drop", "t_int_drop"], as_index=False)
        .size()
        .rename(columns={"size": "n_questions"})
    )

    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True, sharey=True)
    axes = axes.ravel()
    size_scale = 18.0

    for i, method in enumerate(METHODS):
        ax = axes[i]
        m = agg[agg["method"] == method]
        if m.empty:
            ax.set_title(f"{method} (no data)", fontsize=9)
            ax.axis("off")
            continue
        for subset in SUBSETS:
            g = m[m["subset"] == subset]
            if g.empty:
                continue
            ax.scatter(
                g["t_ext_drop"],
                g["t_int_drop"],
                s=np.maximum(20.0, g["n_questions"].to_numpy() * size_scale),
                alpha=0.65,
                color=SUBSET_COLORS.get(subset, "#333333"),
                edgecolor="black",
                linewidth=0.3,
                label=subset,
            )
        ax.plot([0, 9], [0, 9], linestyle="--", color="black", linewidth=1.0, alpha=0.8)
        ax.set_title(method, fontsize=10)
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.985), title="Final subset")
    fig.suptitle("Transition Timing Bubble Scatter by Method", fontsize=14, y=0.997)
    fig.supxlabel("t_ext_drop (first checkpoint where K_external <= 0.5)")
    fig.supylabel("t_int_drop (first checkpoint where K_internal <= 0.5)")
    fig.tight_layout(rect=[0.03, 0.06, 1, 0.94])
    fig.savefig(outdir / "transition_bubble_scatter_by_method.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / "transition_bubble_scatter_by_method.pdf", bbox_inches="tight")
    plt.close(fig)


def make_count_heatmaps(df: pd.DataFrame, outdir: Path):
    # per-method files
    for method in METHODS:
        m = df[df["method"] == method]
        if m.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
        axes = axes.ravel()
        v_max = 1
        mats = {}
        for subset in SUBSETS:
            sm = m[m["subset"] == subset]
            mat = np.zeros((10, 10), dtype=float)
            for _, r in sm.iterrows():
                x = int(r["t_ext_drop"])
                y = int(r["t_int_drop"])
                if 0 <= x <= 9 and 0 <= y <= 9:
                    mat[y, x] += 1
            mats[subset] = mat
            v_max = max(v_max, int(mat.max()))

        for i, subset in enumerate(SUBSETS):
            ax = axes[i]
            mat = mats[subset]
            im = ax.imshow(mat, origin="lower", aspect="equal", cmap="YlOrRd", vmin=0, vmax=v_max)
            ax.plot([0, 9], [0, 9], linestyle="--", color="black", linewidth=0.8, alpha=0.8)
            ax.set_title(f"{subset}", fontsize=10)
            style_axes(ax)

        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.022, pad=0.02)
        cbar.set_label("Question count", rotation=90)
        fig.suptitle(f"{method}: Transition Timing Count Heatmap by Final Subset", fontsize=13, y=0.995)
        fig.supxlabel("t_ext_drop")
        fig.supylabel("t_int_drop")
        fig.subplots_adjust(left=0.08, right=0.90, bottom=0.10, top=0.93, wspace=0.18, hspace=0.22)
        stem = outdir / f"{method}_transition_count_heatmap_by_subset"
        fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    # contact sheet (method x subset)
    fig, axes = plt.subplots(len(METHODS), len(SUBSETS), figsize=(16, 24), sharex=True, sharey=True)
    global_vmax = 1
    mats = {}
    for method in METHODS:
        m = df[df["method"] == method]
        for subset in SUBSETS:
            mat = np.zeros((10, 10), dtype=float)
            sm = m[m["subset"] == subset]
            for _, r in sm.iterrows():
                x = int(r["t_ext_drop"])
                y = int(r["t_int_drop"])
                if 0 <= x <= 9 and 0 <= y <= 9:
                    mat[y, x] += 1
            mats[(method, subset)] = mat
            global_vmax = max(global_vmax, int(mat.max()))

    last_im = None
    for i, method in enumerate(METHODS):
        for j, subset in enumerate(SUBSETS):
            ax = axes[i, j]
            mat = mats[(method, subset)]
            last_im = ax.imshow(mat, origin="lower", aspect="equal", cmap="YlGnBu", vmin=0, vmax=global_vmax)
            ax.plot([0, 9], [0, 9], linestyle="--", color="black", linewidth=0.6, alpha=0.7)
            if i == 0:
                ax.set_title(subset, fontsize=10)
            if j == 0:
                ax.set_ylabel(method, fontsize=9)
            ax.set_xticks([0, 3, 6, 9])
            ax.set_xticklabels(["base", "ck3", "ck6", "none"], fontsize=7, rotation=25)
            ax.set_yticks([0, 3, 6, 9])
            ax.set_yticklabels(["base", "ck3", "ck6", "none"], fontsize=7)

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.01, pad=0.01)
    cbar.set_label("Question count")
    fig.suptitle("Transition Timing Count Heatmaps Contact Sheet", fontsize=14, y=0.997)
    fig.supxlabel("t_ext_drop")
    fig.supylabel("t_int_drop")
    fig.subplots_adjust(left=0.05, right=0.92, bottom=0.05, top=0.96, wspace=0.10, hspace=0.10)
    fig.savefig(outdir / "transition_count_heatmap_contact.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / "transition_count_heatmap_contact.pdf", bbox_inches="tight")
    plt.close(fig)


def make_lag_boxplots(df: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharey=True)
    axes = axes.ravel()

    for i, method in enumerate(METHODS):
        ax = axes[i]
        m = df[df["method"] == method]
        if m.empty:
            ax.set_title(f"{method} (no data)", fontsize=9)
            ax.axis("off")
            continue

        data = []
        labels = []
        for s in SUBSETS:
            vals = m[m["subset"] == s]["suppression_lag"].to_numpy()
            if len(vals) > 0:
                data.append(vals)
                labels.append(s)
        if not data:
            ax.axis("off")
            continue

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, lbl in zip(bp["boxes"], labels):
            patch.set_facecolor(SUBSET_COLORS.get(lbl, "#cccccc"))
            patch.set_alpha(0.5)

        rng = np.random.default_rng(42)
        for idx, lbl in enumerate(labels, start=1):
            vals = m[m["subset"] == lbl]["suppression_lag"].to_numpy()
            if len(vals) == 0:
                continue
            x = idx + rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(x, vals, s=8, alpha=0.25, color=SUBSET_COLORS.get(lbl, "#333333"), edgecolors="none")

        ax.axhline(0, linestyle="--", color="black", linewidth=0.8, alpha=0.8)
        ax.set_title(method, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=SUBSET_COLORS[s], markersize=10, label=s, alpha=0.7)
        for s in SUBSETS
    ]
    fig.legend(legend_handles, SUBSETS, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.985), title="Final subset")
    fig.suptitle("Suppression Lag Distribution by Method and Subset", fontsize=14, y=0.997)
    fig.supxlabel("Final subset")
    fig.supylabel("suppression_lag = t_int_drop - t_ext_drop")
    fig.tight_layout(rect=[0.03, 0.05, 1, 0.94])
    fig.savefig(outdir / "suppression_lag_box_by_method.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / "suppression_lag_box_by_method.pdf", bbox_inches="tight")
    plt.close(fig)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))


def bootstrap_ci_mean_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42):
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = np.mean(xb) - np.mean(yb)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def compute_summary_tables(df: pd.DataFrame, outdir: Path):
    summary = (
        df.groupby(["method", "subset"], as_index=False)
        .agg(
            n_questions=("question_idx", "size"),
            mean_t_ext_drop=("t_ext_drop", "mean"),
            mean_t_int_drop=("t_int_drop", "mean"),
            mean_suppression_lag=("suppression_lag", "mean"),
            median_suppression_lag=("suppression_lag", "median"),
            int_censored_rate=("int_censored", "mean"),
            ext_censored_rate=("ext_censored", "mean"),
        )
    )

    # suppressed-forgotten method diff
    piv = summary.pivot_table(index="method", columns="subset", values="mean_suppression_lag")
    diff_rows = []
    for method in METHODS:
        if method not in piv.index:
            continue
        sup = piv.loc[method, "suppressed"] if "suppressed" in piv.columns else np.nan
        fog = piv.loc[method, "forgotten"] if "forgotten" in piv.columns else np.nan
        diff_rows.append({
            "method": method,
            "lag_diff_suppressed_minus_forgotten": sup - fog if np.isfinite(sup) and np.isfinite(fog) else np.nan,
        })
    diff_df = pd.DataFrame(diff_rows)

    out = summary.merge(diff_df, on="method", how="left")
    out.to_csv(outdir / "transition_timing_improved_summary.csv", index=False)

    md_lines = ["# Transition Timing Improved Summary", ""]
    md_lines.append("| method | subset | n_questions | mean_t_ext_drop | mean_t_int_drop | mean_suppression_lag | median_suppression_lag | int_censored_rate | ext_censored_rate | lag_diff_suppressed_minus_forgotten |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in out.sort_values(["method", "subset"]).iterrows():
        md_lines.append(
            f"| {r['method']} | {r['subset']} | {int(r['n_questions'])} | {r['mean_t_ext_drop']:.3f} | {r['mean_t_int_drop']:.3f} | "
            f"{r['mean_suppression_lag']:.3f} | {r['median_suppression_lag']:.3f} | {r['int_censored_rate']:.3f} | {r['ext_censored_rate']:.3f} | "
            f"{r['lag_diff_suppressed_minus_forgotten']:.3f} |"
        )
    (outdir / "transition_timing_improved_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    return out


def run_tests(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        m = df[df["method"] == method]
        sup = m[m["subset"] == "suppressed"]["suppression_lag"].to_numpy(dtype=float)
        fog = m[m["subset"] == "forgotten"]["suppression_lag"].to_numpy(dtype=float)
        if len(sup) == 0 or len(fog) == 0:
            rows.append({
                "method": method,
                "n_suppressed": len(sup),
                "n_forgotten": len(fog),
                "mean_lag_suppressed": np.nan,
                "mean_lag_forgotten": np.nan,
                "mean_diff": np.nan,
                "median_lag_suppressed": np.nan,
                "median_lag_forgotten": np.nan,
                "mannwhitney_p": np.nan,
                "effect_size": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
            })
            continue

        p = np.nan
        if mannwhitneyu is not None and len(sup) >= 2 and len(fog) >= 2:
            try:
                p = float(mannwhitneyu(sup, fog, alternative="greater").pvalue)
            except Exception:
                p = np.nan
        ci_low, ci_high = bootstrap_ci_mean_diff(sup, fog)
        rows.append({
            "method": method,
            "n_suppressed": len(sup),
            "n_forgotten": len(fog),
            "mean_lag_suppressed": float(np.mean(sup)),
            "mean_lag_forgotten": float(np.mean(fog)),
            "mean_diff": float(np.mean(sup) - np.mean(fog)),
            "median_lag_suppressed": float(np.median(sup)),
            "median_lag_forgotten": float(np.median(fog)),
            "mannwhitney_p": p,
            "effect_size": cliffs_delta(sup, fog),
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    tdf = pd.DataFrame(rows)
    tdf.to_csv(outdir / "suppressed_vs_forgotten_lag_tests.csv", index=False)
    return tdf


def write_markdown_summary(summary_df: pd.DataFrame, tests_df: pd.DataFrame, outdir: Path):
    lines = [
        "# Transition Timing Improved Plot Summary",
        "",
        "The original scatter was hard to read because many questions shared identical discrete coordinates, causing heavy overlap.",
        "The bubble scatter aggregates identical coordinates and uses bubble size for counts.",
        "The count heatmaps show exact per-cell counts per method/subset, making diagonal/above-diagonal structure visible.",
        "",
        "Interpretation guide: above the diagonal means internal drop happened later than external drop; `y=9` means internal did not drop by ck8.",
        "",
    ]

    piv = summary_df.pivot_table(index="method", columns="subset", values="mean_suppression_lag")
    effects = []
    for method in METHODS:
        if method not in piv.index:
            continue
        sup = piv.loc[method, "suppressed"] if "suppressed" in piv.columns else np.nan
        fog = piv.loc[method, "forgotten"] if "forgotten" in piv.columns else np.nan
        if np.isfinite(sup) and np.isfinite(fog):
            effects.append((method, float(sup - fog)))

    if effects:
        effects_sorted = sorted(effects, key=lambda x: x[1], reverse=True)
        lines.append("Strongest timing-lag effects (suppressed minus forgotten mean lag):")
        for m, d in effects_sorted[:3]:
            lines.append(f"- {m}: {d:.3f}")
        lines.append("")
        lines.append("Weak/ambiguous timing-lag effects:")
        for m, d in sorted(effects, key=lambda x: abs(x[1]))[:3]:
            lines.append(f"- {m}: {d:.3f}")

    if not tests_df.empty:
        lines.append("")
        lines.append("Statistical test note: Mann-Whitney U uses alternative='greater' (suppressed lag > forgotten lag).")

    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    input_path = Path("plots/transition_timing/transition_timing_by_question.csv")
    outdir = Path("plots/transition_timing_improved")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_transition_table(input_path)
    df = normalize_columns(df)

    # keep known methods/subsets if present; do not fail on missing
    df = df[df["method"].isin(METHODS) & df["subset"].isin(SUBSETS)].copy()
    if df.empty:
        raise RuntimeError("No rows remain after filtering to expected methods/subsets.")

    make_bubble_scatter(df, outdir)
    make_count_heatmaps(df, outdir)
    make_lag_boxplots(df, outdir)
    summary_df = compute_summary_tables(df, outdir)
    tests_df = run_tests(df, outdir)
    write_markdown_summary(summary_df, tests_df, outdir)

    print("Saved outputs to", outdir)


if __name__ == "__main__":
    main()
