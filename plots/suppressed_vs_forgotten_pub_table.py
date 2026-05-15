#!/usr/bin/env python3
"""
Task C: Publication-ready suppressed-vs-forgotten trajectory table.

Adds to the existing stats:
  - Benjamini-Hochberg FDR q-values
  - Bootstrap 95% CI for the mean difference
  - Cohen's d
  - Cliff's delta

Outputs (in plots/ckpt_layer_trajectory_metrics/):
  suppressed_vs_forgotten_publication_table.csv
  suppressed_vs_forgotten_publication_table.md
  suppressed_vs_forgotten_effect_sizes.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "plots"))

TRAJ_BY_Q = REPO / "plots" / "ckpt_layer_trajectory_metrics" / "trajectory_metrics_by_method_subset.csv"
OUT_DIR = REPO / "plots" / "ckpt_layer_trajectory_metrics"

METHOD_ORDER = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
N_BOOT = 5000
SEED = 42


def bootstrap_ci(a, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    diffs = [
        float(rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean())
        for _ in range(n_boot)
    ]
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def cliffs_delta(a, b):
    greater = sum(ai > bi for ai in a for bi in b)
    less = sum(ai < bi for ai in a for bi in b)
    return float((greater - less) / (len(a) * len(b)))


def bh_correction(p_values):
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = np.array(p_values) * n / ranks
    q_adj = q.copy()
    for i in range(n - 2, -1, -1):
        q_adj[order[i]] = min(q_adj[order[i]], q_adj[order[i + 1]])
    return np.minimum(q_adj, 1.0)


def main():
    df = pd.read_csv(TRAJ_BY_Q)
    full = df[(df["layer_band"] == "full") & (df["subset"].isin(["suppressed", "forgotten"]))].copy()

    rows = []
    for method in METHOD_ORDER:
        m = full[full["method"] == method]
        s = m[m["subset"] == "suppressed"]["auc"].dropna().to_numpy()
        f = m[m["subset"] == "forgotten"]["auc"].dropna().to_numpy()
        if len(s) < 2 or len(f) < 2:
            continue

        u, p = mannwhitneyu(s, f, alternative="greater")
        boot_mean, ci_lo, ci_hi = bootstrap_ci(s, f)
        d = cohens_d(s, f)
        delta = cliffs_delta(s, f)

        rows.append({
            "method": method,
            "n_suppressed": len(s),
            "n_forgotten": len(f),
            "mean_suppressed": round(s.mean(), 4),
            "mean_forgotten": round(f.mean(), 4),
            "diff": round(s.mean() - f.mean(), 4),
            "ci95_lo": round(ci_lo, 4),
            "ci95_hi": round(ci_hi, 4),
            "p_value": p,
            "cohens_d": round(d, 3),
            "cliffs_delta": round(delta, 3),
        })

    pub = pd.DataFrame(rows)
    pub["q_value"] = bh_correction(pub["p_value"].to_numpy())
    pub["sig"] = pub["q_value"].apply(
        lambda q: "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else "ns"))
    )

    pub["p_value"] = pub["p_value"].apply(lambda x: float(f"{x:.3e}"))
    pub["q_value"] = pub["q_value"].apply(lambda x: float(f"{x:.3e}"))

    col_order = [
        "method", "n_suppressed", "n_forgotten",
        "mean_suppressed", "mean_forgotten", "diff",
        "ci95_lo", "ci95_hi", "p_value", "q_value", "sig",
        "cohens_d", "cliffs_delta",
    ]
    pub = pub[col_order]

    pub.to_csv(OUT_DIR / "suppressed_vs_forgotten_publication_table.csv", index=False)

    pub[["method", "n_suppressed", "n_forgotten", "cohens_d", "cliffs_delta",
         "diff", "ci95_lo", "ci95_hi"]].to_csv(
        OUT_DIR / "suppressed_vs_forgotten_effect_sizes.csv", index=False
    )

    lines = [
        "# Suppressed vs Forgotten — Publication Table",
        "",
        "Full-layer (all-33-layer) internal K_int trajectory (mean over ck1-ck8). "
        "Bootstrap 95% CI (n=5000). "
        "p-values: one-sided Mann-Whitney U (suppressed > forgotten). "
        "q-values: Benjamini-Hochberg FDR. "
        "* q<0.05  ** q<0.01  *** q<0.001",
        "",
        "| Method | n supp | n forg | Mean supp | Mean forg | Diff | 95% CI | p | q | sig | Cohen d | Cliff d |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|",
    ]
    for _, r in pub.iterrows():
        ci = f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]"
        lines.append(
            f"| {r['method']} | {r['n_suppressed']} | {r['n_forgotten']} "
            f"| {r['mean_suppressed']:.3f} | {r['mean_forgotten']:.3f} "
            f"| {r['diff']:.3f} | {ci} "
            f"| {r['p_value']:.2e} | {r['q_value']:.2e} | {r['sig']} "
            f"| {r['cohens_d']:.3f} | {r['cliffs_delta']:.3f} |"
        )

    md_text = "\n".join(lines)
    (OUT_DIR / "suppressed_vs_forgotten_publication_table.md").write_text(md_text, encoding="utf-8")

    print(md_text)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
