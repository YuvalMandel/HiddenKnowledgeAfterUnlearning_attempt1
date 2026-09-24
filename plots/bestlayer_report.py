#!/usr/bin/env python3
"""Merge the per-method bestlayer_skip runs and put them beside the layer-32 edit.

bestlayer_skip.py writes one CSV set per method (it is run one method per
process so the login node can hold three at a time). This merges them, joins
readout_edit_cv.csv, and answers the question the run was for: does sourcing the
True/False evidence from the best layer reach knowledge the layer-32 edit
cannot?

Usage: python plots/bestlayer_report.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # no display on the server
import matplotlib.pyplot as plt             # noqa: E402
import pandas as pd                         # noqa: E402

AV = Path(__file__).resolve().parent / "activation_vectors"
ROOT_IMGS = Path(__file__).resolve().parent.parent / "overleaf_claims" / "imgs"
ORDER = ["base", "GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR",
         "PB_J"]


def merge(stem):
    # bestlayer_skip_*.csv also matches bestlayer_skip_perq_*.csv, so the
    # method suffix has to be checked against the method list, not globbed.
    files = [AV / f"{stem}_{m}.csv" for m in ORDER]
    parts = [pd.read_csv(p) for p in files if p.exists()]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["_o"] = df.method.map({m: i for i, m in enumerate(ORDER)})
    df = df.sort_values(["_o"] + (["layer"] if "layer" in df else [])).drop(
        columns="_o")
    df.to_csv(AV / f"{stem}.csv", index=False)
    return df


skip = merge("bestlayer_skip")
lens = merge("bestlayer_lens")
resid = merge("bestlayer_resid")
merge("bestlayer_skip_perq")
if skip.empty:
    raise SystemExit("no bestlayer_skip_*.csv found -- has the Newton run landed?")

real = skip[skip.labels == "real"].set_index("method")
shuf = skip[skip.labels == "shuffled"].set_index("method")
edit = pd.read_csv(AV / "readout_edit_cv.csv")
edit = edit[edit.labels == "real"].set_index("method")
# lambda=1e6 with the retention constraint dropped: the probe REPLACING the
# readout. Written by bestlayer_lambda_sweep.py from the cached projections.
lam = (pd.read_csv(AV / "bestlayer_lambda_points.csv").set_index("method")
       if (AV / "bestlayer_lambda_points.csv").exists() else None)

# K_int at 32 and at the best layer, both under this script's honest CV.
best = lens.loc[lens.groupby("method").k_int_cv.idxmax()].set_index("method")
at32 = lens[lens.layer == 32].set_index("method")

print("K_ext, then what each readout recovers. * = 95% bootstrap CI excludes 0.")
print("edit@32  = readout_edit_cv.csv (rank-1 unembedding edit, layer 32)")
print("skip     = same protocol, evidence from layer L*, retention-constrained")
print("free     = same, retention constraint dropped")
print("po       = folds where free picked lambda=1e6, i.e. the readout was")
print("           REPLACED by the probe. Those folds restate K_int@L* and are")
print("           not independent evidence (KNOWN_ISSUES: circular metrics).")
print("lens@L*  = the model's OWN direction v applied at L*, zero parameters.\n")
hdr = (f"{'method':<9}{'K_ext':>7}{'edit@32':>9}{'skip':>9}{'free':>9}"
       f"{'shuf':>8}{'po':>4}{'L*':>5}{'Kint@32':>9}{'Kint@L*':>9}{'lens@L*':>9}")
print(hdr)
print("-" * len(hdr))
rows = []
for m in ORDER:
    if m not in real.index:
        continue
    r, b = real.loc[m], best.loc[m]
    e = float(edit.loc[m, "gain"]) if m in edit.index else float("nan")
    lbest = float(lens[(lens.method == m) &
                       (lens.layer == b.layer)].k_lens.iloc[0])
    po = str(r.lambdas_free).split("/").count("1e+06")
    print(f"{m:<9}{r.k_ext:>7.3f}{e:>+9.4f}"
          f"{r.gain:>+8.4f}{'*' if r.sig else ' '}"
          f"{r.gain_free:>+8.4f}{'*' if r.sig_free else ' '}"
          f"{shuf.loc[m, 'gain_free']:>+8.4f}{po:>4}"
          f"{int(b.layer):>5}{at32.loc[m, 'k_int_cv']:>9.3f}"
          f"{b.k_int_cv:>9.3f}{lbest:>9.3f}")
    rows.append(dict(method=m, k_ext=r.k_ext, edit32=e, skip=r.gain,
                     skip_sig=r.sig, skip_lo=r.lo, skip_hi=r.hi,
                     free=r.gain_free, free_sig=r.sig_free,
                     free_lo=r.lo_free, free_hi=r.hi_free,
                     shuffled=shuf.loc[m, "gain_free"], probe_only_folds=po,
                     best_layer=int(b.layer),
                     kint32=at32.loc[m, "k_int_cv"], kint_best=b.k_int_cv,
                     lens_best=lbest,
                     lam1e6=(float(lam.loc[m, "k_lam1e6_free"])
                             if lam is not None and m in lam.index else None),
                     resid=(float(resid[resid.method == m].gain.iloc[0])
                            if not resid.empty and (resid.method == m).any()
                            else None)))
out = pd.DataFrame(rows)
out.to_csv(AV / "bestlayer_summary.csv", index=False)
print(f"\nwrote {AV/'bestlayer_summary.csv'}")

# ---- figure 1: what each readout reaches, per method.
fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(out) + 2.4))
y = range(len(out))
for i, r in out.iterrows():
    ax.plot([r.k_ext, r.kint_best], [i, i], color="0.88", lw=7, zorder=1,
            solid_capstyle="butt")
    ax.plot([r.k_ext + r.skip_lo, r.k_ext + r.skip_hi], [i, i],
            color="#2b7bba", lw=1.8, zorder=3)
ax.scatter(out.k_ext, y, s=46, marker="o", color="#333", zorder=4,
           label="K_ext, unedited")
ax.scatter(out.k_ext + out.edit32, y, s=46, marker="s", color="#e07b39",
           zorder=4, label="rank-1 edit at layer 32 (previous result)")
ax.scatter(out.k_ext + out.skip, y, s=62, marker="D", color="#2b7bba",
           zorder=5, label="skip from layer L* (this run)")
if out.lam1e6.notna().all():
    ax.scatter(out.lam1e6, y, s=52, marker=">", color="#4d9221", zorder=4,
               label="lambda = 1e6: probe replaces the readout")
ax.scatter(out.kint_best, y, s=85, marker="*", color="#c0392b", zorder=4,
           label="K_int at L*: the ceiling any lambda could reach")
ax.scatter(out.lens_best, y, s=40, marker="x", color="#7a5195", zorder=4,
           label="v applied at L* (logit lens, 0 parameters)")
for i, r in out.iterrows():
    ax.annotate(f"L*={r.best_layer}", (r.kint_best, i),
                textcoords="offset points", xytext=(9, -3), fontsize=7.5,
                color="#c0392b")
ax.axvline(0.5, color="k", lw=0.8, ls=":")
ax.text(0.5, -0.62, "chance", fontsize=7.5, ha="center", color="0.35")
ax.set_yticks(list(y))
ax.set_yticklabels(list(out.method))
ax.set_xlabel("K (fraction of the 3 distractors the correct claim outscores)")
ax.set_title("Sourcing the True/False evidence from layer L* instead of 32",
             fontsize=11)
ax.set_xlim(min(out.lens_best.min(), out.k_ext.min()) - 0.012,
            out.kint_best.max() + 0.065)
ax.set_ylim(len(out) - 0.4, -0.95)
ax.legend(fontsize=7.5, loc="upper center", ncol=3,
          bbox_to_anchor=(0.5, -0.12), frameon=False)
ax.grid(axis="x", color="0.93")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(AV / "bestlayer_skip.png", dpi=180)
print(f"wrote {AV/'bestlayer_skip.png'}")
# vector copy straight into the Overleaf checkout, so the paper figure and the
# CSV it came from can never drift apart
imgs = ROOT_IMGS
if imgs.is_dir():
    fig.savefig(imgs / "bestlayer_skip.pdf", bbox_inches="tight")
    print(f"wrote {imgs/'bestlayer_skip.pdf'}")

# ---- figure 2: the point of the whole run. A probe finds the answer in the
# middle layers of every model; the model's OWN readout direction finds it
# there only in the base model.
titles = ["A.  the model's own axis $v$ at layer $\ell$\n(logit lens, no fitted parameters)",
          "B.  a probe fitted at layer $\ell$\n($K_{int}$, cross-validated)"]
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45), sharey=True)
for ax, col, ttl in zip(axes, ["k_lens", "k_int_cv"], titles):
    for m in ORDER:
        d = lens[lens.method == m]
        if d.empty:
            continue
        style = (dict(lw=1.5, color="k", ls="--", zorder=5) if m == "base"
                 else dict(lw=0.9))
        ax.plot(d.layer, d[col], marker="o", ms=1.8,
                label=m.replace("_", "&"), **style)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xlabel("layer $\\ell$", fontsize=7)
    ax.set_title(ttl, fontsize=7)
    ax.grid(color="0.93", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("$K$ (win fraction over the 3 distractors)", fontsize=7)
axes[0].text(33, 0.494, "chance", fontsize=6, va="top", ha="right")
axes[1].legend(fontsize=5.8, ncol=5, loc="lower right", frameon=False,
               handlelength=1.0, handletextpad=0.3, labelspacing=0.25,
               columnspacing=0.6, borderaxespad=0.2)
fig.tight_layout(pad=0.4)
fig.savefig(AV / "bestlayer_lens_by_layer.png", dpi=300)
print(f"wrote {AV/'bestlayer_lens_by_layer.png'}")
if ROOT_IMGS.is_dir():
    fig.savefig(ROOT_IMGS / "logit_lens_by_layer.pdf", bbox_inches="tight")
    print(f"wrote {ROOT_IMGS/'logit_lens_by_layer.pdf'}")
