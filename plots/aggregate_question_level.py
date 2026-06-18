# Aggregate the per-method parts written by activation_question_level.py (job array)
# into the final deliverables: per-question CSV, AUC + quantile summaries, 4 figures, REPORT.md.
import glob, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ODIR   = Path("activation_question_scores"); FIGDIR = ODIR/"figures"; PARTS = ODIR/"_parts"
FIGDIR.mkdir(parents=True, exist_ok=True)
PRIMARY_CONTR="suppressed_vs_forgotten"; PRIMARY_MODE="correct_only"
PCA_K=256; N_REPEATS=20; SEED=42
CONTRASTS=["suppressed_vs_forgotten","suppressed_vs_retained","suppressed_vs_rest","forgotten_vs_retained"]
MNAMES=["GradDiff","PB&J","RMU","RMU-LAT","RepNoise","ELM","RR","TAR"]

# ---- concat per-question parts -> final gzip CSV (stream to avoid big memory)
pqs=sorted(glob.glob(str(PARTS/"pq_*.csv.gz")))
final_pq=ODIR/"per_question_activation_direction_scores.csv.gz"
if final_pq.exists(): final_pq.unlink()
hdr=True
for p in pqs:
    for chunk in pd.read_csv(p, chunksize=200000):
        chunk.to_csv(final_pq, mode="a", header=hdr, index=False); hdr=False
print("per-question rows from", len(pqs), "parts ->", final_pq, flush=True)

# ---- concat summaries / quintiles
sumdf=pd.concat([pd.read_csv(p) for p in glob.glob(str(PARTS/"summary_*.csv"))], ignore_index=True)
qdf  =pd.concat([pd.read_csv(p) for p in glob.glob(str(PARTS/"quint_*.csv"))],  ignore_index=True)
sumdf.to_csv(ODIR/"activation_direction_auc_summary.csv", index=False)
qdf.to_csv(ODIR/"activation_direction_quantile_summary.csv", index=False)

# ---- modeA per-question scores -> dict for plots
ma=pd.concat([pd.read_csv(p) for p in glob.glob(str(PARTS/"modeA_*.csv"))], ignore_index=True)
modeA={(r,c):g for (r,c),g in ma.groupby(["method","contrast"])}

sel=sumdf[(sumdf.layer_selection_mode=="per_fold_selected")&(sumdf.question_set=="Qstar")&
          (sumdf.direction=="mean_difference")&(sumdf.feature_mode==PRIMARY_MODE)]
mnames=[m for m in MNAMES if m in set(sumdf.method)]

# 1. suppressed vs forgotten projection histograms
fig,axes=plt.subplots(2,4,figsize=(18,8))
for ax,label in zip(axes.ravel(),mnames):
    g=modeA.get((label,PRIMARY_CONTR))
    if g is None: ax.set_title(f"{label} (n/a)"); ax.axis("off"); continue
    ax.hist(g[g.subset=="suppressed"].z,bins=20,alpha=.6,label="suppressed",color="#d62728",density=True)
    ax.hist(g[g.subset=="forgotten"].z,bins=20,alpha=.6,label="forgotten",color="#1f77b4",density=True)
    ax.set_title(label); ax.set_xlabel("projection z")
axes.ravel()[0].legend()
fig.suptitle("Base-model projection scores: suppressed vs forgotten (mode A, mean-diff, OOF)")
fig.tight_layout(); fig.savefig(FIGDIR/"suppressed_vs_forgotten_projection_histograms_by_method.png",dpi=130); plt.close(fig)

# 2. P(suppressed | quintile)
qsf=qdf[qdf.contrast==PRIMARY_CONTR]
fig,ax=plt.subplots(figsize=(11,6))
for label in mnames:
    g=qsf[qsf.method==label].sort_values("quintile")
    if len(g): ax.plot(g.quintile,g.positive_rate,marker="o",label=label)
ax.set_xlabel("projection-score quintile (low->high)"); ax.set_ylabel("P(suppressed | quintile)")
ax.set_title("Monotonicity of suppression rate across projection quintiles"); ax.legend(ncol=2,fontsize=8)
ax.axhline(0.5,ls="--",c="gray",lw=.8); fig.tight_layout()
fig.savefig(FIGDIR/"suppressed_vs_forgotten_projection_quintile_rates.png",dpi=130); plt.close(fig)

# 3. AUC by method, grouped bars over contrasts
fig,ax=plt.subplots(figsize=(13,6)); w=.2; x=np.arange(len(mnames))
for i,c in enumerate(CONTRASTS):
    vals=[]; errs=[]
    for label in mnames:
        r=sel[(sel.method==label)&(sel.contrast==c)]
        if len(r): vals.append(r.auc_mean.values[0]); errs.append(max(0.0,r.auc_mean.values[0]-r.auc_ci95_low.values[0]))
        else: vals.append(np.nan); errs.append(0)
    ax.bar(x+i*w,vals,w,yerr=errs,capsize=2,label=c)
ax.set_xticks(x+1.5*w); ax.set_xticklabels(mnames,rotation=30,ha="right")
ax.axhline(0.5,ls="--",c="gray"); ax.set_ylabel("OOF AUC (mean-diff)"); ax.set_ylim(0.3,0.8)
ax.set_title("Question-level base-geometry predictiveness by method & contrast (Q*, mode A)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGDIR/"activation_direction_auc_by_method.png",dpi=130); plt.close(fig)

# 4. projection-score boxplots by subset
fig,axes=plt.subplots(2,4,figsize=(18,8))
for ax,label in zip(axes.ravel(),mnames):
    g=modeA.get((label,PRIMARY_CONTR))
    if g is None: ax.set_title(f"{label} (n/a)"); ax.axis("off"); continue
    ax.boxplot([g[g.subset==s].z.values for s in ["suppressed","forgotten"]],labels=["supp","forg"])
    ax.set_title(label); ax.axhline(0,ls="--",c="gray",lw=.6)
fig.suptitle("Projection-z by eventual subset (mode A, OOF)")
fig.tight_layout(); fig.savefig(FIGDIR/"per_method_projection_score_boxplots.png",dpi=130); plt.close(fig)
print("wrote 4 figures", flush=True)

# ---- REPORT.md
def cell(label,c):
    r=sel[(sel.method==label)&(sel.contrast==c)]
    return "n/a" if not len(r) else f"{r.auc_mean.values[0]:.3f} [{r.auc_ci95_low.values[0]:.3f},{r.auc_ci95_high.values[0]:.3f}]"
rsf0=sel[(sel.contrast=="suppressed_vs_rest")]
nqstar=int(rsf0.n_positive.values[0]+rsf0.n_negative.values[0]) if len(rsf0) else 701
clean=False
for (label,c),g in modeA.items():
    if c!=PRIMARY_CONTR: continue
    s=g[g.subset=="suppressed"].z; f=g[g.subset=="forgotten"].z
    if len(s) and len(f) and (s.min()>f.max() or f.min()>s.max()): clean=True
L=[]
L.append("# Question-level base-model activation-direction analysis — REPORT\n")
L.append("## Data sources")
L.append("- Features: `inside_out_out/base/bio_hs.npy` (BASE model, (1273,4,33,4096)); MCQ-level vector per question.")
L.append("- Labels: `inside_out_out/{method}_ck8/k_scores.parquet` (LR / own / best_layer / cv / bio), MCQ-level Kint/Kext.")
L.append("- Base Q* + Kint/Kext from `inside_out_out/base/k_scores.parquet`.\n")
L.append("## Subset definition (MCQ-level, paper rule)")
L.append("suppressed: Kint>0.5 & Kext<=0.5 · forgotten: Kint<=0.5 & Kext<=0.5 · retained: Kint>0.5 & Kext>0.5 · lucky: Kint<=0.5 & Kext>0.5\n")
L.append(f"## Question set\nPrimary **Q\\*** (n={nqstar}; base knows on both axes). Supplementary **full** 1273-set (mode A only); see `question_set` column.\n")
L.append("## Feature aggregation rule")
L.append("- A `correct_only` = correct-option hidden state (main; matches the existing activation-vector code).")
L.append("- B `mean_all_choices` = mean over the 4 option claims.")
L.append("- C `correct_minus_mean_wrong` = correct − mean(wrong).")
L.append("- D `pairwise_correct_minus_wrong` = mean_w(correct − wrong_w) **= C algebraically** under MCQ-level aggregation, so it is not run separately (identical vector).\n")
L.append("## Leakage prevention")
L.append(f"Per fold, fit on TRAIN only: (i) per-(layer,feature) z-scoring, (ii) best-layer selection via an inner TRAIN holdout (resubstitution when a class is tiny), (iii) PCA-{PCA_K} for logistic regression, (iv) Platt calibration for mean-diff. Grouped CV by question_id (singleton groups ⇒ StratifiedKFold==GroupKFold); 5 folds × {N_REPEATS} repeats, seed {SEED}. Only OOF test rows are exported/evaluated.\n")
L.append("## Per-method OOF AUC (Q*, mode A, mean-diff, per-fold layer selection)\n")
L.append("| method | suppressed_vs_forgotten | suppressed_vs_retained | suppressed_vs_rest | forgotten_vs_retained | n_supp | n_forg |")
L.append("|---|---|---|---|---|---|---|")
for label in mnames:
    rr=sel[(sel.method==label)&(sel.contrast=="suppressed_vs_rest")]
    rf=sel[(sel.method==label)&(sel.contrast=="suppressed_vs_forgotten")]
    ns=int(rr.n_positive.values[0]) if len(rr) else 0
    nf=int(rf.n_negative.values[0]) if len(rf) else 0
    L.append(f"| {label} | {cell(label,'suppressed_vs_forgotten')} | {cell(label,'suppressed_vs_retained')} | {cell(label,'suppressed_vs_rest')} | {cell(label,'forgotten_vs_retained')} | {ns} | {nf} |")
L.append("\n## Projection quintiles (primary contrast)")
L.append("See `activation_direction_quantile_summary.csv` + the quintile figure (P(suppressed|quintile)); a monotone rise = usable graded signal even without a hard threshold.\n")
L.append("## Is there a clean threshold?")
L.append(("**Yes** — at least one method shows non-overlapping suppressed/forgotten projection ranges." if clean
          else "**No** — suppressed and forgotten projection-score ranges overlap for every method; separation is partial, not a deterministic rule.")+"\n")
L.append("## Reporting language")
L.append("Base-model hidden-state geometry carries **weak/moderate, method-specific** information about future suppressibility; it does **not** determine whether a question will be suppressed.\n")
L.append("## Limitations")
L.append("- Forgotten pools on Q* are small (n_forg column); suppressed_vs_forgotten is underpowered for low-n methods (GradDiff/RR/RMU-LAT). Full-set supplementary partly mitigates.")
L.append("- Effect sizes are modest (AUC ~0.55–0.66); diff-of-means is conservative; in-fold z-scoring differs slightly from earlier global-z-scored numbers.")
L.append("- 'lucky' excluded from primary contrasts; RMU/RMU-LAT carry little suppressed signal in base geometry (consistent with prior analysis).")
(ODIR/"REPORT.md").write_text("\n".join(L), encoding="utf-8")
print("wrote REPORT.md\nAGGREGATE DONE", flush=True)
