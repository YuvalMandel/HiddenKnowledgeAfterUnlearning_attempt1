# Generate FEATURE-MODE-C figures alongside the existing mode-A ones, from the
# already-computed artifacts (no recompute). Mode C = correct - mean(wrong), the
# contrastive feature that predicts suppressibility best. Files get a _modeC suffix;
# the original (mode A) figures are left untouched. Also emits one A-vs-C comparison.
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ODIR=Path("activation_question_scores"); FIG=ODIR/"figures"
PRIMARY="suppressed_vs_forgotten"; MODEC="correct_minus_mean_wrong"; MODEA="correct_only"
CONTRASTS=["suppressed_vs_forgotten","suppressed_vs_retained","suppressed_vs_rest","forgotten_vs_retained"]
MNAMES=["GradDiff","PB&J","RMU","RMU-LAT","RepNoise","ELM","RR","TAR"]

# ---- per-question OOF (mode C, mean-diff, Q*): mean projection_zscore per question
cols=["question_id","method","contrast","feature_mode","direction","question_set",
      "binary_label","true_subset_ck8","projection_zscore"]
pq=pd.read_csv(ODIR/"per_question_activation_direction_scores.csv.gz", usecols=cols)
pq=pq[(pq.direction=="mean_difference")&(pq.question_set=="Qstar")&(pq.feature_mode==MODEC)]
prim=pq[pq.contrast==PRIMARY]
# per-question mean over repeats
perq=(prim.groupby(["method","question_id"])
          .agg(z=("projection_zscore","mean"),
               subset=("true_subset_ck8","first"),
               pos=("binary_label","first")).reset_index())
modeC={m:g for m,g in perq.groupby("method")}
mnames=[m for m in MNAMES if m in modeC]

# ---- summaries (have all modes)
s=pd.read_csv(ODIR/"activation_direction_auc_summary.csv")
selC=s[(s.layer_selection_mode=="per_fold_selected")&(s.question_set=="Qstar")&
       (s.direction=="mean_difference")&(s.feature_mode==MODEC)]
selA=s[(s.layer_selection_mode=="per_fold_selected")&(s.question_set=="Qstar")&
       (s.direction=="mean_difference")&(s.feature_mode==MODEA)]

# ====== 1. histograms suppressed vs forgotten (mode C) ======
fig,axes=plt.subplots(2,4,figsize=(18,8))
for ax,m in zip(axes.ravel(),mnames):
    g=modeC[m]
    ax.hist(g[g.subset=="suppressed"].z,bins=20,alpha=.6,label="suppressed",color="#d62728",density=True)
    ax.hist(g[g.subset=="forgotten"].z,bins=20,alpha=.6,label="forgotten",color="#1f77b4",density=True)
    ax.set_title(m); ax.set_xlabel("projection z")
axes.ravel()[0].legend()
fig.suptitle("FEATURE MODE C  [correct - mean(wrong)]  -- projection scores: suppressed vs forgotten (mean-diff, OOF)")
fig.tight_layout(); fig.savefig(FIG/"suppressed_vs_forgotten_projection_histograms_by_method_modeC.png",dpi=130); plt.close(fig)

# ====== 2. P(suppressed | quintile) (mode C) ======
fig,ax=plt.subplots(figsize=(11,6))
qrows=[]
for m in mnames:
    g=modeC[m].copy()
    try: g["q"]=pd.qcut(g.z,5,labels=False,duplicates="drop")+1
    except Exception: g["q"]=1
    gg=g.groupby("q").pos.mean()
    ax.plot(gg.index,gg.values,marker="o",label=m)
    for qv,r in gg.items(): qrows.append(dict(method=m,quintile=int(qv),suppressed_rate=float(r)))
ax.set_xlabel("projection-score quintile (low->high)"); ax.set_ylabel("P(suppressed | quintile)")
ax.set_title("FEATURE MODE C  [correct - mean(wrong)]  -- suppression rate across projection quintiles")
ax.axhline(0.5,ls="--",c="gray",lw=.8); ax.legend(ncol=2,fontsize=8); fig.tight_layout()
fig.savefig(FIG/"suppressed_vs_forgotten_projection_quintile_rates_modeC.png",dpi=130); plt.close(fig)
pd.DataFrame(qrows).to_csv(ODIR/"quantile_summary_modeC_suppressed_vs_forgotten.csv",index=False)

# ====== 3. AUC by method & contrast (mode C) ======
fig,ax=plt.subplots(figsize=(13,6)); w=.2; x=np.arange(len(mnames))
for i,c in enumerate(CONTRASTS):
    vals=[]; errs=[]
    for m in mnames:
        r=selC[(selC.method==m)&(selC.contrast==c)]
        if len(r): vals.append(r.auc_mean.values[0]); errs.append(max(0.,r.auc_mean.values[0]-r.auc_ci95_low.values[0]))
        else: vals.append(np.nan); errs.append(0)
    ax.bar(x+i*w,vals,w,yerr=errs,capsize=2,label=c)
ax.set_xticks(x+1.5*w); ax.set_xticklabels(mnames,rotation=30,ha="right")
ax.axhline(0.5,ls="--",c="gray"); ax.set_ylabel("OOF AUC (mean-diff)"); ax.set_ylim(0.3,0.8)
ax.set_title("FEATURE MODE C  [correct - mean(wrong)]  -- base-geometry predictiveness by method & contrast (Q*)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG/"activation_direction_auc_by_method_modeC.png",dpi=130); plt.close(fig)

# ====== 4. boxplots by subset (mode C) ======
fig,axes=plt.subplots(2,4,figsize=(18,8))
for ax,m in zip(axes.ravel(),mnames):
    g=modeC[m]
    ax.boxplot([g[g.subset=="suppressed"].z.values,g[g.subset=="forgotten"].z.values],tick_labels=["supp","forg"])
    ax.set_title(m); ax.axhline(0,ls="--",c="gray",lw=.6)
fig.suptitle("FEATURE MODE C  [correct - mean(wrong)]  -- projection-z by eventual subset (OOF)")
fig.tight_layout(); fig.savefig(FIG/"per_method_projection_score_boxplots_modeC.png",dpi=130); plt.close(fig)

# ====== 5. BONUS: mode A vs mode C AUC on the headline contrast ======
fig,ax=plt.subplots(figsize=(12,6)); x=np.arange(len(mnames)); w=.38
def col(sel,m):
    r=sel[(sel.method==m)&(sel.contrast==PRIMARY)]; return r.auc_mean.values[0] if len(r) else np.nan
ax.bar(x-w/2,[col(selA,m) for m in mnames],w,label="mode A (correct only)",color="#9ecae1")
ax.bar(x+w/2,[col(selC,m) for m in mnames],w,label="mode C (correct - mean wrong)",color="#fb6a4a")
ax.set_xticks(x); ax.set_xticklabels(mnames,rotation=30,ha="right")
ax.axhline(0.5,ls="--",c="gray"); ax.axhline(0.70,ls=":",c="green",lw=.8)
ax.set_ylabel("OOF AUC (suppressed_vs_forgotten)"); ax.set_ylim(0.3,0.8)
ax.set_title("Mode A vs Mode C on the headline contrast (green dotted = 0.70 'strong')")
ax.legend(); fig.tight_layout(); fig.savefig(FIG/"modeA_vs_modeC_auc_suppressed_vs_forgotten.png",dpi=130); plt.close(fig)
print("wrote mode-C figures:",sorted(p.name for p in FIG.glob("*modeC*")) + ["modeA_vs_modeC_auc_suppressed_vs_forgotten.png"])
