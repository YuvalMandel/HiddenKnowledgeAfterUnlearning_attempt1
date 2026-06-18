# Question-level activation-direction scores: predict eventual ck8 subset
# (suppressed / forgotten / retained / lucky) from BASE-model hidden states.
#
# Implements the partner spec "base model question level activaton task.md":
#   - unit = original MCQ question_id (one MCQ-level vector per question)
#   - 4 feature modes (A correct_only / B mean_all_choices / C correct_minus_mean_wrong
#       / D pairwise_correct_minus_wrong  -- NOTE: under MCQ-level aggregation D == C
#       algebraically [mean_w(correct-wrong_w) = correct - mean_w(wrong)], so we run A,B,C
#       and label C as both correct_minus_mean_wrong and pairwise (identical))
#   - 4 contrasts: suppressed_vs_forgotten (primary), suppressed_vs_retained,
#       suppressed_vs_rest, forgotten_vs_retained
#   - 2 directions: Option 1 mean-difference (+ in-fold Platt prob), Option 2 logistic
#       regression on in-fold PCA
#   - grouped CV by question_id (singleton groups => StratifiedKFold == GroupKFold here),
#       5 folds x 20 repeats, seed 42
#   - leakage-safe: z-scoring, PCA, and best-layer selection all fit on TRAIN fold only
#   - Q* primary + full-set supplementary (question_set column)
#
# Outputs under activation_question_scores/:
#   per_question_activation_direction_scores.csv.gz   (OOF test rows; gzipped for size)
#   activation_direction_auc_summary.csv
#   activation_direction_quantile_summary.csv
#   figures/*.png
#   REPORT.md
#
# Runs on Newton (needs inside_out_out/base/bio_hs.npy + inside_out_out/{m}_ck8/k_scores.parquet).
import os, sys, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             average_precision_score, log_loss, brier_score_loss)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO   = Path(".")
OUT    = REPO / "inside_out_out"
ODIR   = REPO / "activation_question_scores"
FIGDIR = ODIR / "figures"
PARTS  = ODIR / "_parts"                           # per-method worker outputs (job array)
ODIR.mkdir(parents=True, exist_ok=True); FIGDIR.mkdir(parents=True, exist_ok=True); PARTS.mkdir(exist_ok=True)
TASK   = sys.argv[1] if len(sys.argv) > 1 else None    # method id => run only that method (array worker)

N_REPEATS = 20
N_FOLDS   = 5
SEED      = 42
PCA_K     = 256          # in-fold PCA dim cap for logistic regression
METHODS   = [("GradDiff","GradDiff"),("PB_J","PB&J"),("RMU","RMU"),("RMU-LAT","RMU-LAT"),
             ("RepNoise","RepNoise"),("ELM","ELM"),("RR","RR"),("TAR","TAR")]
# contrast -> (positive subset, negative subsets)
CONTRASTS = {
    "suppressed_vs_forgotten": ({"suppressed"}, {"forgotten"}),
    "suppressed_vs_retained":  ({"suppressed"}, {"retained"}),
    "suppressed_vs_rest":      ({"suppressed"}, {"retained","forgotten","lucky"}),
    "forgotten_vs_retained":   ({"forgotten"},  {"retained"}),
}
# feature modes to run (D collapses to C; we keep both labels pointing at the same vector)
FEATURE_MODES = ["correct_only", "mean_all_choices", "correct_minus_mean_wrong"]
PRIMARY_MODE  = "correct_only"          # "Run A as the main version"
PRIMARY_CONTR = "suppressed_vs_forgotten"

def load_k(mid):
    p = OUT / mid / "k_scores.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    df = df[(df.split_type=="cv")&(df.domain=="bio")&(df.clf=="LR")&
            (df.probe_type=="own")&(df.layer_config=="best_layer")]
    return df[["question_idx","k_internal","k_external"]].groupby("question_idx",as_index=False).mean()

def subset_of(ki, ke):
    if ki>0.5 and ke>0.5:  return "retained"
    if ki>0.5 and ke<=0.5: return "suppressed"
    if ki<=0.5 and ke<=0.5:return "forgotten"
    return "lucky"

# ---------------------------------------------------------------- load activations
print("loading base hidden states ...", flush=True)
tf  = pd.read_csv(REPO/"data"/"wmdp_tf_pairs.csv")
ci  = tf.drop_duplicates("original_id").set_index("original_id")["correct_idx"].reindex(range(1273)).astype(int).values
hs  = np.load(OUT/"base"/"bio_hs.npy", mmap_mode="r")          # (1273, 4, 33, 4096)
NQ, NOPT, NL, NH = hs.shape
print("hs", hs.shape, flush=True)

# Build the three distinct MCQ-level feature tensors once over the full question universe.
H = np.asarray(hs, dtype=np.float32)                            # materialize (≈2.7 GB)
rows_q = np.arange(NQ)
A_all = H[rows_q, ci]                                           # (NQ,33,4096) correct option
B_all = H.mean(axis=1)                                          # (NQ,33,4096) mean over 4 options
wrong_sum = H.sum(axis=1) - A_all                               # sum of the 3 wrong options
C_all = A_all - wrong_sum/(NOPT-1)                              # correct - mean(wrong)  (== pairwise)
del H, wrong_sum
FEAT = {"correct_only":A_all, "mean_all_choices":B_all, "correct_minus_mean_wrong":C_all}

# ---------------------------------------------------------------- labels
base_k = load_k("base")
base_sub = {int(r.question_idx): subset_of(r.k_internal, r.k_external) for r in base_k.itertuples()}
base_kint = {int(r.question_idx): float(r.k_internal) for r in base_k.itertuples()}
base_kext = {int(r.question_idx): float(r.k_external) for r in base_k.itertuples()}
QSTAR = sorted(q for q,s in base_sub.items() if base_kint[q]==1.0 and base_kext[q]==1.0)
print("Q*", len(QSTAR), flush=True)

method_lab = {}      # mid -> dict(qid -> (subset,kint,kext))
for mid,_ in METHODS:
    k = load_k(f"{mid}_ck8")
    if k is None: print(mid,"NO DATA",flush=True); continue
    method_lab[mid] = {int(r.question_idx):(subset_of(r.k_internal,r.k_external),
                                            float(r.k_internal),float(r.k_external)) for r in k.itertuples()}

# ---------------------------------------------------------------- helpers
def zscore_fit(Xtr):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True)+1e-6
    return mu, sd

def auc_allcols(S, y):
    """Vectorized rank-sum (Mann-Whitney) AUC for every column of S (n,L) vs binary y -> (L,)."""
    n = len(y); order = np.argsort(S, axis=0, kind="stable")
    ranks = np.empty_like(S, dtype=np.float64)
    rr = np.arange(1, n+1, dtype=np.float64)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rr, S.shape), axis=0)
    npos = float((y==1).sum()); nneg = float(n-npos)
    if npos==0 or nneg==0: return np.full(S.shape[1], np.nan)
    return (ranks[y==1].sum(0) - npos*(npos+1)/2.0) / (npos*nneg)

def select_layer(Xtr, ytr):
    """Best layer using TRAIN ONLY: one stratified inner holdout (resubstitution if a class is tiny).
    Vectorized over all layers (no per-layer sklearn calls)."""
    mc = min((ytr==1).sum(), (ytr==0).sum())
    if mc >= 4:
        itr, ite = train_test_split(np.arange(len(ytr)), test_size=0.25,
                                    stratify=ytr, random_state=0)
        V = Xtr[itr][ytr[itr]==1].mean(0) - Xtr[itr][ytr[itr]==0].mean(0)   # (33,4096)
        S = np.einsum("nlf,lf->nl", Xtr[ite], V)                            # (n_ite,33)
        a = auc_allcols(S, ytr[ite])
    else:                                                                   # tiny class -> resubstitution
        V = Xtr[ytr==1].mean(0) - Xtr[ytr==0].mean(0)
        S = np.einsum("nlf,lf->nl", Xtr, V)
        a = auc_allcols(S, ytr)
    a = np.where(np.isnan(a), 0.5, a)
    return int(np.argmax(a))

def fit_meandiff(Xtr_l, ytr, Xte_l):
    v = Xtr_l[ytr==1].mean(0) - Xtr_l[ytr==0].mean(0)
    s_tr = Xtr_l @ v; s_te = Xte_l @ v
    cal = LogisticRegression(max_iter=1000).fit(s_tr.reshape(-1,1), ytr)   # in-fold Platt
    p_te = cal.predict_proba(s_te.reshape(-1,1))[:,1]
    return s_te, p_te

def fit_logreg(Xtr_l, ytr, Xte_l):
    k = int(min(PCA_K, Xtr_l.shape[0]-1, Xtr_l.shape[1]))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(Xtr_l)
    Ztr = pca.transform(Xtr_l); Zte = pca.transform(Xte_l)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, ytr)
    return clf.decision_function(Zte), clf.predict_proba(Zte)[:,1]

def boot_ci(per_repeat):
    a = np.array([x for x in per_repeat if x is not None and not np.isnan(x)])
    if len(a)==0: return np.nan, np.nan, np.nan
    return float(a.mean()), float(np.percentile(a,2.5)), float(np.percentile(a,97.5))

# ---------------------------------------------------------------- main loop
if TASK is not None:                                # array worker: only this method
    METHODS = [(m,l) for m,l in METHODS if m==TASK]
    assert METHODS, f"unknown method {TASK}"
SUFFIX  = TASK if TASK else "all"
pq_path = PARTS / f"pq_{SUFFIX}.csv.gz"
if pq_path.exists(): pq_path.unlink()
PQ_COLS = ["question_id","method","contrast","feature_mode","direction","question_set",
           "layer","is_selected_layer","cv_repeat","cv_fold","true_subset_ck8","binary_label",
           "projection_score","projection_zscore","predicted_probability","train_or_test",
           "is_out_of_fold","Kint_base","Kext_base","Kint_ck8","Kext_ck8"]
pd.DataFrame(columns=PQ_COLS).to_csv(pq_path, index=False)          # header

summary_rows, quint_rows = [], []
modeA_scores = {}      # (method,contrast) -> per-question mean OOF zscore (mode A, mean-diff, Qstar) for plots

DIRECTIONS = [("mean_difference", fit_meandiff), ("logistic_regression", fit_logreg)]

for qset in ["Qstar","full"]:
    universe = QSTAR if qset=="Qstar" else list(range(NQ))
    modes = FEATURE_MODES if qset=="Qstar" else [PRIMARY_MODE]   # supplementary = mode A only
    for mid,label in METHODS:
        if mid not in method_lab: continue
        lab = method_lab[mid]
        for cname,(pos,neg) in CONTRASTS.items():
            elig = [q for q in universe if q in lab and lab[q][0] in (pos|neg)]
            y_all = np.array([1 if lab[q][0] in pos else 0 for q in elig])
            if y_all.sum()<10 or (1-y_all).sum()<10:
                print(f"[skip] {qset} {label} {cname}: pos={int(y_all.sum())} neg={int((1-y_all).sum())}",flush=True)
                continue
            elig = np.array(elig)
            for mode in modes:
                Xuniv = FEAT[mode]
                Xc = Xuniv[elig]                                  # (n,33,4096)
                # accumulate OOF per direction
                oof = {d:{"score":np.full(len(elig)*N_REPEATS,np.nan),
                          "prob":np.full(len(elig)*N_REPEATS,np.nan),
                          "qid":np.empty(len(elig)*N_REPEATS,int),
                          "rep":np.empty(len(elig)*N_REPEATS,int),
                          "fold":np.empty(len(elig)*N_REPEATS,int),
                          "lay":np.empty(len(elig)*N_REPEATS,int)} for d,_ in DIRECTIONS}
                per_rep_metrics = {d:{m:[] for m in ["auc","bacc","ap","ll","brier"]} for d,_ in DIRECTIONS}
                layerOOF = np.full((len(elig)*N_REPEATS, NL), np.nan) if qset=="Qstar" else None
                for rep in range(N_REPEATS):
                    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED+rep)
                    rep_pred = {d:{"y":[],"p":[]} for d,_ in DIRECTIONS}
                    for fold,(itr,ite) in enumerate(skf.split(Xc, y_all)):
                        Xtr_raw, Xte_raw = Xc[itr], Xc[ite]
                        ytr, yte = y_all[itr], y_all[ite]
                        mu,sd = zscore_fit(Xtr_raw)
                        Xtr = (Xtr_raw-mu)/sd; Xte = (Xte_raw-mu)/sd      # in-fold z-score
                        L = select_layer(Xtr, ytr)                        # in-fold layer pick
                        if layerOOF is not None:                          # all-layer mean-diff OOF (Q5)
                            V = Xtr[ytr==1].mean(0) - Xtr[ytr==0].mean(0)  # (33,4096)
                            layerOOF[rep*len(elig)+ite, :] = np.einsum("nlf,lf->nl", Xte, V)
                        for d,fn in DIRECTIONS:
                            s_te, p_te = fn(Xtr[:,L], ytr, Xte[:,L])
                            # store into flat arrays at positions rep*len+ite
                            base = rep*len(elig)
                            oof[d]["score"][base+ite] = s_te
                            oof[d]["prob"][base+ite]  = p_te
                            oof[d]["qid"][base+ite]   = elig[ite]
                            oof[d]["rep"][base+ite]   = rep
                            oof[d]["fold"][base+ite]  = fold
                            oof[d]["lay"][base+ite]   = L
                            rep_pred[d]["y"].append(yte); rep_pred[d]["p"].append(p_te)
                    for d,_ in DIRECTIONS:
                        yy=np.concatenate(rep_pred[d]["y"]); pp=np.concatenate(rep_pred[d]["p"])
                        if len(np.unique(yy))<2: continue
                        per_rep_metrics[d]["auc"].append(roc_auc_score(yy,pp))
                        per_rep_metrics[d]["bacc"].append(balanced_accuracy_score(yy,(pp>=0.5).astype(int)))
                        per_rep_metrics[d]["ap"].append(average_precision_score(yy,pp))
                        try: per_rep_metrics[d]["ll"].append(log_loss(yy,pp,labels=[0,1]))
                        except Exception: pass
                        per_rep_metrics[d]["brier"].append(brier_score_loss(yy,pp))
                # ---- write per-question OOF + summary per direction
                sub_arr = np.array([lab[q][0] for q in elig])
                for d,_ in DIRECTIONS:
                    sc = oof[d]["score"]; valid = ~np.isnan(sc)
                    zs = np.full_like(sc, np.nan)
                    zs[valid] = (sc[valid]-sc[valid].mean())/(sc[valid].std()+1e-9)
                    qid=oof[d]["qid"]; rr=oof[d]["rep"]; ff=oof[d]["fold"]; ll=oof[d]["lay"]; pr=oof[d]["prob"]
                    df = pd.DataFrame({
                        "question_id":qid[valid], "method":label, "contrast":cname,
                        "feature_mode":mode, "direction":d, "question_set":qset,
                        "layer":ll[valid], "is_selected_layer":True,
                        "cv_repeat":rr[valid], "cv_fold":ff[valid],
                        "true_subset_ck8":[lab[q][0] for q in qid[valid]],
                        "binary_label":[1 if lab[q][0] in pos else 0 for q in qid[valid]],
                        "projection_score":sc[valid], "projection_zscore":zs[valid],
                        "predicted_probability":pr[valid], "train_or_test":"test", "is_out_of_fold":True,
                        "Kint_base":[base_kint.get(q,np.nan) for q in qid[valid]],
                        "Kext_base":[base_kext.get(q,np.nan) for q in qid[valid]],
                        "Kint_ck8":[lab[q][1] for q in qid[valid]],
                        "Kext_ck8":[lab[q][2] for q in qid[valid]],
                    })[PQ_COLS]
                    df.to_csv(pq_path, mode="a", header=False, index=False)
                    am,al,ah = boot_ci(per_rep_metrics[d]["auc"])
                    bm,bl,bh = boot_ci(per_rep_metrics[d]["bacc"])
                    pm,pl,ph = boot_ci(per_rep_metrics[d]["ap"])
                    summary_rows.append(dict(method=label,contrast=cname,feature_mode=mode,direction=d,
                        question_set=qset, layer_selection_mode="per_fold_selected",
                        selected_layer_or_mean=round(float(np.nanmean(ll[valid])),1),
                        n_positive=int(y_all.sum()), n_negative=int((1-y_all).sum()),
                        auc_mean=am,auc_ci95_low=al,auc_ci95_high=ah,
                        balanced_accuracy_mean=bm,balanced_accuracy_ci95_low=bl,balanced_accuracy_ci95_high=bh,
                        average_precision_mean=pm,average_precision_ci95_low=pl,average_precision_ci95_high=ph,
                        log_loss_mean=float(np.mean(per_rep_metrics[d]["ll"])) if per_rep_metrics[d]["ll"] else np.nan,
                        brier_score_mean=float(np.mean(per_rep_metrics[d]["brier"])) if per_rep_metrics[d]["brier"] else np.nan))
                # ---- each_layer AUC profile (mean-diff, Q5 stability) from layerOOF, vectorized
                if layerOOF is not None:
                    per_rep_layer=[]                                   # list of (NL,) AUC per repeat
                    for rep in range(N_REPEATS):
                        block=layerOOF[rep*len(elig):(rep+1)*len(elig)]   # (n_elig,NL)
                        if np.isnan(block).all(): continue
                        per_rep_layer.append(auc_allcols(np.nan_to_num(block,nan=0.0), y_all))
                    per_rep_layer=np.array(per_rep_layer) if per_rep_layer else np.zeros((0,NL))
                    for l in range(NL):
                        m,lo,hi=boot_ci(list(per_rep_layer[:,l])) if per_rep_layer.shape[0] else (np.nan,np.nan,np.nan)
                        summary_rows.append(dict(method=label,contrast=cname,feature_mode=mode,
                            direction="mean_difference",question_set=qset,layer_selection_mode="each_layer",
                            selected_layer_or_mean=l,n_positive=int(y_all.sum()),n_negative=int((1-y_all).sum()),
                            auc_mean=m,auc_ci95_low=lo,auc_ci95_high=hi,
                            balanced_accuracy_mean=np.nan,balanced_accuracy_ci95_low=np.nan,balanced_accuracy_ci95_high=np.nan,
                            average_precision_mean=np.nan,average_precision_ci95_low=np.nan,average_precision_ci95_high=np.nan,
                            log_loss_mean=np.nan,brier_score_mean=np.nan))
                # ---- quintiles + plot cache (mode A, mean-diff, Qstar)
                if qset=="Qstar" and mode==PRIMARY_MODE:
                    d="mean_difference"; sc=oof[d]["score"]; valid=~np.isnan(sc)
                    zs=np.full_like(sc,np.nan); zs[valid]=(sc[valid]-sc[valid].mean())/(sc[valid].std()+1e-9)
                    qid=oof[d]["qid"]
                    dfq=pd.DataFrame({"qid":qid[valid],"z":zs[valid]}).groupby("qid",as_index=False).z.mean()
                    dfq["subset"]=[lab[q][0] for q in dfq.qid]; dfq["pos"]=dfq.subset.isin(pos).astype(int)
                    modeA_scores[(label,cname)]=dfq.copy()
                    try: dfq["quintile"]=pd.qcut(dfq.z,5,labels=False,duplicates="drop")+1
                    except Exception: dfq["quintile"]=1
                    for qv,g in dfq.groupby("quintile"):
                        quint_rows.append(dict(method=label,contrast=cname,feature_mode=mode,
                            direction=d,question_set=qset,quintile=int(qv),n_questions=len(g),
                            positive_rate=float(g.pos.mean()),
                            suppressed_rate=float((g.subset=="suppressed").mean()),
                            forgotten_rate=float((g.subset=="forgotten").mean()),
                            retained_rate=float((g.subset=="retained").mean()),
                            mean_projection_score=float(g.z.mean()),
                            projection_score_min=float(g.z.min()),projection_score_max=float(g.z.max())))
                print(f"[done] {qset} {label} {cname} {mode}",flush=True)

# ---------------------------------------------------------------- write per-method parts (array worker)
pd.DataFrame(summary_rows).to_csv(PARTS/f"summary_{SUFFIX}.csv",index=False)
pd.DataFrame(quint_rows).to_csv(PARTS/f"quint_{SUFFIX}.csv",index=False)
_ma=[]
for (label,cname),g in modeA_scores.items():
    gg=g.copy(); gg["method"]=label; gg["contrast"]=cname; _ma.append(gg)
(pd.concat(_ma,ignore_index=True)[["method","contrast","qid","z","subset","pos"]] if _ma
 else pd.DataFrame(columns=["method","contrast","qid","z","subset","pos"])).to_csv(PARTS/f"modeA_{SUFFIX}.csv",index=False)
print(f"wrote parts for {SUFFIX}; ALL DONE",flush=True)
sys.exit(0)
# ============ everything below is superseded by aggregate_question_level.py (unreachable) ============

# ---------------------------------------------------------------- figures
sumdf=pd.DataFrame(summary_rows)
sel=sumdf[(sumdf.layer_selection_mode=="per_fold_selected")&(sumdf.question_set=="Qstar")&
          (sumdf.direction=="mean_difference")&(sumdf.feature_mode==PRIMARY_MODE)]
mnames=[l for _,l in METHODS]

# 1. suppressed_vs_forgotten projection histograms by method
fig,axes=plt.subplots(2,4,figsize=(18,8))
for ax,label in zip(axes.ravel(),mnames):
    g=modeA_scores.get((label,PRIMARY_CONTR))
    if g is None: ax.set_title(f"{label} (n/a)"); ax.axis("off"); continue
    ax.hist(g[g.subset=="suppressed"].z,bins=20,alpha=.6,label="suppressed",color="#d62728",density=True)
    ax.hist(g[g.subset=="forgotten"].z,bins=20,alpha=.6,label="forgotten",color="#1f77b4",density=True)
    ax.set_title(label);ax.set_xlabel("projection z")
axes.ravel()[0].legend()
fig.suptitle("Base-model projection scores: suppressed vs forgotten (mode A, mean-diff, OOF)")
fig.tight_layout();fig.savefig(FIGDIR/"suppressed_vs_forgotten_projection_histograms_by_method.png",dpi=130);plt.close(fig)

# 2. P(suppressed | quintile) by method
qdf=pd.DataFrame(quint_rows); qsf=qdf[qdf.contrast==PRIMARY_CONTR]
fig,ax=plt.subplots(figsize=(11,6))
for label in mnames:
    g=qsf[qsf.method==label].sort_values("quintile")
    if len(g): ax.plot(g.quintile,g.positive_rate,marker="o",label=label)
ax.set_xlabel("projection-score quintile (low→high)");ax.set_ylabel("P(suppressed | quintile)")
ax.set_title("Monotonicity of suppression rate across projection quintiles");ax.legend(ncol=2,fontsize=8)
ax.axhline(0.5,ls="--",c="gray",lw=.8);fig.tight_layout()
fig.savefig(FIGDIR/"suppressed_vs_forgotten_projection_quintile_rates.png",dpi=130);plt.close(fig)

# 3. AUC by method, grouped bars over contrasts
fig,ax=plt.subplots(figsize=(13,6));cl=list(CONTRASTS);w=.2;x=np.arange(len(mnames))
for i,c in enumerate(cl):
    vals=[];errs=[]
    for label in mnames:
        r=sel[(sel.method==label)&(sel.contrast==c)]
        if len(r): vals.append(r.auc_mean.values[0]); errs.append((r.auc_mean.values[0]-r.auc_ci95_low.values[0]))
        else: vals.append(np.nan);errs.append(0)
    ax.bar(x+i*w,vals,w,yerr=errs,capsize=2,label=c)
ax.set_xticks(x+1.5*w);ax.set_xticklabels(mnames,rotation=30,ha="right")
ax.axhline(0.5,ls="--",c="gray");ax.set_ylabel("OOF AUC (mean-diff)");ax.set_ylim(0.3,0.8)
ax.set_title("Question-level base-geometry predictiveness by method & contrast (Q*, mode A)")
ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIGDIR/"activation_direction_auc_by_method.png",dpi=130);plt.close(fig)

# 4. projection-score boxplots by true subset, per method (primary contrast universe)
fig,axes=plt.subplots(2,4,figsize=(18,8))
for ax,label in zip(axes.ravel(),mnames):
    g=modeA_scores.get((label,PRIMARY_CONTR))
    if g is None: ax.set_title(f"{label} (n/a)");ax.axis("off");continue
    data=[g[g.subset==s].z.values for s in ["suppressed","forgotten"]]
    ax.boxplot(data,labels=["supp","forg"]);ax.set_title(label);ax.axhline(0,ls="--",c="gray",lw=.6)
fig.suptitle("Projection-z by eventual subset (mode A, OOF)")
fig.tight_layout();fig.savefig(FIGDIR/"per_method_projection_score_boxplots.png",dpi=130);plt.close(fig)
print("wrote figures",flush=True)

# ---------------------------------------------------------------- REPORT.md
def auc_cell(label,c):
    r=sel[(sel.method==label)&(sel.contrast==c)]
    if not len(r): return "n/a"
    return f"{r.auc_mean.values[0]:.3f} [{r.auc_ci95_low.values[0]:.3f},{r.auc_ci95_high.values[0]:.3f}]"

lines=[]
lines.append("# Question-level base-model activation-direction analysis — REPORT\n")
lines.append("## Data sources")
lines.append("- Features: `inside_out_out/base/bio_hs.npy` (BASE model, shape (1273,4,33,4096)); MCQ-level vectors per question.")
lines.append("- Labels: `inside_out_out/{method}_ck8/k_scores.parquet` (LR / own / best_layer / cv / bio), MCQ-level Kint/Kext.")
lines.append("- Base Q* membership + Kint/Kext from `inside_out_out/base/k_scores.parquet`.\n")
lines.append("## Subset definition (MCQ-level, paper rule)")
lines.append("suppressed: Kint>0.5 & Kext<=0.5 · forgotten: Kint<=0.5 & Kext<=0.5 · retained: Kint>0.5 & Kext>0.5 · lucky: Kint<=0.5 & Kext>0.5\n")
lines.append(f"## Question set\nPrimary: **Q\\*** (n={len(QSTAR)}, base knows on both axes). Supplementary: **full** 1273-question set (mode A only), `question_set` column.\n")
lines.append("## Feature aggregation rule")
lines.append("- A `correct_only` = correct-option hidden state (main; matches existing activation-vector code).")
lines.append("- B `mean_all_choices` = mean over the 4 option claims.")
lines.append("- C `correct_minus_mean_wrong` = correct − mean(wrong).")
lines.append("- D `pairwise_correct_minus_wrong` = mean_w(correct − wrong_w) **= C algebraically** under MCQ-level aggregation, so it is not run separately (identical vector).\n")
lines.append("## Leakage prevention")
lines.append("Per fold, fit on TRAIN only: (i) per-(layer,feature) z-scoring, (ii) best-layer selection via inner train CV (resubstitution when a class <3), (iii) PCA-"+str(PCA_K)+" for logistic regression, (iv) Platt prob calibration for mean-diff. Grouped CV by question_id (singleton groups ⇒ StratifiedKFold==GroupKFold); 5 folds × "+str(N_REPEATS)+" repeats, seed "+str(SEED)+". Only OOF test rows are exported/evaluated.\n")
lines.append("## Per-method OOF AUC (Q*, mode A, mean-diff direction, per-fold layer selection)\n")
lines.append("| method | suppressed_vs_forgotten | suppressed_vs_retained | suppressed_vs_rest | forgotten_vs_retained | n_supp | n_forg |")
lines.append("|---|---|---|---|---|---|---|")
for label in mnames:
    rsf=sel[(sel.method==label)&(sel.contrast=="suppressed_vs_rest")]
    rff=sel[(sel.method==label)&(sel.contrast=="suppressed_vs_forgotten")]
    nsupp=int(rsf.n_positive.values[0]) if len(rsf) else 0
    nforg=int(rff.n_negative.values[0]) if len(rff) else 0
    lines.append(f"| {label} | {auc_cell(label,'suppressed_vs_forgotten')} | {auc_cell(label,'suppressed_vs_retained')} | {auc_cell(label,'suppressed_vs_rest')} | {auc_cell(label,'forgotten_vs_retained')} | {nsupp} | {nforg} |")
# threshold / separation verdict
clean=False
for (label,c),g in modeA_scores.items():
    if c!=PRIMARY_CONTR: continue
    sup=g[g.subset=="suppressed"].z; forg=g[g.subset=="forgotten"].z
    if len(sup) and len(forg) and (sup.min()>forg.max() or forg.min()>sup.max()): clean=True
lines.append("\n## Projection-quintile result (primary contrast)")
lines.append("See `activation_direction_quantile_summary.csv` and the quintile figure: P(suppressed|quintile). "
             "A monotone increase indicates usable graded signal even without a hard threshold.\n")
lines.append("## Is there a clean threshold?")
lines.append(("**Yes** — at least one method shows non-overlapping suppressed/forgotten projection ranges." if clean
              else "**No** — suppressed and forgotten projection-score ranges overlap for every method; separation is partial, not a deterministic rule.")+"\n")
lines.append("## Reporting language")
lines.append("Base-model hidden-state geometry contains **weak/moderate, method-specific** information about future suppressibility; "
             "it does **not** determine whether a question will be suppressed.\n")
lines.append("## Limitations")
lines.append("- Forgotten pools on Q* are small (see n_forg column); suppressed_vs_forgotten is underpowered for low-n methods (GradDiff/RR/RMU-LAT). The full-set supplementary partially mitigates this.")
lines.append("- Effect sizes are modest (AUC ~0.55–0.66); diff-of-means is conservative; in-fold z-scoring differs slightly from the earlier global-z-scored numbers.")
lines.append("- 'lucky' excluded from primary contrasts; RMU/RMU-LAT carry little suppressed signal in base geometry (consistent with prior analysis).")
(ODIR/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
print("wrote REPORT.md",flush=True)
print("ALL DONE",flush=True)
