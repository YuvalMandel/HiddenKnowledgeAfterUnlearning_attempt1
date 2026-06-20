# Distractor-structure analysis (Ziv task): does preserving the 3 wrong-answer structure
# improve suppressed-vs-forgotten ranking over the correct-minus-mean-wrong baseline?
# Array worker: one method per task -> writes parts to activation_question_scores/_parts_distractor/.
#
# Feature variants (base-model only; MCQ-level; per layer); distractor selection is
# LABEL-INDEPENDENT (uses base ext score or base hidden-state distance, never the subset label):
#   correct_only                         (reference)
#   correct_minus_mean_wrong             (baseline = mode C)
#   correct_minus_strongest_wrong        wrong with max base ext score
#   correct_minus_closest_wrong          wrong with min mean-layer L2 distance to correct
#   correct_minus_farthest_wrong         wrong with max mean-layer L2 distance to correct
#   concat_all_wrong_ordered             [c-w1 ; c-w2 ; c-w3] ordered strongest->weakest by ext (3*4096)
#   mean_plus_spread                     (c - mean wrong) ++ 4 per-layer spread scalars (4096+4)
# Contrast: suppressed_vs_forgotten on Q*. Directions: mean_difference + logistic_regression.
import os, sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score

REPO=Path("."); OUT=REPO/"inside_out_out"; EXT=REPO/"inside_out_ext"
ODIR=REPO/"activation_question_scores"; PARTS=ODIR/"_parts_distractor"
ODIR.mkdir(exist_ok=True); PARTS.mkdir(parents=True, exist_ok=True)
TASK=sys.argv[1] if len(sys.argv)>1 else None
N_REPEATS=20; N_FOLDS=5; SEED=42; PCA_K=256
METHODS=[("GradDiff","GradDiff"),("PB_J","PB&J"),("RMU","RMU"),("RMU-LAT","RMU-LAT"),
         ("RepNoise","RepNoise"),("ELM","ELM"),("RR","RR"),("TAR","TAR")]
VARIANTS=["correct_only","correct_minus_mean_wrong","correct_minus_strongest_wrong",
          "correct_minus_closest_wrong","correct_minus_farthest_wrong",
          "concat_all_wrong_ordered","mean_plus_spread"]
RULE={"correct_only":"none","correct_minus_mean_wrong":"mean_all_wrong",
      "correct_minus_strongest_wrong":"max_base_ext","correct_minus_closest_wrong":"min_hidden_dist",
      "correct_minus_farthest_wrong":"max_hidden_dist","concat_all_wrong_ordered":"all_ordered_by_ext",
      "mean_plus_spread":"mean_plus_spread"}

def load_k(mid):
    p=OUT/mid/"k_scores.parquet"
    if not p.exists(): return None
    df=pd.read_parquet(p)
    df=df[(df.split_type=="cv")&(df.domain=="bio")&(df.clf=="LR")&(df.probe_type=="own")&(df.layer_config=="best_layer")]
    return df[["question_idx","k_internal","k_external"]].groupby("question_idx",as_index=False).mean()

def subset_of(ki,ke):
    if ki>0.5 and ke>0.5: return "retained"
    if ki>0.5 and ke<=0.5: return "suppressed"
    if ki<=0.5 and ke<=0.5: return "forgotten"
    return "lucky"

def auc_allcols(S,y):
    n=len(y); order=np.argsort(S,axis=0,kind="stable"); ranks=np.empty_like(S,dtype=np.float64)
    rr=np.arange(1,n+1,dtype=np.float64)[:,None]; np.put_along_axis(ranks,order,np.broadcast_to(rr,S.shape),axis=0)
    npos=float((y==1).sum()); nneg=float(n-npos)
    if npos==0 or nneg==0: return np.full(S.shape[1],np.nan)
    return (ranks[y==1].sum(0)-npos*(npos+1)/2.0)/(npos*nneg)

def select_layer(Xtr,ytr):
    mc=min((ytr==1).sum(),(ytr==0).sum())
    if mc>=4:
        itr,ite=train_test_split(np.arange(len(ytr)),test_size=0.25,stratify=ytr,random_state=0)
        V=Xtr[itr][ytr[itr]==1].mean(0)-Xtr[itr][ytr[itr]==0].mean(0)
        S=np.einsum("nlf,lf->nl",Xtr[ite],V); a=auc_allcols(S,ytr[ite])
    else:
        V=Xtr[ytr==1].mean(0)-Xtr[ytr==0].mean(0); S=np.einsum("nlf,lf->nl",Xtr,V); a=auc_allcols(S,ytr)
    return int(np.argmax(np.where(np.isnan(a),0.5,a)))

def fit_meandiff(Xtr_l,ytr,Xte_l):
    v=Xtr_l[ytr==1].mean(0)-Xtr_l[ytr==0].mean(0); s_tr=Xtr_l@v; s_te=Xte_l@v
    cal=LogisticRegression(max_iter=1000).fit(s_tr.reshape(-1,1),ytr)
    return s_te, cal.predict_proba(s_te.reshape(-1,1))[:,1]

def fit_logreg(Xtr_l,ytr,Xte_l):
    k=int(min(PCA_K,Xtr_l.shape[0]-1,Xtr_l.shape[1]))
    pca=PCA(n_components=k,svd_solver="randomized",random_state=0).fit(Xtr_l)
    Ztr=pca.transform(Xtr_l); Zte=pca.transform(Xte_l)
    clf=LogisticRegression(max_iter=2000,C=1.0).fit(Ztr,ytr)
    return clf.decision_function(Zte), clf.predict_proba(Zte)[:,1]

def boot_ci(v):
    a=np.array([x for x in v if x is not None and not np.isnan(x)])
    if len(a)==0: return np.nan,np.nan,np.nan
    return float(a.mean()),float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

# ---------------- load
tf=pd.read_csv(REPO/"data"/"wmdp_tf_pairs.csv")
ci=tf.drop_duplicates("original_id").set_index("original_id")["correct_idx"].reindex(range(1273)).astype(int).values
hs=np.load(OUT/"base"/"bio_hs.npy",mmap_mode="r")           # (1273,4,33,4096)
ext=np.load(EXT/"base_bio_ext.npy")                          # (1273,4) base per-option external score
base_k=load_k("base")
base_kint={int(r.question_idx):float(r.k_internal) for r in base_k.itertuples()}
base_kext={int(r.question_idx):float(r.k_external) for r in base_k.itertuples()}
QSTAR=sorted(q for q in base_kint if base_kint[q]==1.0 and base_kext[q]==1.0)
if TASK is not None:
    METHODS=[(m,l) for m,l in METHODS if m==TASK]; assert METHODS,f"unknown {TASK}"
SUFFIX=TASK if TASK else "all"
print("Q*",len(QSTAR),"method(s)",[l for _,l in METHODS],flush=True)

DIRECTIONS=[("mean_difference",fit_meandiff),("logistic_regression",fit_logreg)]
PQ_COLS=["question_id","method","feature_variant","selected_distractor_id","selected_distractor_rule",
         "true_subset","binary_label","direction","oof_score","oof_probability","cv_repeat","cv_fold"]
pq_path=PARTS/f"pq_{SUFFIX}.csv.gz"
if pq_path.exists(): pq_path.unlink()
pd.DataFrame(columns=PQ_COLS).to_csv(pq_path,index=False)
summary_rows=[]

def build_variant(name, He, cie, exte):
    """He:(n,4,33,F0) raw; returns X:(n,33,F), sel_id:(n,) int (-1 if n/a)."""
    n=He.shape[0]; ar=np.arange(n)
    Hc=He[ar,cie]                                           # (n,33,4096) correct option
    wrong_mask=np.ones((n,4),bool); wrong_mask[ar,cie]=False
    wrong_idx=np.array([np.where(wrong_mask[i])[0] for i in range(n)])   # (n,3)
    # order wrongs strongest->weakest by base ext
    order=np.argsort(-exte[ar[:,None],wrong_idx],axis=1)
    wrong_ord=np.take_along_axis(wrong_idx,order,axis=1)     # (n,3) ext desc
    # mean-layer L2 distance of each wrong to correct (base hidden states)
    dist=np.full((n,4),np.inf)
    for j in range(4):
        d=He[:,j]-Hc                                         # (n,33,4096)
        dist[:,j]=np.sqrt((d*d).sum(-1)).mean(-1)
    dist[ar,cie]=np.inf
    meanwrong=(He.sum(1)-Hc)/3.0
    if name=="correct_only":           return Hc, np.full(n,-1)
    if name=="correct_minus_mean_wrong": return Hc-meanwrong, np.full(n,-1)
    if name=="correct_minus_strongest_wrong":
        s=wrong_ord[:,0]; return Hc-He[ar,s], s
    if name=="correct_minus_closest_wrong":
        s=np.argmin(dist,axis=1); return Hc-He[ar,s], s
    if name=="correct_minus_farthest_wrong":
        d2=dist.copy(); d2[np.isinf(d2)]=-np.inf; s=np.argmax(d2,axis=1); return Hc-He[ar,s], s
    if name=="concat_all_wrong_ordered":
        parts=[Hc-He[ar,wrong_ord[:,k]] for k in range(3)]   # each (n,33,4096) ordered
        return np.concatenate(parts,axis=2), np.full(n,-1)
    if name=="mean_plus_spread":
        d=[Hc-He[ar,wrong_ord[:,k]] for k in range(3)]       # 3 x (n,33,4096)
        norms=np.stack([np.sqrt((dk*dk).sum(-1)) for dk in d],axis=-1)  # (n,33,3)
        pw=[np.sqrt(((d[a]-d[b])**2).sum(-1)) for a,b in [(0,1),(0,2),(1,2)]]  # 3 x (n,33)
        pw=np.stack(pw,axis=-1)                               # (n,33,3)
        spread=np.stack([pw.mean(-1),pw.min(-1),pw.max(-1),norms.std(-1)],axis=-1)  # (n,33,4)
        return np.concatenate([Hc-meanwrong,spread],axis=2), np.full(n,-1)
    raise ValueError(name)

for mid,label in METHODS:
    k=load_k(f"{mid}_ck8")
    if k is None: print(mid,"NO DATA",flush=True); continue
    lab={int(r.question_idx):(subset_of(r.k_internal,r.k_external),float(r.k_internal),float(r.k_external))
         for r in k.itertuples()}
    elig=np.array([q for q in QSTAR if q in lab and lab[q][0] in ("suppressed","forgotten")])
    y=np.array([1 if lab[q][0]=="suppressed" else 0 for q in elig])
    if y.sum()<10 or (1-y).sum()<10:
        print(f"[skip] {label}: supp={int(y.sum())} forg={int((1-y).sum())}",flush=True); continue
    He=np.asarray(hs[elig],dtype=np.float32)                 # (n,4,33,4096)
    cie=ci[elig]; exte=ext[elig]
    for variant in VARIANTS:
        Xc,sel=build_variant(variant,He,cie,exte)
        oof={d:{"score":np.full(len(elig)*N_REPEATS,np.nan),"prob":np.full(len(elig)*N_REPEATS,np.nan),
                "qid":np.empty(len(elig)*N_REPEATS,int),"rep":np.empty(len(elig)*N_REPEATS,int),
                "fold":np.empty(len(elig)*N_REPEATS,int)} for d,_ in DIRECTIONS}
        prm={d:{m:[] for m in ["auc","bacc","ap"]} for d,_ in DIRECTIONS}
        for rep in range(N_REPEATS):
            skf=StratifiedKFold(N_FOLDS,shuffle=True,random_state=SEED+rep)
            rp={d:{"y":[],"p":[]} for d,_ in DIRECTIONS}
            for fold,(itr,ite) in enumerate(skf.split(Xc,y)):
                mu=Xc[itr].mean(0,keepdims=True); sd=Xc[itr].std(0,keepdims=True)+1e-6
                Xtr=(Xc[itr]-mu)/sd; Xte=(Xc[ite]-mu)/sd
                L=select_layer(Xtr,y[itr]); base=rep*len(elig)
                for d,fn in DIRECTIONS:
                    s_te,p_te=fn(Xtr[:,L],y[itr],Xte[:,L])
                    oof[d]["score"][base+ite]=s_te; oof[d]["prob"][base+ite]=p_te
                    oof[d]["qid"][base+ite]=elig[ite]; oof[d]["rep"][base+ite]=rep; oof[d]["fold"][base+ite]=fold
                    rp[d]["y"].append(y[ite]); rp[d]["p"].append(p_te)
            for d,_ in DIRECTIONS:
                yy=np.concatenate(rp[d]["y"]); pp=np.concatenate(rp[d]["p"])
                if len(np.unique(yy))<2: continue
                prm[d]["auc"].append(roc_auc_score(yy,pp))
                prm[d]["bacc"].append(balanced_accuracy_score(yy,(pp>=0.5).astype(int)))
                prm[d]["ap"].append(average_precision_score(yy,pp))
        selmap={int(q):int(s) for q,s in zip(elig,sel)}
        for d,_ in DIRECTIONS:
            sc=oof[d]["score"]; valid=~np.isnan(sc); qid=oof[d]["qid"]
            df=pd.DataFrame({"question_id":qid[valid],"method":label,"feature_variant":variant,
                "selected_distractor_id":[selmap[q] for q in qid[valid]],"selected_distractor_rule":RULE[variant],
                "true_subset":[lab[q][0] for q in qid[valid]],
                "binary_label":[1 if lab[q][0]=="suppressed" else 0 for q in qid[valid]],
                "direction":d,"oof_score":sc[valid],"oof_probability":oof[d]["prob"][valid],
                "cv_repeat":oof[d]["rep"][valid],"cv_fold":oof[d]["fold"][valid]})[PQ_COLS]
            df.to_csv(pq_path,mode="a",header=False,index=False)
            am,al,ah=boot_ci(prm[d]["auc"]); bm,bl,bh=boot_ci(prm[d]["bacc"]); pm,pl,ph=boot_ci(prm[d]["ap"])
            summary_rows.append(dict(method=label,feature_variant=variant,direction=d,
                selected_distractor_rule=RULE[variant],n_suppressed=int(y.sum()),n_forgotten=int((1-y).sum()),
                auc_mean=am,auc_ci95_low=al,auc_ci95_high=ah,
                balanced_accuracy_mean=bm,balanced_accuracy_ci95_low=bl,balanced_accuracy_ci95_high=bh,
                average_precision_mean=pm,average_precision_ci95_low=pl,average_precision_ci95_high=ph))
        print(f"[done] {label} {variant}",flush=True)
        del Xc

pd.DataFrame(summary_rows).to_csv(PARTS/f"summary_{SUFFIX}.csv",index=False)
print(f"wrote parts for {SUFFIX}; ALL DONE",flush=True)
