# Per-layer activation (steering) vector analysis on the BASE model.
# Q: in the PRE-unlearning representation, is there a significant diff-of-means
# direction that separates the questions a method will SUPPRESS (vs forget/retain/etc)?
# Activations: base model. Labels: each method's ck8 subset. Non-circular.
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

REPO=Path('.'); OUT=REPO/'inside_out_out'
SAVE=REPO/'plots'/'activation_vectors'; SAVE.mkdir(parents=True, exist_ok=True)
NPERM=500; SEED=0
rng=np.random.default_rng(SEED)
METHODS=[("GradDiff","GradDiff"),("PB_J","PB&J"),("RMU","RMU"),("RMU-LAT","RMU-LAT"),
         ("RepNoise","RepNoise"),("ELM","ELM"),("RR","RR"),("TAR","TAR")]
SUBSETS=["retained","suppressed","forgotten","lucky"]

def load_k(mid):
    p=OUT/mid/'k_scores.parquet'
    if not p.exists(): return None
    df=pd.read_parquet(p)
    df=df[(df.split_type=='cv')&(df.domain=='bio')&(df.clf=='LR')&(df.probe_type=='own')&(df.layer_config=='best_layer')]
    return df[['question_idx','k_internal','k_external']].groupby('question_idx',as_index=False).mean()

tf=pd.read_csv(REPO/'data'/'wmdp_tf_pairs.csv')
ci=tf.drop_duplicates('original_id').set_index('original_id')['correct_idx'].reindex(range(1273)).astype(int).values
base=load_k('base')
qstar=sorted(base.loc[(base.k_internal==1.0)&(base.k_external==1.0),'question_idx'].astype(int).tolist())
print("Q*",len(qstar),flush=True)

# BASE activations, loaded ONCE, correct-option last-token hidden state, z-scored per (layer,feat)
hs=np.load(OUT/'base'/'bio_hs.npy',mmap_mode='r')
Hbase=np.stack([np.asarray(hs[q,ci[q]],dtype=np.float32) for q in qstar])   # (701,33,4096)
Xs=(Hbase-Hbase.mean(0,keepdims=True))/(Hbase.std(0,keepdims=True)+1e-6)
print("Xs",Xs.shape,flush=True)

def subset_of(ki,ke):
    if ki>0.5 and ke>0.5: return 'retained'
    if ki>0.5 and ke<=0.5: return 'suppressed'
    if ki<=0.5 and ke<=0.5: return 'forgotten'
    return 'lucky'

def oof_auc_layers(X,y,folds):
    n,L,F=X.shape; oof=np.zeros((n,L),dtype=np.float32)
    for tr,te in folds:
        ytr=y[tr]; v=X[tr][ytr==1].mean(0)-X[tr][ytr==0].mean(0)
        oof[te]=np.einsum('nlf,lf->nl',X[te],v,optimize=True)
    return np.array([roc_auc_score(y,oof[:,l]) for l in range(L)])

qpos={q:i for i,q in enumerate(qstar)}
rows=[]; vecs={}
for mname,label in METHODS:
    k=load_k(f"{mname}_ck8")
    if k is None: print(label,"NO DATA",flush=True); continue
    km=k.set_index('question_idx')
    idx=np.array([qpos[q] for q in qstar if q in km.index])
    sub=np.array([subset_of(km.loc[q,'k_internal'],km.loc[q,'k_external']) for q in qstar if q in km.index])
    Xm=Xs[idx]
    contrasts=[(f"{s}_vs_rest",Xm,(sub==s).astype(int)) for s in SUBSETS]
    m2=np.isin(sub,['suppressed','retained'])
    contrasts.append(("suppressed_vs_retained",Xm[m2],(sub[m2]=='suppressed').astype(int)))
    for cname,Xc,yc in contrasts:
        npos,nneg=int(yc.sum()),int((1-yc).sum())
        if npos<10 or nneg<10: print(f"{label} {cname}: skip ({npos}/{nneg})",flush=True); continue
        folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(Xc,yc))
        aucs=oof_auc_layers(Xc,yc,folds); bL=int(aucs.argmax()); bA=float(aucs[bL])
        cnt=0; nm=np.empty(NPERM)
        for i in range(NPERM):
            a=oof_auc_layers(Xc,rng.permutation(yc),folds); nm[i]=a.max()
            if nm[i]>=bA: cnt+=1
        p=(1+cnt)/(NPERM+1)
        vecs[f"{mname}::{cname}"]=(Xc[yc==1][:,bL].mean(0)-Xc[yc==0][:,bL].mean(0)).astype(np.float32)
        rows.append(dict(method=label,contrast=cname,npos=npos,nneg=nneg,best_layer=bL,
            best_cv_auc=round(bA,4),perm_p=round(p,4),null95=round(float(np.quantile(nm,.95)),4),
            auc_profile=";".join(f"{a:.3f}" for a in aucs)))
        print(f"{label:9} {cname:24} bestL={bL:2} AUC={bA:.3f} p={p:.4f} null95={np.quantile(nm,.95):.3f}",flush=True)

res=pd.DataFrame(rows); res['perm_q']=np.nan
for c in res.contrast.unique():
    ii=res.index[res.contrast==c]; pv=res.loc[ii,'perm_p'].values
    o=np.argsort(pv); m=len(pv); q=pv[o]*m/(np.arange(m)+1)
    q=np.minimum.accumulate(q[::-1])[::-1]; qq=np.empty(m); qq[o]=np.clip(q,0,1); res.loc[ii,'perm_q']=qq
res.to_csv(SAVE/'activation_vector_base_results.csv',index=False)
np.savez(SAVE/'steering_vectors_base_bestlayer.npz',**vecs)
print("\nSAVED to",SAVE,flush=True)
print(res[['method','contrast','best_layer','best_cv_auc','perm_p','perm_q']].to_string(index=False),flush=True)
