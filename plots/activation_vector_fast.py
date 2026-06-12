# Vectorized per-layer activation-vector analysis on BASE model.
# Batched permutations (matmul) + rank-sum AUC. Labels = each method's ck8 subset.
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

REPO=Path('.'); OUT=REPO/'inside_out_out'
SAVE=REPO/'plots'/'activation_vectors'; SAVE.mkdir(parents=True,exist_ok=True)
NPERM=500; SEED=0; rng=np.random.default_rng(SEED)
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
hs=np.load(OUT/'base'/'bio_hs.npy',mmap_mode='r')
Hbase=np.stack([np.asarray(hs[q,ci[q]],dtype=np.float32) for q in qstar])
Xs=((Hbase-Hbase.mean(0,keepdims=True))/(Hbase.std(0,keepdims=True)+1e-6)).astype(np.float32)
print("Q*",len(qstar),"Xs",Xs.shape,flush=True)
qpos={q:i for i,q in enumerate(qstar)}

def subset_of(ki,ke):
    if ki>0.5 and ke>0.5: return 'retained'
    if ki>0.5 and ke<=0.5: return 'suppressed'
    if ki<=0.5 and ke<=0.5: return 'forgotten'
    return 'lucky'

def auc_cols(scores,Y):                      # scores (n,C), Y (n,C) binary -> (C,)
    n=scores.shape[0]
    order=np.argsort(scores,axis=0,kind='stable')
    ranks=np.empty_like(scores)
    rr=np.arange(1,n+1,dtype=np.float32)[:,None]
    np.put_along_axis(ranks,order,np.broadcast_to(rr,scores.shape),axis=0)
    npos=Y.sum(0); nneg=n-npos
    return (( (ranks*Y).sum(0) - npos*(npos+1)/2 )/(np.maximum(npos,1)*np.maximum(nneg,1)))

def run_contrast(X,y,label,cname):
    n,L,F=X.shape
    folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    Y=np.empty((n,NPERM+1),dtype=np.float32); Y[:,0]=y
    for b in range(1,NPERM+1): Y[:,b]=rng.permutation(y)
    oof=np.zeros((n,L,NPERM+1),dtype=np.float32)
    for tr,te in folds:
        Ytr=Y[tr]; Xtr=X[tr]; Xte=X[te]
        npos=np.maximum(Ytr.sum(0),1); nneg=np.maximum(Xtr.shape[0]-Ytr.sum(0),1)
        for l in range(L):
            Xl=Xtr[:,l,:]; sp=Ytr.T@Xl; sall=Xl.sum(0); sn=sall[None,:]-sp
            v=(sp/npos[:,None])-(sn/nneg[:,None])
            oof[te,l,:]=Xte[:,l,:]@v.T
    Yexp=np.broadcast_to(Y[:,None,:],(n,L,NPERM+1)).reshape(n,L*(NPERM+1))
    A=auc_cols(oof.reshape(n,L*(NPERM+1)),Yexp).reshape(L,NPERM+1)
    real=A[:,0]; bL=int(real.argmax()); bA=float(real[bL])
    nullmax=A[:,1:].max(0); p=(1+int((nullmax>=bA).sum()))/(NPERM+1)
    yb=y.astype(bool); v=X[yb][:,bL].mean(0)-X[~yb][:,bL].mean(0)
    print(f"{label:9} {cname:24} bestL={bL:2} AUC={bA:.3f} p={p:.4f} null95={np.quantile(nullmax,.95):.3f}",flush=True)
    return dict(method=label,contrast=cname,npos=int(yb.sum()),nneg=int((~yb).sum()),best_layer=bL,
        best_cv_auc=round(bA,4),perm_p=round(p,4),null95=round(float(np.quantile(nullmax,.95)),4),
        auc_profile=";".join(f"{a:.3f}" for a in real)), v.astype(np.float32)

rows=[]; vecs={}
for mname,label in METHODS:
    k=load_k(f"{mname}_ck8")
    if k is None: print(label,"NO DATA",flush=True); continue
    km=k.set_index('question_idx')
    keep=[q for q in qstar if q in km.index]; idx=np.array([qpos[q] for q in keep])
    sub=np.array([subset_of(km.loc[q,'k_internal'],km.loc[q,'k_external']) for q in keep]); Xm=Xs[idx]
    for s in SUBSETS:
        y=(sub==s).astype(np.float32)
        if y.sum()<10 or (1-y).sum()<10: print(f"{label} {s}_vs_rest skip",flush=True); continue
        r,v=run_contrast(Xm,y,label,f"{s}_vs_rest"); rows.append(r); vecs[f"{mname}::{s}_vs_rest"]=v
    m2=np.isin(sub,['suppressed','retained'])
    if m2.sum()>=20:
        r,v=run_contrast(Xm[m2],(sub[m2]=='suppressed').astype(np.float32),label,"suppressed_vs_retained"); rows.append(r); vecs[f"{mname}::suppressed_vs_retained"]=v

res=pd.DataFrame(rows); res['perm_q']=np.nan
for c in res.contrast.unique():
    ii=res.index[res.contrast==c]; pv=res.loc[ii,'perm_p'].values
    o=np.argsort(pv); m=len(pv); q=pv[o]*m/(np.arange(m)+1)
    q=np.minimum.accumulate(q[::-1])[::-1]; qq=np.empty(m); qq[o]=np.clip(q,0,1); res.loc[ii,'perm_q']=qq
res.to_csv(SAVE/'activation_vector_base_results.csv',index=False)
np.savez(SAVE/'steering_vectors_base_bestlayer.npz',**vecs)
print("\nSAVED",flush=True)
print(res[['method','contrast','best_layer','best_cv_auc','perm_p','perm_q']].to_string(index=False),flush=True)
