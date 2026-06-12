# All-layers-concat + band-concat diff-of-means AUC (+perm) and cross-method cosine, BASE model.
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
REPO=Path('.'); OUT=REPO/'inside_out_out'; SAVE=REPO/'plots'/'activation_vectors'
NPERM=500; SEED=0
METHODS=[("GradDiff","GradDiff"),("PB_J","PB&J"),("RMU","RMU"),("RMU-LAT","RMU-LAT"),
         ("RepNoise","RepNoise"),("ELM","ELM"),("RR","RR"),("TAR","TAR")]
SUBSETS=["retained","suppressed","forgotten","lucky"]
def load_k(mid):
    p=OUT/mid/'k_scores.parquet'
    if not p.exists(): return None
    df=pd.read_parquet(p); df=df[(df.split_type=='cv')&(df.domain=='bio')&(df.clf=='LR')&(df.probe_type=='own')&(df.layer_config=='best_layer')]
    return df[['question_idx','k_internal','k_external']].groupby('question_idx',as_index=False).mean()
tf=pd.read_csv(REPO/'data'/'wmdp_tf_pairs.csv')
ci=tf.drop_duplicates('original_id').set_index('original_id')['correct_idx'].reindex(range(1273)).astype(int).values
base=load_k('base'); qstar=sorted(base.loc[(base.k_internal==1)&(base.k_external==1),'question_idx'].astype(int))
hs=np.load(OUT/'base'/'bio_hs.npy',mmap_mode='r')
H=np.stack([np.asarray(hs[q,ci[q]],dtype=np.float32) for q in qstar])
Xs=((H-H.mean(0,keepdims=True))/(H.std(0,keepdims=True)+1e-6)).astype(np.float32)
print("Q*",len(qstar),"Xs",Xs.shape,flush=True)
qpos={q:i for i,q in enumerate(qstar)}
def sub_of(ki,ke):
    if ki>.5 and ke>.5: return 'retained'
    if ki>.5: return 'suppressed'
    if ke<=.5: return 'forgotten'
    return 'lucky'
def auc_cols(s,Y):
    n=s.shape[0]; order=np.argsort(s,0,kind='stable'); r=np.empty_like(s)
    rr=np.arange(1,n+1,dtype=np.float32)[:,None]; np.put_along_axis(r,order,np.broadcast_to(rr,s.shape),0)
    npos=Y.sum(0); nneg=n-npos
    return ((r*Y).sum(0)-npos*(npos+1)/2)/(np.maximum(npos,1)*np.maximum(nneg,1))
def run_flat(X2d,y,seed=SEED):
    rng=np.random.default_rng(seed); n,F=X2d.shape
    folds=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X2d,y))
    Y=np.empty((n,NPERM+1),np.float32); Y[:,0]=y
    for b in range(1,NPERM+1): Y[:,b]=rng.permutation(y)
    oof=np.zeros((n,NPERM+1),np.float32)
    for tr,te in folds:
        Ytr=Y[tr]; Xtr=X2d[tr]; Xte=X2d[te]; npos=np.maximum(Ytr.sum(0),1); nneg=np.maximum(len(tr)-Ytr.sum(0),1)
        sp=Ytr.T@Xtr; sn=Xtr.sum(0)[None,:]-sp; v=(sp/npos[:,None])-(sn/nneg[:,None])
        oof[te]=Xte@v.T
    A=auc_cols(oof,Y); return float(A[0]),(1+int((A[1:]>=A[0]).sum()))/(NPERM+1)
pl=pd.read_csv(SAVE/'activation_vector_base_results.csv')
def band_of(method,contrast):
    r=pl[(pl.method==method)&(pl.contrast==contrast)]
    if len(r)==0: return []
    r=r.iloc[0]; prof=np.array([float(x) for x in r.auc_profile.split(';')])
    return [l for l in range(33) if prof[l]>=r.null95]
rows=[]; dirs={}
for mname,label in METHODS:
    k=load_k(f"{mname}_ck8")
    if k is None: continue
    km=k.set_index('question_idx'); keep=[q for q in qstar if q in km.index]
    idx=np.array([qpos[q] for q in keep]); sub=np.array([sub_of(km.loc[q,'k_internal'],km.loc[q,'k_external']) for q in keep])
    Xm=Xs[idx]
    contrasts=[(f"{s}_vs_rest",np.ones(len(keep),bool),(sub==s).astype(np.float32)) for s in SUBSETS]
    m2=np.isin(sub,['suppressed','retained']); contrasts.append(("suppressed_vs_retained",m2,(sub[m2]=='suppressed').astype(np.float32)))
    for cname,mask,y in contrasts:
        Xc=Xm[mask]
        if y.sum()<10 or (len(y)-y.sum())<10: continue
        aA,aP=run_flat(Xc.reshape(len(y),-1),y)
        band=band_of(label,cname); nb=len(band); bA=bP=np.nan
        if nb>0: bA,bP=run_flat(Xc[:,band,:].reshape(len(y),-1),y)
        rows.append(dict(method=label,contrast=cname,n=len(y),npos=int(y.sum()),
            all_layers_auc=round(aA,4),all_layers_p=round(aP,4),
            band_layers=nb,band_str=(f"L{min(band)}-L{max(band)}" if band else "none"),
            band_auc=(round(bA,4) if nb>0 else np.nan),band_p=(round(bP,4) if nb>0 else np.nan)))
        print(f"{label:9} {cname:22} all AUC{aA:.3f}/p{aP:.3f} | band{nb:2}L {('AUC%.3f/p%.3f'%(bA,bP)) if nb>0 else 'none'}",flush=True)
    if m2.sum()>=20:
        Xr=Xm[m2]; yr=(sub[m2]=='suppressed')
        dirs[mname]=np.stack([Xr[yr][:,l].mean(0)-Xr[~yr][:,l].mean(0) for l in range(33)]).astype(np.float32)
agg=pd.DataFrame(rows)
def bh(p):
    p=np.asarray(p,float); ok=~np.isnan(p); q=np.full(len(p),np.nan)
    if ok.sum()==0: return q
    pv=p[ok]; o=np.argsort(pv); m=len(pv); qq=pv[o]*m/(np.arange(m)+1); qq=np.minimum.accumulate(qq[::-1])[::-1]
    tmp=np.empty(m); tmp[o]=np.clip(qq,0,1); q[ok]=tmp; return q
agg['all_layers_q']=np.nan; agg['band_q']=np.nan
for c in agg.contrast.unique():
    ii=agg.index[agg.contrast==c]
    agg.loc[ii,'all_layers_q']=bh(agg.loc[ii,'all_layers_p'].values)
    agg.loc[ii,'band_q']=bh(agg.loc[ii,'band_p'].values)
agg.to_csv(SAVE/'activation_vector_aggregate.csv',index=False)
np.savez(SAVE/'suppvret_dir_per_layer.npz',**dirs)
SIG=["PB_J","RepNoise","TAR","GradDiff","ELM","RR"]
def cosmat(layer,names):
    V=np.stack([dirs[n][layer] for n in names]); V=V/np.linalg.norm(V,axis=1,keepdims=True); return V@V.T
mp=np.array([cosmat(l,SIG)[np.triu_indices(len(SIG),1)].mean() for l in range(33)]); bestl=int(mp.argmax())
print("\nMean pairwise cosine (6 sig) per layer:"); print(" ".join(f"L{l}:{mp[l]:+.2f}" for l in range(33)),flush=True)
print(f"Max mean-cosine at L{bestl}={mp[bestl]:.3f}",flush=True)
for l in sorted(set([3,bestl,16])):
    print(f"\nCosine @ L{l} order={SIG}:"); print(np.round(cosmat(l,SIG),2),flush=True)
np.save(SAVE/'cos_meanpair_perlayer.npy',mp)
print("\n=AGG="); print(agg.to_string(index=False),flush=True); print("DONE",flush=True)
