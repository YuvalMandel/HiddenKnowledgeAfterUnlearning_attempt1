import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SAVE='plots/activation_vectors'
pl=pd.read_csv(f'{SAVE}/activation_vector_base_results.csv')
ag=pd.read_csv(f'{SAVE}/activation_vector_aggregate.csv')
dirs=np.load(f'{SAVE}/suppvret_dir_per_layer.npz'); mp=np.load(f'{SAVE}/cos_meanpair_perlayer.npy')
order=["PB&J","RepNoise","TAR","GradDiff","ELM","RR","RMU","RMU-LAT"]; CON="suppressed_vs_retained"
best={r.method:r.best_cv_auc for _,r in pl[pl.contrast==CON].iterrows()}
bestL={r.method:int(r.best_layer) for _,r in pl[pl.contrast==CON].iterrows()}
agc=ag[ag.contrast==CON].set_index('method')
fig,ax=plt.subplots(figsize=(11,5)); x=np.arange(len(order)); w=0.27
b=[best.get(m,np.nan) for m in order]
bd=[agc.loc[m,'band_auc'] if m in agc.index else np.nan for m in order]
al=[agc.loc[m,'all_layers_auc'] if m in agc.index else np.nan for m in order]
ax.bar(x-w,b,w,label='best single layer',color='#4C72B0')
ax.bar(x  ,bd,w,label='band concat',color='#DD8452')
ax.bar(x+w,al,w,label='all 33 layers',color='#55A868')
for i,m in enumerate(order):
    ax.text(x[i]-w,(b[i] or .5)+.004,f"L{bestL.get(m,'')}",ha='center',fontsize=7)
    if m in agc.index and not pd.isna(agc.loc[m,'band_auc']): ax.text(x[i],bd[i]+.004,f"{int(agc.loc[m,'band_layers'])}L",ha='center',fontsize=7)
ax.axhline(0.5,color='k',ls=':',lw=.8); ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(0.5,0.72)
ax.set_ylabel('CV AUC'); ax.set_title('Suppressed-vs-Retained: best layer vs band vs all-33-layers (base model)'); ax.legend()
fig.tight_layout(); fig.savefig(f'{SAVE}/auc_variant_comparison.png',dpi=150,bbox_inches='tight')
SIG=["PB_J","RepNoise","TAR","GradDiff","ELM","RR"]; LAB={"PB_J":"PB&J","RepNoise":"RepNoise","TAR":"TAR","GradDiff":"GradDiff","ELM":"ELM","RR":"RR"}
bestl=int(np.nanargmax(mp))
def cosmat(layer):
    V=np.stack([dirs[n][layer] for n in SIG]); nrm=np.linalg.norm(V,axis=1,keepdims=True); V=V/np.where(nrm==0,1,nrm); return V@V.T
fig,axes=plt.subplots(1,3,figsize=(17,4.8))
for ax,l in zip(axes[:2],[3,bestl]):
    C=cosmat(l); im=ax.imshow(C,cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set_xticks(range(len(SIG))); ax.set_xticklabels([LAB[s] for s in SIG],rotation=40,ha='right')
    ax.set_yticks(range(len(SIG))); ax.set_yticklabels([LAB[s] for s in SIG])
    for i in range(len(SIG)):
        for j in range(len(SIG)): ax.text(j,i,f"{C[i,j]:.2f}",ha='center',va='center',fontsize=8)
    ax.set_title(f'cosine of suppressed dir @ L{l}')
axes[2].plot(range(33),mp,marker='o',ms=3); axes[2].axhline(0,color='k',lw=.7)
axes[2].axvline(bestl,color='r',ls='--',lw=.8); axes[2].set_xlabel('layer'); axes[2].set_ylabel('mean pairwise cosine (6 sig)')
axes[2].set_title('cross-method alignment vs layer'); axes[2].grid(alpha=.3); axes[2].set_xlim(0,32)
fig.colorbar(im,ax=axes[1],shrink=.8); fig.tight_layout(); fig.savefig(f'{SAVE}/cross_method_cosine.png',dpi=150,bbox_inches='tight')
print("SAVED auc_variant_comparison.png cross_method_cosine.png ; max-align L",bestl)
