import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SAVE='plots/activation_vectors'; d=pd.read_csv(f'{SAVE}/activation_vector_base_results.csv')
order=["PB&J","RepNoise","TAR","GradDiff","ELM","RR","RMU","RMU-LAT"]
cmap=plt.cm.tab10; colors={m:cmap(i) for i,m in enumerate(order)}
cons=["retained_vs_rest","forgotten_vs_rest","lucky_vs_rest"]
def prof(r): return np.array([float(x) for x in r.auc_profile.split(';')])
for con in cons:
    print(f"\n== {con} ==")
    sub=d[d.contrast==con].set_index('method')
    for m in order:
        r=sub.loc[m]; p=prof(r); ab=[l for l in range(33) if p[l]>=r.null95]
        band=f"L{min(ab)}-L{max(ab)}" if ab else "none"
        print(f"{m:8} {'SIG' if r.perm_q<.05 else 'ns ':3} peakL{int(p.argmax()):2}({p.max():.3f}) q={r.perm_q:.3f} band {band} ({len(ab)}L)")
fig,axes=plt.subplots(1,3,figsize=(18,5),sharey=True)
for ax,con in zip(axes,cons):
    sub=d[d.contrast==con].set_index('method'); thrs=[]
    for m in order:
        r=sub.loc[m]; p=prof(r); sig=r.perm_q<.05; thrs.append(r.null95)
        ax.plot(range(33),p,color=colors[m],lw=2.2 if sig else 1.3,ls='-' if sig else '--',alpha=.95 if sig else .45,label=m+('' if sig else ' (n.s.)'))
        bl=int(p.argmax()); ax.scatter([bl],[p[bl]],color=colors[m],s=30,zorder=6,edgecolor='white',lw=.6)
    ax.axhspan(0.5,float(np.median(thrs)),color='0.85',alpha=.5); ax.axhline(.5,color='k',lw=.8,ls=':')
    ax.set_title(con,fontweight='bold'); ax.set_xlabel('layer'); ax.set_xlim(0,32); ax.grid(alpha=.2)
axes[0].set_ylabel('CV AUC (base diff-of-means direction)'); axes[0].legend(fontsize=8,ncol=2)
fig.suptitle('Base-model activation direction — per-layer separability: retained / forgotten / lucky',y=1.02,fontsize=12)
fig.tight_layout(); fig.savefig(f'{SAVE}/other3_layer_profiles.png',dpi=150,bbox_inches='tight'); print("\nSAVED other3_layer_profiles.png")
