# Correlate per-method Hidden-Knowledge gap with recoverability.
# HK gap = mean(K_int - K_ext) over Q* for the ck8 model.
# recoverability = directed recovery = supp recover@best - random  (cross-model d_S and within-model r).
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
OUT=Path("inside_out_out"); SAVE=Path("plots/activation_vectors")
METHODS=["GradDiff","RMU","RMU-LAT","RepNoise","ELM","RR","TAR","PB_J"]
def load_k(mid):
    df=pd.read_parquet(OUT/mid/"k_scores.parquet")
    df=df[(df.split_type=="cv")&(df.domain=="bio")&(df.clf=="LR")&(df.probe_type=="own")&(df.layer_config=="best_layer")]
    return df[["question_idx","k_internal","k_external"]].groupby("question_idx",as_index=False).mean()
base=load_k("base"); qstar=set(base.loc[(base.k_internal==1)&(base.k_external==1),"question_idx"].astype(int))
dS=pd.read_csv(SAVE/"recovery_summary_dS.csv").set_index("method")
ex=pd.read_csv(SAVE/"recovery_summary_expr.csv").set_index("method")
rows=[]
for m in METHODS:
    k=load_k(m+"_ck8"); kq=k[k.question_idx.isin(qstar)]
    hk=float(kq.k_internal.mean()-kq.k_external.mean())
    rows.append(dict(method=m,HKgap=round(hk,3),Kint=round(float(kq.k_internal.mean()),3),Kext=round(float(kq.k_external.mean()),3),
        dS_recover=round(float(dS.loc[m,"supp_dS"]),3), dS_directed=round(float(dS.loc[m,"supp_dS"]-dS.loc[m,"random"]),3),
        r_recover=round(float(ex.loc[m,"expr2"]),3), r_directed=round(float(ex.loc[m,"expr2"]-ex.loc[m,"rand1"]),3)))
df=pd.DataFrame(rows); df.to_csv(SAVE/"hk_vs_recovery.csv",index=False)
print(df.to_string(index=False))
pr,pp=pearsonr(df.HKgap,df.dS_directed); sr,sp=spearmanr(df.HKgap,df.dS_directed)
prr,ppr=pearsonr(df.HKgap,df.r_directed)
print("d_S directed vs HKgap: Pearson r=%.3f p=%.3f ; Spearman rho=%.3f p=%.3f"%(pr,pp,sr,sp))
print("r   directed vs HKgap: Pearson r=%.3f p=%.3f"%(prr,ppr))
fig,ax=plt.subplots(figsize=(8,6))
ax.scatter(df.HKgap,df.dS_directed,s=70,color="#1f77b4",label="cross-model d_S")
ax.scatter(df.HKgap,df.r_directed,s=70,color="#d62728",marker="^",label="within-model r")
for _,r in df.iterrows():
    ax.annotate(r.method,(r.HKgap,r.dS_directed),fontsize=8,xytext=(4,3),textcoords="offset points")
    ax.annotate(r.method,(r.HKgap,r.r_directed),fontsize=7,color="#d62728",xytext=(4,-9),textcoords="offset points")
ax.axhline(0,color="k",lw=.5); ax.grid(alpha=.3); ax.legend()
ax.set_xlabel("HK gap = mean(K_int - K_ext) over Q*  (ck8)")
ax.set_ylabel("directed recoverability (recover@best - random)")
ax.set_title("Recoverability vs Hidden-Knowledge gap\nd_S: Pearson r=%.2f (p=%.2f), Spearman rho=%.2f"%(pr,pp,sr))
fig.tight_layout(); fig.savefig(SAVE/"hk_vs_recovery.png",dpi=150,bbox_inches="tight")
print("SAVED hk_vs_recovery.{csv,png}")
