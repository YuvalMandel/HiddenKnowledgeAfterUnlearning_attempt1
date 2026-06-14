# Combine per-method cross-model d_S recovery results into one summary panel.
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
SAVE=Path("plots/activation_vectors")
METHODS=["GradDiff","RMU","RMU-LAT","RepNoise","ELM","RR","TAR","PB_J"]
rows=[]
for m in METHODS:
    p=SAVE/("causal_recover_%s.csv"%m)
    if not p.exists(): print("MISSING",m); continue
    d=pd.read_csv(p)
    def g(cond,a):
        s=d[(d.condition==cond)&(np.isclose(d.alpha,a))]; return float(s.kext.iloc[0]) if len(s) else np.nan
    rows.append((m,g("supp_dS",0.0),g("supp_dS",1.0),g("supp_random",1.0),g("forg_dF",1.0)))
df=pd.DataFrame(rows,columns=["method","base","supp_dS","random","forg"]).sort_values("supp_dS",ascending=False)
df["specificity"]=df.supp_dS-df["random"]
df.to_csv(SAVE/"recovery_summary_dS.csv",index=False)
print(df.round(3).to_string(index=False))
x=np.arange(len(df)); w=0.25
fig,ax=plt.subplots(figsize=(11,5))
ax.bar(x-w,df.supp_dS,w,label="supp_dS (recover) @a=1",color="#d62728")
ax.bar(x,  df.forg,  w,label="forg_dF @a=1",color="#7b3294")
ax.bar(x+w,df["random"],w,label="random @a=1",color="#888888")
ax.scatter(x-w,df.base,color="k",zorder=5,s=22,label="baseline supp (a=0)")
ax.axhline(0.5,color="k",ls=":",lw=.6)
ax.set_xticks(x); ax.set_xticklabels(df.method,rotation=20); ax.set_ylabel("K_ext (held-out)"); ax.set_ylim(0,0.8)
ax.set_title("Cross-model d_S recovery per method (alpha=1): recover vs forgotten vs random")
ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(SAVE/"recovery_summary_dS.png",dpi=150,bbox_inches="tight")
print("SAVED recovery_summary_dS.png + .csv")
