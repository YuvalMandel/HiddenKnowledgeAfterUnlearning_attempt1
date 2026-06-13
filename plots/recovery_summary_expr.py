# Combine per-method within-model (r) vs cross-model (d_S) recovery into one panel.
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
SAVE=Path("plots/activation_vectors")
METHODS=["RepNoise","PB_J","RR","GradDiff","RMU","RMU-LAT","ELM","TAR"]
rows=[]
for m in METHODS:
    p=SAVE/("causal_recover_expr_%s.csv"%m)
    if not p.exists(): print("MISSING",m); continue
    d=pd.read_csv(p)
    def g(c,a):
        s=d[(d.condition==c)&(np.isclose(d.alpha,a))]; return float(s.kext.iloc[0]) if len(s) else np.nan
    rows.append(dict(method=m,base=g("supp_expr",0.0),expr1=g("supp_expr",1.0),expr2=g("supp_expr",2.0),
        dS1=g("supp_dS",1.0),dS2=g("supp_dS",2.0),rand1=g("supp_random",1.0),forg1=g("forg_expr",1.0)))
df=pd.DataFrame(rows); df.to_csv(SAVE/"recovery_summary_expr.csv",index=False)
print(df.round(3).to_string(index=False))
x=np.arange(len(df)); w=0.2
fig,ax=plt.subplots(figsize=(12,5))
ax.bar(x-1.5*w,df.expr2,w,label="within-model r @a=2",color="#d62728")
ax.bar(x-0.5*w,df.dS2,w,label="cross-model d_S @a=2",color="#1f77b4")
ax.bar(x+0.5*w,df.forg1,w,label="forgotten (r) @a=1",color="#7b3294")
ax.bar(x+1.5*w,df.rand1,w,label="random @a=1",color="#888888")
ax.scatter(x-1.5*w,df.base,color="k",s=18,zorder=5,label="baseline")
ax.axhline(0.5,color="k",ls=":",lw=.6)
ax.set_xticks(x); ax.set_xticklabels(df.method,rotation=20); ax.set_ylabel("K_ext (held-out)"); ax.set_ylim(0,0.8)
ax.set_title("Recovery per method: within-model expression r vs cross-model d_S")
ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(SAVE/"recovery_summary_expr.png",dpi=150,bbox_inches="tight")
print("SAVED recovery_summary_expr.{csv,png}")
