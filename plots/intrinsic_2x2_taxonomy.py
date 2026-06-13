# INTRINSIC 4-category taxonomy: classify ALL bio questions by the UNLEARNED ck8 model's own
# (K_int, K_ext) 2x2 -- NO base-model prefilter. Per method.
#   know_both     : K_int>0.5 & K_ext>0.5   (knows internally AND externally)
#   internal_only : K_int>0.5 & K_ext<=0.5  (knows internally, NOT externally) <- hidden
#   external_only : K_int<=0.5 & K_ext>0.5  (not internally, but externally)   <- "lucky"
#   unknown       : K_int<=0.5 & K_ext<=0.5 (neither)
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path("inside_out_out"); SAVE=Path("plots/activation_vectors")
METHODS=["GradDiff","RMU","RMU-LAT","RepNoise","ELM","RR","TAR","PB_J"]
def load_k(mid):
    df=pd.read_parquet(OUT/mid/"k_scores.parquet")
    df=df[(df.split_type=="cv")&(df.domain=="bio")&(df.clf=="LR")&(df.probe_type=="own")&(df.layer_config=="best_layer")]
    return df[["question_idx","k_internal","k_external"]].groupby("question_idx",as_index=False).mean()
rows=[]
for m in METHODS:
    k=load_k(m+"_ck8"); ki=k.k_internal>0.5; ke=k.k_external>0.5; n=len(k)
    rows.append(dict(method=m,n=int(n),
        know_both=int((ki&ke).sum()), internal_only=int((ki&~ke).sum()),
        external_only=int((~ki&ke).sum()), unknown=int((~ki&~ke).sum())))
df=pd.DataFrame(rows)
for c in ["know_both","internal_only","external_only","unknown"]:
    df[c+"_pct"]=(df[c]/df.n*100).round(1)
df.to_csv(SAVE/"intrinsic_2x2_taxonomy.csv",index=False)
print(df[["method","n","know_both","internal_only","external_only","unknown"]].to_string(index=False))
print()
print(df[["method","know_both_pct","internal_only_pct","external_only_pct","unknown_pct"]].to_string(index=False))
# stacked bar (percentages)
cats=["know_both","internal_only","external_only","unknown"]
labs=["knows both (int+ext)","internal only (hidden)","external only (lucky)","unknown (neither)"]
cols=["#2ca02c","#d62728","#1f77b4","#888888"]
fig,ax=plt.subplots(figsize=(11,5)); x=np.arange(len(df)); bottom=np.zeros(len(df))
for c,l,co in zip(cats,labs,cols):
    vals=df[c]/df.n*100; ax.bar(x,vals,bottom=bottom,label=l,color=co); bottom+=vals.values
ax.set_xticks(x); ax.set_xticklabels(df.method,rotation=20); ax.set_ylabel("% of all WMDP-bio questions")
ax.set_title("Intrinsic (K_int,K_ext) 2x2 taxonomy of the UNLEARNED model -- no base prefilter")
ax.legend(fontsize=8,ncol=2,loc="lower right"); fig.tight_layout()
fig.savefig(SAVE/"intrinsic_2x2_taxonomy.png",dpi=150,bbox_inches="tight")
print("SAVED intrinsic_2x2_taxonomy.{csv,png}")
