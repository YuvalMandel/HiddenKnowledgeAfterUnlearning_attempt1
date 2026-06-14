# Aggregate the lambda-sweep per-gamma CSVs into one table + plot. Usage: python agg_transform_lambda.py <METHOD>
# Confirms: ridge K_ext climbs toward (but does not exceed) the translation (gamma=inf) as gamma grows;
# small gamma (~OLS) overfits and underperforms.
import sys, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
METHOD=sys.argv[1] if len(sys.argv)>1 else "RepNoise"
SAVE=Path("plots")/"activation_vectors"
files=sorted(glob.glob(str(SAVE/("transform_lambda_%s_g*.csv"%METHOD))))
if not files:
    print("no files for",METHOD); sys.exit(0)
df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
# split finite gammas vs inf
df["is_inf"]=df.gamma.astype(str).str.lower().isin(["inf","infinity"])
fin=df[~df.is_inf].copy(); fin["g"]=fin.gamma.astype(float); fin=fin.sort_values("g")
inf_row=df[df.is_inf]
ref=float(inf_row.kext.iloc[0]) if len(inf_row) else np.nan
df_out=pd.concat([fin[["method","gamma","kext"]],inf_row[["method","gamma","kext"]]],ignore_index=True)
df_out.to_csv(SAVE/("transform_lambda_%s.csv"%METHOD),index=False)
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(fin.g,fin.kext,"-o",color="#1f77b4",label="ridge (beta=1)")
if not np.isnan(ref): ax.axhline(ref,color="k",ls="--",lw=1,label="translation (gamma=inf) %.3f"%ref)
ax.axhline(0.5,color="k",lw=.5,ls=":")
ax.set_xscale("log"); ax.set_ylim(0,1)
ax.set_xlabel("ridge gamma (toward identity)  -- larger = closer to translation")
ax.set_ylabel("K_ext (held-out suppressed)")
ax.set_title("%s: ridge lambda sweep -> climbs toward translation"%METHOD)
ax.legend(fontsize=9); ax.grid(alpha=.3,which="both")
fig.tight_layout(); fig.savefig(SAVE/("transform_lambda_%s.png"%METHOD),dpi=150,bbox_inches="tight")
print(df_out.to_string(index=False)); print("SAVED",METHOD,flush=True)
