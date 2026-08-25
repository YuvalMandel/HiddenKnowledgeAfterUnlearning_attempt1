# Heatmap: per-layer K_internal, base model vs ck8 (unsteered) vs ck8+steering (alpha=1, the
# pre-registered "full reversal" point -- no alpha sweep, matches causal_recover_pvalue.py's own
# no-cherry-picking rationale). Grid = 8 methods (rows) x {suppressed, forgotten} (cols); each cell
# is its own small heatmap with 3 rows (base/unsteered/steered) x 32 layers (embedding layer 0
# excluded, same convention as L_STEER).
#
# SMART LAYER CHOICE: each method's "steered" row uses whichever of {fixed, topk5, band5} gave the
# best K_ext(d_S) at alpha=1 in causal_recover_<M>[_topk5|_band5].csv (point estimates, see
# tab:layer-targeting) -- not the same generic grid for every method. Fixed already wins for
# RepNoise/RR/PB_J/RMU (their signal peaks early but needs a deeper-reaching grid to survive to the
# output); GradDiff/TAR do best under topk5; ELM/RMU-LAT do best under band5.
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hk_utils import ICLR_TEXTWIDTH_IN, use_iclr_style

use_iclr_style()

SAVE=Path("plots/activation_vectors")
METHODS=["RepNoise","GradDiff","PB_J","TAR","RR","ELM","RMU","RMU-LAT"]  # significant recoverers first, then not
# Split into two figures: 8x2 panels at ICLR's 5.5in column forced a 0.43x
# downscale (3pt text). Four methods per figure keeps every panel at true size.
# The split point is the paper's own significance boundary (Table
# tab:causal-recovery), not an arbitrary halving.
GROUPS=[("sig",   METHODS[:4], "significant causal recovery"),
        ("nonsig",METHODS[4:], "no significant causal recovery")]
LAYERS=list(range(1,33))
ROW_LABELS=["Base","Unsteered","Steered\n"+r"$\alpha{=}1$"]

# per-method best layer-selection mode, from tab:layer-targeting K_ext(d_S)@alpha=1 point estimates
BEST_MODE={"RepNoise":"fixed","GradDiff":"topk5","PB_J":"fixed","TAR":"topk5",
           "RR":"fixed","ELM":"band5","RMU":"fixed","RMU-LAT":"band5"}
SUFFIX={"fixed":"","topk5":"_topk5","band5":"_band5"}
L_STEER_FIXED=[3,6,9,12,15]

def select_layers(method,mode,k=5):
    if mode=="fixed": return L_STEER_FIXED
    prof=pd.read_csv(SAVE/"activation_vector_base_results.csv")
    csv_method={"PB_J":"PB&J"}.get(method,method)
    row=prof[(prof.method==csv_method)&(prof.contrast=="suppressed_vs_retained")].iloc[0]
    auc=np.array([float(x) for x in row.auc_profile.split(";")])
    if mode=="topk": return sorted(sorted(range(1,33),key=lambda L:-auc[L])[:k])
    if mode=="band":
        c=int(row.best_layer); half=k//2; s,e=c-half,c-half+k-1
        if s<1: e+=1-s; s=1
        if e>32: s-=e-32; e=32
        return list(range(s,e+1))

def method_panel(method,qset):
    mode=BEST_MODE[method]; suf=SUFFIX[mode]
    df=pd.read_csv(SAVE/("causal_recover_perlayer_%s%s.csv"%(method,suf)))
    base_cond = "base_model" if qset=="supp" else "base_model_forg"
    steer_cond = "supp_dS" if qset=="supp" else "forg_dF"
    M=np.full((3,len(LAYERS)),np.nan)
    conds=[(base_cond,None),(steer_cond,0.0),(steer_cond,1.0)]
    for r,(cond,alpha) in enumerate(conds):
        sub=df[df.condition==cond]
        sub=sub[sub.alpha==alpha] if alpha is not None else sub
        sub=sub.set_index("layer")
        for c,L in enumerate(LAYERS):
            if L in sub.index: M[r,c]=sub.loc[L,"k_internal"]
    return M

norm=TwoSlopeNorm(vmin=0.0,vcenter=0.5,vmax=1.0)

def build(tag,methods,blurb):
    # RdBu stays: the scale is diverging about chance (0.5), so a sequential map
    # would lose the above/below-chance reading. It is also the colour-blind-safe
    # diverging choice.
    #
    # Greyscale: any diverging map is luminance-symmetric, so 0.0 and 1.0 print
    # as the same grey and colour alone cannot say which side of chance a cell
    # is on. Rather than change the encoding, we add a REDUNDANT non-colour
    # channel: a contour drawn at K_int=0.5. In colour it marks the chance
    # boundary explicitly; in greyscale it is the only thing needed to read the
    # figure, since every cell is then identifiable as inside or outside the
    # above-chance region. This satisfies the ICLR black/white caveat without
    # sacrificing the diverging scale.
    fig,axes=plt.subplots(len(methods),2,
                          figsize=(ICLR_TEXTWIDTH_IN,ICLR_TEXTWIDTH_IN*0.86),
                          sharex=True)
    im=None
    for i,method in enumerate(methods):
        l_steer=select_layers(method,{"fixed":"fixed","topk5":"topk","band5":"band"}[BEST_MODE[method]])
        for j,(qset,qlabel) in enumerate([("supp","Suppressed Q"),("forg","Forgotten Q")]):
            ax=axes[i,j]
            M=method_panel(method,qset)
            im=ax.imshow(M,aspect="auto",cmap="RdBu",norm=norm,
                          extent=[LAYERS[0]-.5,LAYERS[-1]+.5,3-.5,-.5])
            # Redundant greyscale channel: outline the chance level. A diverging
            # map prints 0.0 and 1.0 as the same grey, so colour alone cannot say
            # which side of chance a cell is on; the contour marks the boundary
            # in a channel that survives black-and-white printing.
            # (A hatched below-chance overlay was tried and rejected: at 32x3
            # cells the hatch either renders invisibly or swamps the panel.)
            if np.isfinite(M).any() and np.nanmin(M)<0.5<np.nanmax(M):
                ax.contour(np.arange(LAYERS[0],LAYERS[-1]+1),np.arange(3),
                           np.nan_to_num(M,nan=0.5),levels=[0.5],
                           colors="k",linewidths=0.7)
            for L in l_steer: ax.axvline(L,color="k",lw=.6,ls=":",alpha=.6)
            ax.set_yticks([0,1,2])
            ax.set_yticklabels(ROW_LABELS if j==0 else [],fontsize=6)
            if j==0:
                label="%s\n(%s)"%({"PB_J":"PB&J"}.get(method,method),BEST_MODE[method])
                ax.set_ylabel(label,fontsize=7,fontweight="bold",rotation=0,ha="right",va="center",labelpad=6)
            if i==0: ax.set_title(qlabel,fontsize=8)
            ax.set_xticks(range(0,33,4))
            ax.tick_params(axis="x",labelsize=7)
    for j in (0,1):
        axes[-1,j].set_xlabel("layer (dotted = best injection layers)",fontsize=7)
    fig.colorbar(im,ax=axes,shrink=0.6,
                 label="$K_\\mathrm{int}$ (frozen-probe AUC; 0.5=chance)")
    fig.suptitle("Per-layer $K_\\mathrm{int}$ — methods with %s"%blurb,fontsize=8)
    stem=SAVE/("heatmap_perlayer_hk_%s"%tag)
    fig.savefig("%s.png"%stem,dpi=150,bbox_inches="tight")
    fig.savefig("%s.pdf"%stem,bbox_inches="tight")
    plt.close(fig)
    print("SAVED %s.pdf (%d methods)"%(stem,len(methods)))

for tag,methods,blurb in GROUPS:
    build(tag,methods,blurb)
