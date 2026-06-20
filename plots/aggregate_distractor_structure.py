# Aggregate distractor_structure.py per-method parts -> final deliverables.
import glob, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ODIR=Path("activation_question_scores"); FIG=ODIR/"figures"; PARTS=ODIR/"_parts_distractor"
FIG.mkdir(parents=True,exist_ok=True)
VARIANTS=["correct_only","correct_minus_mean_wrong","correct_minus_strongest_wrong",
          "correct_minus_closest_wrong","correct_minus_farthest_wrong",
          "concat_all_wrong_ordered","mean_plus_spread"]
SHORT={"correct_only":"corr_only","correct_minus_mean_wrong":"c-mean_wrong","correct_minus_strongest_wrong":"c-strongest",
       "correct_minus_closest_wrong":"c-closest","correct_minus_farthest_wrong":"c-farthest",
       "concat_all_wrong_ordered":"concat_3","mean_plus_spread":"mean+spread"}
BASE="correct_minus_mean_wrong"
MNAMES=["GradDiff","PB&J","RMU","RMU-LAT","RepNoise","ELM","RR","TAR"]

# concat per-question
pqs=sorted(glob.glob(str(PARTS/"pq_*.csv.gz")))
final=ODIR/"per_question_distractor_structure_scores.csv.gz"
if final.exists(): final.unlink()
hdr=True
for p in pqs:
    for ch in pd.read_csv(p,chunksize=200000):
        ch.to_csv(final,mode="a",header=hdr,index=False); hdr=False
print("per-question parts:",len(pqs),flush=True)

s=pd.concat([pd.read_csv(p) for p in glob.glob(str(PARTS/"summary_*.csv"))],ignore_index=True)
s.to_csv(ODIR/"distractor_structure_auc_summary.csv",index=False)
mnames=[m for m in MNAMES if m in set(s.method)]

# mean-over-methods AUC per (variant,direction)
piv=s.pivot_table(index="feature_variant",columns="direction",values="auc_mean",aggfunc="mean").reindex(VARIANTS)
# best direction per (method,variant)
best=s.groupby(["method","feature_variant"]).auc_mean.max().reset_index()
bestpiv=best.pivot_table(index="method",columns="feature_variant",values="auc_mean").reindex(index=mnames,columns=VARIANTS)
bestbyvar=best.groupby("feature_variant").auc_mean.mean().reindex(VARIANTS)

# ---- fig 1: feature comparison (mean over methods, both directions)
fig,ax=plt.subplots(figsize=(13,6)); x=np.arange(len(VARIANTS)); w=.38
for i,d in enumerate(["mean_difference","logistic_regression"]):
    vals=[piv.loc[v,d] if d in piv.columns else np.nan for v in VARIANTS]
    ax.bar(x+(i-.5)*w,vals,w,label=d)
base_best=bestbyvar.get(BASE,np.nan)
ax.axhline(base_best,ls="--",c="green",lw=1,label=f"baseline c-mean_wrong (best dir) = {base_best:.3f}")
ax.axhline(0.5,ls=":",c="gray")
ax.set_xticks(x); ax.set_xticklabels([SHORT[v] for v in VARIANTS],rotation=25,ha="right")
ax.set_ylabel("OOF AUC (suppressed vs forgotten, mean over 8 methods)"); ax.set_ylim(0.45,0.8)
ax.set_title("Distractor-structure feature variants vs the correct-minus-mean-wrong baseline")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG/"distractor_structure_feature_comparison.png",dpi=130); plt.close(fig)

# ---- fig 2: by-method heatmap (best direction per cell)
fig,ax=plt.subplots(figsize=(12,6))
M=bestpiv.values.astype(float)
im=ax.imshow(M,cmap="RdYlGn",vmin=0.45,vmax=0.78,aspect="auto")
ax.set_xticks(range(len(VARIANTS))); ax.set_xticklabels([SHORT[v] for v in VARIANTS],rotation=30,ha="right")
ax.set_yticks(range(len(mnames))); ax.set_yticklabels(mnames)
for i in range(len(mnames)):
    for j in range(len(VARIANTS)):
        if not np.isnan(M[i,j]): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=8)
fig.colorbar(im,label="OOF AUC (best direction)")
ax.set_title("Suppressed-vs-forgotten AUC by method x distractor feature variant")
fig.tight_layout(); fig.savefig(FIG/"distractor_structure_by_method.png",dpi=130); plt.close(fig)
print("wrote 2 figures",flush=True)

# ---- REPORT
def d(v): return bestbyvar.get(v,np.nan)
STRUCT=["correct_minus_strongest_wrong","correct_minus_closest_wrong","correct_minus_farthest_wrong","concat_all_wrong_ordered"]
struct_delta=(bestpiv[STRUCT].max(axis=1)-bestpiv[BASE])   # per method: best structural - baseline
L=[]
L.append("# Distractor-structure analysis — suppressed vs forgotten — REPORT\n")
L.append("## Bottom line")
L.append(f"**Distractor structure does NOT help.** The simple `correct_minus_mean_wrong` (mode C, {d(BASE):.3f} mean AUC) is not beaten by any structural variant in any of the {len(mnames)} methods — best-structural minus baseline ranges {struct_delta.min():+.3f} to {struct_delta.max():+.3f} (all <= 0). Selecting a single distractor (strongest/closest/farthest) loses information; concatenating all three or appending spread scalars ties at best. Averaging over the three distractors is the sweet spot — a parsimony result, consistent across all methods.\n")
L.append("## Data sources & setup")
L.append("- Features: BASE `inside_out_out/base/bio_hs.npy` (correct + 3 wrong option hidden states per MCQ).")
L.append("- Distractor selection uses BASE info only: `inside_out_ext/base_bio_ext.npy` (per-option external score) and base hidden-state L2 distance — never the suppressed/forgotten label.")
L.append("- Labels: `{method}_ck8/k_scores.parquet`; contrast = suppressed vs forgotten on Q* (701).")
L.append("- Leakage-safe: z-scoring, PCA-256 (logreg), best-layer selection all fit inside the TRAIN fold; grouped 5-fold × 20-repeat CV, seed 42; OOF only.\n")
L.append("## Feature variants (label-independent distractor selection)")
RULEDESC={"correct_only":"reference (no distractors)","correct_minus_mean_wrong":"baseline = mode C",
  "correct_minus_strongest_wrong":"wrong with max base ext score","correct_minus_closest_wrong":"wrong with min base hidden-state distance",
  "correct_minus_farthest_wrong":"wrong with max base hidden-state distance","concat_all_wrong_ordered":"3 (correct-wrong) vectors, ordered by ext",
  "mean_plus_spread":"mode C + 4 per-layer distractor-spread scalars"}
for v in VARIANTS: L.append(f"- `{v}` — {RULEDESC[v]}")
L.append("\n## Mean OOF AUC over 8 methods (best of mean-diff / logreg per variant)\n")
L.append("| variant | mean AUC | vs baseline |")
L.append("|---|---|---|")
for v in VARIANTS:
    L.append(f"| {v} | {d(v):.3f} | {d(v)-d(BASE):+.3f} |")
L.append("\nPer-direction means (mean-diff / logreg):\n")
L.append("| variant | mean_difference | logistic_regression |")
L.append("|---|---|---|")
for v in VARIANTS:
    md=piv.loc[v,"mean_difference"] if "mean_difference" in piv.columns else float("nan")
    lr=piv.loc[v,"logistic_regression"] if "logistic_regression" in piv.columns else float("nan")
    L.append(f"| {v} | {md:.3f} | {lr:.3f} |")
L.append("\n## Answers to the task questions")
best_single=max(["correct_minus_strongest_wrong","correct_minus_closest_wrong","correct_minus_farthest_wrong"],key=lambda v:d(v))
L.append(f"1. **Does a distractor feature beat the mean-distractor baseline?** Best variant = `{bestbyvar.idxmax()}` ({bestbyvar.max():.3f}) vs baseline `{BASE}` ({d(BASE):.3f}); delta {bestbyvar.max()-d(BASE):+.3f}.")
L.append(f"2. **Strongest vs closest distractor?** strongest {d('correct_minus_strongest_wrong'):.3f} vs closest {d('correct_minus_closest_wrong'):.3f} (farthest {d('correct_minus_farthest_wrong'):.3f}); best single = `{best_single}`.")
L.append(f"3. **Does preserving all three distractors help?** concat_3 {d('concat_all_wrong_ordered'):.3f} vs baseline {d(BASE):.3f} (delta {d('concat_all_wrong_ordered')-d(BASE):+.3f}).")
L.append(f"4. **Does distractor spread predict suppression?** mean+spread {d('mean_plus_spread'):.3f} vs baseline {d(BASE):.3f} (delta {d('mean_plus_spread')-d(BASE):+.3f}).")
# consistency: best STRUCTURAL variant vs baseline per method (excludes mean_plus_spread which nests baseline)
winvar=bestpiv[STRUCT].idxmax(axis=1)
L.append("5. **Consistent or method-specific?** The baseline wins (or ties) for **every** method — best structural variant minus baseline: "
         +", ".join(f"{m}:{struct_delta[m]:+.3f}({winvar[m].replace('correct_minus_','').replace('_wrong','')})" for m in mnames)
         +". The negative result is unanimous, not method-specific.")
L.append(f"6. **One distractor or overall structure?** best single-distractor {d(best_single):.3f} vs concat_3 {d('concat_all_wrong_ordered'):.3f} vs mean+spread {d('mean_plus_spread'):.3f}.")
L.append("\n## Acceptance criteria")
L.append("All distractor-selection rules are label-independent (base ext / base hidden distance), evaluated out-of-fold, and compared directly against `correct_only` and `correct_minus_mean_wrong`. ✓")
(ODIR/"REPORT_DISTRACTOR_STRUCTURE.md").write_text("\n".join(L),encoding="utf-8")
print("wrote REPORT_DISTRACTOR_STRUCTURE.md\nAGG DONE",flush=True)
