#!/usr/bin/env python3
"""True/False decisions from the best-layer logit, not just the ranking.

bestlayer_skip.py reports K, a WITHIN-question ranking, which is blind to any
uniform shift of the margin. The model's actual answer is not a ranking: it is
sign(logit_True - logit_False) per claim. This script takes the (L, lambda) each
fold already chose -- read straight out of bestlayer_skip.csv, no new selection,
no new hyperparameter -- and reports what the model would then SAY:

    margin(q,a) = v.h32 + lambda*||v||*(u_L . h^L)      (the skip's logit)
    answer      = True  iff margin > 0
    truth       = True  for the correct claim, False for the 3 distractors

so accuracy is over 4*1273 claims, and TPR/TNR are reported separately because
the classes are 1:3 and a method that answers "False" to everything scores 75%.

That last case is not hypothetical -- TAR says True to 0.0% of claims and scores
exactly the 0.75 prior, GradDiff says True to 99% and scores 0.26. sign(margin)
is therefore an uncalibrated readout, and its accuracy moves mostly with the
yes/no bias. So two threshold-honest numbers are reported beside it:

  auc   -- margin as a score, correct claim vs the 3 distractors. No threshold.
  bacc  -- balanced accuracy at a threshold FITTED ON THAT FOLD'S VAL SET,
           never on test. Chance is 0.5 for both.

The u refit here is the same deterministic fit on the same TRAIN fold, so it
reproduces the run's directions exactly.

Usage: python plots/bestlayer_tf.py [METHOD ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inside_out_knowledge as iok  # noqa: E402
from bestlayer_skip import (AV, CKPT, SLUGS, fit_u, get_v,  # noqa: E402
                            load_layer)


def truth_of(margin, ci, idx):
    t = np.zeros(margin.shape, dtype=bool)
    t[np.arange(len(idx)), ci[idx]] = True
    return t


def best_threshold(margin, truth):
    """Threshold maximising balanced accuracy. Fitted on VAL, applied to TEST."""
    x, y = margin.ravel(), truth.ravel()
    cand = np.quantile(x, np.linspace(0.001, 0.999, 400))
    said = x[None, :] > cand[:, None]
    tpr = (said & y).sum(1) / max(y.sum(), 1)
    tnr = (~said & ~y).sum(1) / max((~y).sum(), 1)
    return float(cand[np.argmax(tpr + tnr)])


def tf_stats(margin, ci, idx, thr=0.0):
    """margin: (n, 4). Correct claim is True, the other three are False."""
    truth = truth_of(margin, ci, idx)
    said = margin > 0
    at = margin > thr
    return dict(acc=float((said == truth).mean()),
                tpr=float(said[truth].mean()),
                tnr=float((~said[~truth]).mean()),
                said_true=float(said.mean()),
                auc=float(roc_auc_score(truth.ravel(), margin.ravel())),
                bacc=float(0.5 * (at[truth].mean() + (~at[~truth]).mean())))


def run(method, rows):
    picks = pd.read_csv(AV / f"bestlayer_skip_{method}.csv")
    picks = picks[picks.labels == "real"].iloc[0]
    Ls = [int(x) for x in str(picks.layers).split("/")]
    lams = [float(x) for x in str(picks.lambdas).split("/")]
    path = iok.OUT_DIR / CKPT[method] / "bio_hs.npy"
    ci = np.load(iok.OUT_DIR / "bio_correct_idx.npy").astype(int)
    nq = np.load(path, mmap_mode="r").shape[0]
    v = get_v(method)
    nv = float(np.linalg.norm(v))
    pv = np.einsum("qod,d->qo", load_layer(path, 32), v)

    splits = list(iok.get_cv_splits(nq))
    m0, m1, qs, t0s, t1s = [], [], [], [], []
    cache = {}
    for (tr, va, te), L, lam in zip(splits, Ls, lams):
        if L not in cache:
            cache = {L: load_layer(path, L)}          # one layer resident
        H = cache[L]
        u = fit_u(H, ci, tr)
        sk = pv + lam * nv * (H @ u)
        # threshold from THIS fold's val, applied to THIS fold's test
        t0s.append(best_threshold(pv[va], truth_of(pv[va], ci, va)))
        t1s.append(best_threshold(sk[va], truth_of(sk[va], ci, va)))
        m0.append(pv[te])
        m1.append(sk[te])
        qs.append(te)
    q = np.concatenate(qs)
    # bacc is averaged over folds because each fold carries its own threshold
    b0 = [tf_stats(m, ci, t, thr) for m, t, thr in zip(m0, [s[2] for s in splits], t0s)]
    b1 = [tf_stats(m, ci, t, thr) for m, t, thr in zip(m1, [s[2] for s in splits], t1s)]
    before = tf_stats(np.concatenate(m0), ci, q)
    after = tf_stats(np.concatenate(m1), ci, q)
    before["bacc"] = float(np.mean([x["bacc"] for x in b0]))
    after["bacc"] = float(np.mean([x["bacc"] for x in b1]))
    rows.append(dict(method=method, layers=picks.layers, lambdas=picks.lambdas,
                     **{f"{k}_before": round(v_, 4) for k, v_ in before.items()},
                     **{f"{k}_after": round(v_, 4) for k, v_ in after.items()},
                     d_acc=round(after["acc"] - before["acc"], 4),
                     d_tpr=round(after["tpr"] - before["tpr"], 4),
                     d_tnr=round(after["tnr"] - before["tnr"], 4),
                     d_auc=round(after["auc"] - before["auc"], 4),
                     d_bacc=round(after["bacc"] - before["bacc"], 4)))
    r = rows[-1]
    print(f"  {method:<10} AUC {r['auc_before']:.4f} -> {r['auc_after']:.4f}"
          f" ({r['d_auc']:+.4f})   bacc {r['bacc_before']:.4f} -> "
          f"{r['bacc_after']:.4f} ({r['d_bacc']:+.4f})   "
          f"acc {r['acc_before']:.3f} -> {r['acc_after']:.3f}   says True "
          f"{r['said_true_before']:.3f} -> {r['said_true_after']:.3f}",
          flush=True)


if __name__ == "__main__":
    ms = sys.argv[1:] or (["base"] + list(SLUGS))
    print("True/False decisions at the (L, lambda) each fold already chose.\n"
          "Claims are 1 true : 3 false, so acc alone is uninformative: read AUC\n"
          "(threshold-free) and bacc (threshold fitted on val). Chance = 0.5.\n")
    rows = []
    for m in ms:
        try:
            run(m, rows)
        except Exception as e:
            print(f"  {m}: FAILED {type(e).__name__}: {e}", flush=True)
    sfx = f"_{ms[0]}" if len(ms) == 1 else ""
    pd.DataFrame(rows).to_csv(AV / f"bestlayer_tf{sfx}.csv", index=False)
    print(f"\nwrote {AV}/bestlayer_tf{sfx}.csv")
