#!/usr/bin/env python3
"""Sweep lambda from the cached projections -- seconds, no cluster, no refit.

lambda never enters a probe fit, so once bestlayer_skip.py has written
`bestlayer_proj_<METHOD>.npz` (one (1273,4) score array per tag/fold/layer, 19 MB
for all nine checkpoints) every question about the lambda grid is arithmetic on
those arrays. This runs locally.

Produces, per method:
  * the full test-K curve over lambda, L selected on val at each lambda
  * the lambda=1e6 operating point on its own -- the probe REPLACING the model's
    readout rather than adding to it, which is what the plot needs as its
    right-hand anchor
  * a reproduction check against bestlayer_skip.csv

Usage: python plots/bestlayer_lambda_sweep.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plots"))
import inside_out_knowledge as iok  # noqa: E402  (get_cv_splits only)

AV = ROOT / "plots" / "activation_vectors"
ORDER = ["base", "GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR",
         "PB_J"]
GRID = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0,
        1024.0, 2048.0, 4096.0, 16384.0, 65536.0, 1e6]
RETAIN = 0.95
LAST = 32


def k_per_q(s, ci, idx):
    return np.array([np.mean([s[i, ci[q]] > s[i, j]
                              for j in range(4) if j != ci[q]])
                     for i, q in enumerate(idx)])


def load_ci():
    # iok.ROOT is hard-coded to the Newton path, so read the labels relatively.
    return np.load(ROOT / "inside_out_out" / "bio_correct_idx.npy").astype(int)


def main():
    ci = load_ci()
    splits = list(iok.get_cv_splits(len(ci)))
    prev = pd.read_csv(AV / "bestlayer_skip.csv")
    prev = prev[prev.labels == "real"].set_index("method")
    rows, curves = [], []
    for m in ORDER:
        proj = AV / f"bestlayer_proj_{m}.npz"
        if not proj.exists():
            print(f"  {m}: no projection cache, skipped")
            continue
        z = np.load(proj)
        layers = z["layers"].tolist()
        nv = float(np.linalg.norm(np.load(AV / f"v_{m}.npz")["v"]))
        pv = z[f"lens_{LAST}"]
        pu = {(fi, L): z[f"pu_real_{fi}_{L}"]
              for fi in range(len(splits)) for L in layers}

        # (a) test K at each fixed lambda, L chosen on val. Two arms: with the
        # retention constraint (the reported one) and without it. At lam=1e6 the
        # constrained arm is DEGENERATE -- replacing the readout by the probe
        # always breaks more than 5% of the already-correct questions, so no
        # layer qualifies and the fallback is K_ext itself. That is why lam=1e6
        # stopped being selected once the grid had usable values in between.
        for lam in GRID:
            k1s, k2s, blocked = [], [], 0
            for fi, (tr, va, te) in enumerate(splits):
                k0v = k_per_q(pv[va], ci, va)
                keep = k0v == 1.0
                bestL, best = None, -1.0
                fL, fbest = layers[0], -1.0
                for L in layers:
                    kk = k_per_q(pv[va] + lam * nv * pu[fi, L][va], ci, va)
                    m_ = kk.mean()
                    if m_ > fbest:
                        fbest, fL = m_, L
                    if keep.any() and kk[keep].mean() < RETAIN:
                        continue
                    if m_ > best:
                        best, bestL = m_, L
                k2s.append(k_per_q(pv[te] + lam * nv * pu[fi, fL][te], ci, te))
                if bestL is None:            # no layer keeps the constraint
                    blocked += 1
                    k1s.append(k_per_q(pv[te], ci, te))
                    continue
                k1s.append(k_per_q(pv[te] + lam * nv * pu[fi, bestL][te],
                                   ci, te))
            curves.append(dict(method=m, lam=lam,
                               k=round(float(np.concatenate(k1s).mean()), 4),
                               k_free=round(float(np.concatenate(k2s).mean()), 4),
                               blocked_folds=blocked))

        k_ext = float(prev.loc[m, "k_ext"])
        c = pd.DataFrame([r for r in curves if r["method"] == m])
        at1e6 = float(c[c.lam == 1e6].k.iloc[0])
        at1e6f = float(c[c.lam == 1e6].k_free.iloc[0])
        best_row = c.loc[c.k.idxmax()]
        rows.append(dict(method=m, k_ext=k_ext,
                         k_lam1e6=at1e6, gain_lam1e6=round(at1e6 - k_ext, 4),
                         k_lam1e6_free=at1e6f,
                         gain_lam1e6_free=round(at1e6f - k_ext, 4),
                         blocked_at_1e6=int(c[c.lam == 1e6]
                                            .blocked_folds.iloc[0]),
                         best_lam=best_row.lam, k_best=best_row.k,
                         gain_best=round(best_row.k - k_ext, 4),
                         reported=float(prev.loc[m, "skipped"])))
        r = rows[-1]
        print(f"  {m:<9} K_ext {k_ext:.4f} | lam=1e6 constrained {at1e6:.4f}"
              f" ({r['blocked_at_1e6']}/5 folds blocked) | lam=1e6 free "
              f"{at1e6f:.4f} ({r['gain_lam1e6_free']:+.4f}) | reported "
              f"{r['reported']:.4f}", flush=True)

    pd.DataFrame(curves).to_csv(AV / "bestlayer_lambda_curve.csv", index=False)
    pd.DataFrame(rows).to_csv(AV / "bestlayer_lambda_points.csv", index=False)
    print(f"\nwrote {AV/'bestlayer_lambda_curve.csv'}")
    print(f"wrote {AV/'bestlayer_lambda_points.csv'}")
    print("\nNOTE: 'best lam' is a max over the whole grid on TEST, so it is a\n"
          "selection number, not a result. The honest value is the per-fold\n"
          "val-selected one in bestlayer_skip.csv ('reported').")


if __name__ == "__main__":
    main()
