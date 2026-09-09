#!/usr/bin/env python3
"""Re-run readout_edit_cv.py's layer-32 edit on a denser lambda grid, offline.

readout_edit_cv.py used LAMBDAS = {0,1,2,4,8,16,32,64,1e6}: 64 is the largest
finite value and 1e6 is 15,000x beyond it. The same gap turned out to be
truncating the layer-skip results (bestlayer_skip_plan.md), so the layer-32
numbers need the same check before they go in a table beside them.

No refit is needed. bestlayer_skip.py already cached the layer-32 probe
projections, and its fit_u is character-for-character readout_edit_cv.py's
fit_dir on the same array, so pu[tag, fold, 32] IS that script's probe and
lens_32 IS its v.h32. This reproduces the published CSV first as a gate, then
re-selects lambda on the denser grid.

Usage: python plots/readout_edit_cv_regrid.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import inside_out_knowledge as iok  # noqa: E402  (get_cv_splits only)

AV = ROOT / "plots" / "activation_vectors"
ORDER = ["base", "GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR",
         "PB_J"]
OLD = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 1e6]
NEW = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
       2048.0, 4096.0, 16384.0, 65536.0, 1e6]
RETAIN = 0.95
LAST = 32


def k_per_q(s, ci, idx):
    return np.array([np.mean([s[i, ci[q]] > s[i, j]
                              for j in range(4) if j != ci[q]])
                     for i, q in enumerate(idx)])


def boot_ci(d, seed=1):
    bs = d[np.random.default_rng(seed).integers(0, len(d), (10000, len(d)))]
    return np.percentile(bs.mean(1), [2.5, 97.5])


def run(m, ci, splits, grid, tag):
    z = np.load(AV / f"bestlayer_proj_{m}.npz")
    nv = float(np.linalg.norm(np.load(AV / f"v_{m}.npz")["v"]))
    pv = z[f"lens_{LAST}"]
    chosen, k0s, k1s = [], [], []
    for fi, (tr, va, te) in enumerate(splits):
        p = z[f"pu_{tag}_{fi}_{LAST}"]
        k0v = k_per_q(pv[va], ci, va)
        keep = k0v == 1.0
        lam, best = 0.0, k0v.mean()
        for cand in grid:
            kk = k_per_q(pv[va] + cand * nv * p[va], ci, va)
            if keep.any() and kk[keep].mean() < RETAIN:
                continue
            if kk.mean() > best:
                lam, best = cand, kk.mean()
        chosen.append(lam)
        k0s.append(k_per_q(pv[te], ci, te))
        k1s.append(k_per_q(pv[te] + lam * nv * p[te], ci, te))
    k0, k1 = np.concatenate(k0s), np.concatenate(k1s)
    d = k1 - k0
    lo, hi = boot_ci(d)
    return dict(method=m, k_ext=round(float(k0.mean()), 4),
                attacked=round(float(k1.mean()), 4),
                gain=round(float(d.mean()), 4), lo=round(lo, 4),
                hi=round(hi, 4), sig=int(lo > 0 or hi < 0),
                lambdas="/".join(f"{x:g}" for x in chosen),
                at_edge=sum(x == max(g for g in grid if g < 1e6)
                            for x in chosen))


def main():
    ci = np.load(ROOT / "inside_out_out" / "bio_correct_idx.npy").astype(int)
    splits = list(iok.get_cv_splits(len(ci)))
    pub = pd.read_csv(AV / "readout_edit_cv.csv")
    pub = pub[pub.labels == "real"].set_index("method")

    print("GATE: reproduce readout_edit_cv.csv from the cached projections")
    ok = True
    for m in ORDER:
        r = run(m, ci, splits, OLD, "real")
        same = (abs(r["gain"] - float(pub.loc[m, "gain"])) < 5e-4
                and r["lambdas"] == str(pub.loc[m, "lambdas"]))
        ok &= same
        print(f"  {m:<9} gain {r['gain']:+.4f} vs {pub.loc[m,'gain']:+.4f}"
              f"  lam {r['lambdas']:<26} {'OK' if same else 'MISMATCH'}")
    print(f"\ngate: {'PASS' if ok else 'FAIL -- do not trust the rest'}\n")
    if not ok:
        return

    rows = []
    print("EXTENDED grid, layer 32, same protocol")
    for m in ORDER:
        r = run(m, ci, splits, NEW, "real")
        s = run(m, ci, splits, NEW, "shuffled")
        r["shuffled_gain"] = s["gain"]
        r["shuffled_sig"] = s["sig"]
        r["old_gain"] = float(pub.loc[m, "gain"])
        r["old_attacked"] = float(pub.loc[m, "attacked"])
        rows.append(r)
        print(f"  {m:<9} {r['k_ext']:.4f} -> {r['attacked']:.4f}  gain "
              f"{r['gain']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]"
              f"{'*' if r['sig'] else ' '}  (was {r['old_gain']:+.4f})"
              f"  lam {r['lambdas']}")
    df = pd.DataFrame(rows)
    df.to_csv(AV / "readout_edit_cv_extended.csv", index=False)
    print(f"\nwrote {AV/'readout_edit_cv_extended.csv'}")


if __name__ == "__main__":
    main()
