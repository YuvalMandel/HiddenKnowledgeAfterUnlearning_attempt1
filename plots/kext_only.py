"""K_ext for models whose probes have not run yet.

K_ext needs no fitting -- it is compute_k() over the saved True/False margins --
so it is available the moment extraction finishes, hours before K_int. Also
reports what the OTHER token-id rule would give, from bio_ext_alt.npy.

Usage: python plots/kext_only.py base l3_dpo l3_npo ...
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inside_out_knowledge import OUT_DIR, compute_k

correct = np.load(OUT_DIR / "bio_correct_idx.npy").astype(int)
qi = np.arange(len(correct))

print(f"{'model':16s} {'K_ext':>7s} {'ret%':>7s}   {'K_ext(alt)':>10s} {'delta':>7s}")
print("-" * 54)
base_k = None
for m in sys.argv[1:]:
    d = OUT_DIR / m
    if not (d / "bio_ext.npy").exists():
        print(f"{m:16s} (no bio_ext.npy)")
        continue
    k = compute_k(np.load(d / "bio_ext.npy"), correct, qi).mean()
    if base_k is None:
        base_k = k
    alt_p = d / "bio_ext_alt.npy"
    if alt_p.exists():
        ka = compute_k(np.load(alt_p), correct, qi).mean()
        alt_s, dl = f"{ka:10.4f}", f"{ka - k:+7.4f}"
    else:
        alt_s, dl = f"{'-':>10s}", f"{'-':>7s}"
    # retention on the chance-to-base scale, 0.5 = floor
    ret = 100 * (k - .5) / (base_k - .5) if base_k > .5 else float("nan")
    print(f"{m:16s} {k:7.4f} {ret:7.1f}   {alt_s} {dl}")
