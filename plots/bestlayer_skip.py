#!/usr/bin/env python3
"""Source the True/False evidence from the BEST layer, not the last one.

Three variants, all reading an intermediate activation instead of h32 only
(`bestlayer_skip_plan.md`):

  A. lens      s = v . RMSNorm(h^L)                           0 free params
  B. residual  s = v . h32 + beta * scaled(v . RMSNorm(h^L))  1 scalar
  C. skip      s = v . h32 + lambda*||v||*(u_L . h^L)         4096 + 1  <- primary

C's lambda -> inf limit is K_int at layer L, so it reaches past the layer-32
ceiling that bounds the readout edit -- which matters most for RMU, whose
layer-32 headroom is 0.004 but whose layer-6 headroom is 0.084.

NO GPU. The skip reads an activation and writes to the logits; it never feeds
back into the network, so no hidden state changes and the edited model's margin
is an explicit function of the UNMODIFIED model's cached activations. (The old
mid-network steering did change h, which is why that needed a forward pass.)

Protocol matches readout_edit_cv.py so the numbers are comparable:
    fit u_L on TRAIN (827) | select (L, lambda) on VAL (191) | report on TEST (255)
L is selected on validation, not taken from k_scores.parquet's recorded
best_layer -- that was chosen with val inside the fit, and reusing it would leak.

hidden_states[32] is post-final-norm but hidden_states[L<32] is not, so A and B
apply the model's own final RMSNorm to h^L before projecting on v. The norm
weight lives in the same shard as lm_head and is cached to
activation_vectors/norm_<METHOD>.npz on first use; v is cached the same way.

WHERE TO RUN IT. Newton rejects CPU-only sbatch ("CPU-only jobs are not allowed
on the public GPU infrastructure") -- use the DARWIN cluster, which shares the
same home directory and conda env: `slurm_bestlayer_darwin.sh`, partition
public. A GPU buys nothing here: the cost is sklearn's lbfgs LogisticRegression,
and moving that to cuML/torch would change the fitted directions and break
comparability with readout_edit_cv.py.

MEMORY. The Newton login node OOM-kills anything near 3 GB. Holding all 11
layers (918 MB) plus the mmap'd 1.3 GB bio_hs.npy plus transformers/torch peaked
at 5.1 GB and was killed twice. So: one layer resident at a time, file pages
dropped with madvise after each read, and torch imported only if the v/norm
caches miss. Peak is then ~1.5 GB.

Outputs (activation_vectors/):
    bestlayer_skip.csv        variant C, per method x {real, shuffled}
    bestlayer_skip_perq.csv   per-question k0/k1 for C, real labels
    bestlayer_resid.csv       variant B, per method
    bestlayer_lens.csv        variant A + CV K_int, per method x layer

Usage: python plots/bestlayer_skip.py [METHOD ...] [--layers=6,32] [--refit]
"""
import gc
import json
import mmap
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inside_out_knowledge as iok  # noqa: E402

AV = Path(__file__).resolve().parent / "activation_vectors"
HF = Path(os.environ.get("KINT_HF_HOME", str(Path.home() / "hf_kint")))
# duplicated from readout_edit_all rather than imported: that module pulls in
# torch + transformers at import time (~800 MB RSS) and this script must not.
SLUGS = {"GradDiff": "graddiff", "RMU": "rmu", "RMU-LAT": "rmu-lat",
         "RepNoise": "repnoise", "ELM": "elm", "RR": "rr", "TAR": "tar",
         "PB_J": "pbj"}
REPOS = {m: f"LLM-GAT/llama-3-8b-instruct-{s}-checkpoint-8"
         for m, s in SLUGS.items()}
REPOS["base"] = "meta-llama/Meta-Llama-3-8B-Instruct"
CKPT = {m: f"{m}_ck8" for m in SLUGS}
CKPT["base"] = "base"

LAYERS = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
LAMBDAS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
           512.0, 1024.0, 2048.0, 4096.0, 16384.0, 65536.0, 1e6]
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e6]
RETAIN = 0.95
LAST = 32
RMS_EPS = 1e-5


def get_v(method):
    """v = W_U[' True'] - W_U[' False'], from whichever cache already has it."""
    cached = AV / f"v_{method}.npz"
    if cached.exists():
        return np.load(cached)["v"]
    npz = AV / f"damage_{method}.npz"
    if npz.exists():
        d = np.load(npz)
        if "v" in d.files:
            np.savez(cached, v=d["v"])
            return d["v"]
    from readout_edit_all import readout_direction   # imports torch
    v = readout_direction(method)[0]
    np.savez(cached, v=v)
    return v


def get_norm(method):
    """model.norm.weight -- same shard as lm_head, so no extra download."""
    cached = AV / f"norm_{method}.npz"
    if cached.exists():
        d = np.load(cached)
        return d["g"], float(d["eps"])
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    repo, cache = REPOS[method], str(HF / "hub")
    idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json",
                                         cache_dir=cache)))
    shard = hf_hub_download(repo, idx["weight_map"]["model.norm.weight"],
                            cache_dir=cache)
    with safe_open(shard, framework="pt") as f:
        g = f.get_tensor("model.norm.weight").float().numpy().astype(np.float32)
    try:
        cfg = json.load(open(hf_hub_download(repo, "config.json",
                                             cache_dir=cache)))
        eps = float(cfg.get("rms_norm_eps", RMS_EPS))
    except Exception:
        eps = RMS_EPS
    np.savez(cached, g=g, eps=eps)
    return g, eps


def load_layer(path, L):
    """One layer as float32, with the file's pages handed straight back."""
    hs = np.load(path, mmap_mode="r")
    out = np.asarray(hs[:, :, L, :], dtype=np.float32)
    try:
        hs._mmap.madvise(mmap.MADV_DONTNEED)
    except Exception:
        pass
    del hs
    gc.collect()
    return out


def rmsnorm(h, g, eps):
    r = np.sqrt(np.mean(h * h, -1, keepdims=True) + eps)
    return (h / r) * g


def fit_u(h, ci, idx):
    p = Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(C=1.0, max_iter=2000,
                                             random_state=iok.SEED))])
    p.fit(h[idx].reshape(-1, h.shape[-1]), iok.build_labels(ci, idx))
    w = p.named_steps["clf"].coef_.ravel() / p.named_steps["sc"].scale_
    return (w / np.linalg.norm(w)).astype(np.float32)


def k_per_q(s, ci, idx):
    return np.array([np.mean([s[i, ci[q]] > s[i, j]
                              for j in range(4) if j != ci[q]])
                     for i, q in enumerate(idx)])


def boot_ci(d, seed=1):
    bs = d[np.random.default_rng(seed).integers(0, len(d), (10000, len(d)))]
    return np.percentile(bs.mean(1), [2.5, 97.5])


def run(method, rows, perq, lens_rows, resid_rows):
    path = iok.OUT_DIR / CKPT[method] / "bio_hs.npy"
    ci = np.load(iok.OUT_DIR / "bio_correct_idx.npy").astype(int)
    nq = np.load(path, mmap_mode="r").shape[0]      # header only
    v = get_v(method)
    nv = float(np.linalg.norm(v))
    try:
        g, eps = get_norm(method)
    except Exception as e:
        g, eps = None, None
        print(f"  {method}: no norm weight ({type(e).__name__}), "
              f"lens runs UNNORMALISED", flush=True)

    splits = list(iok.get_cv_splits(nq))
    ci_shuf = np.random.default_rng(0).integers(0, 4, size=nq)
    tags = (("real", ci), ("shuffled", ci_shuf))

    # ---- one pass over the layers, one layer resident at a time. Everything
    # kept afterwards is a (nq, 4) score array, ~20 KB each -- so the whole set
    # is cached, and a later change to LAMBDAS (which never enters a fit) then
    # costs seconds instead of a refit. Delete the npz or pass --refit to redo.
    lens, pu = {}, {}
    proj = AV / f"bestlayer_proj_{method}.npz"
    if proj.exists() and "--refit" not in sys.argv:
        z = np.load(proj)
        have = set(z["layers"].tolist())
        if set(LAYERS) <= have:
            for L in LAYERS:
                lens[L] = z[f"lens_{L}"]
                for tag, _ in tags:
                    for fi in range(len(splits)):
                        pu[tag, fi, L] = z[f"pu_{tag}_{fi}_{L}"]
            print(f"  {method}: projections from cache ({len(LAYERS)} layers)",
                  flush=True)
        else:
            print(f"  {method}: cache misses layers "
                  f"{sorted(set(LAYERS) - have)}, refitting", flush=True)
            lens, pu = {}, {}
    for L in (L for L in LAYERS if L not in lens):
        H = load_layer(path, L)
        # h32 is ALREADY post-final-norm, so normalising it again would apply g
        # twice; leaving it raw makes lens@32 == K_ext, the consistency check.
        hn = rmsnorm(H, g, eps) if (g is not None and L != LAST) else H
        lens[L] = np.einsum("qod,d->qo", hn, v)
        del hn
        for tag, ci_fit in tags:
            for fi, (tr, va, te) in enumerate(splits):
                pu[tag, fi, L] = H @ fit_u(H, ci_fit, tr)
        del H
        gc.collect()
        print(f"  {method} layer {L} done", flush=True)
    if not proj.exists() or "--refit" in sys.argv:
        np.savez_compressed(
            proj, layers=np.array(LAYERS),
            **{f"lens_{L}": lens[L] for L in LAYERS},
            **{f"pu_{tag}_{fi}_{L}": pu[tag, fi, L] for L in LAYERS
               for tag, _ in tags for fi in range(len(splits))})
        print(f"  {method}: cached projections -> {proj.name}", flush=True)

    pv = lens[LAST]
    k_all = np.arange(nq)
    k_ext_all = k_per_q(pv, ci, k_all).mean()

    for tag, _ in tags:
        picks, fpicks, k0s, k1s, k2s, qs = [], [], [], [], [], []
        kint_te = {L: [] for L in LAYERS}
        for fi, (tr, va, te) in enumerate(splits):
            k0v = k_per_q(pv[va], ci, va)
            keep = k0v == 1.0
            best, pick = k0v.mean(), (LAST, 0.0)
            fbest, fpick = k0v.mean(), (LAST, 0.0)
            for L in LAYERS:
                p = pu[tag, fi, L]
                for lam in LAMBDAS:
                    kk = k_per_q(pv[va] + lam * nv * p[va], ci, va)
                    m = kk.mean()
                    if m > fbest:               # unconstrained ceiling
                        fbest, fpick = m, (L, lam)
                    if keep.any() and kk[keep].mean() < RETAIN:
                        continue
                    if m > best:                # retention-constrained pick
                        best, pick = m, (L, lam)
                kint_te[L].append(k_per_q(p[te], ci, te))
            L, lam = pick
            fL, flam = fpick
            picks.append(pick)
            fpicks.append(fpick)
            k0s.append(k_per_q(pv[te], ci, te))
            k1s.append(k_per_q(pv[te] + lam * nv * pu[tag, fi, L][te], ci, te))
            k2s.append(k_per_q(pv[te] + flam * nv * pu[tag, fi, fL][te], ci, te))
            qs.append(te)
        k0, k1, k2, q = (np.concatenate(x) for x in (k0s, k1s, k2s, qs))
        d, df = k1 - k0, k2 - k0
        lo, hi = boot_ci(d)
        flo, fhi = boot_ci(df)
        rows.append(dict(
            method=method, labels=tag, n=len(d), k_ext=round(k0.mean(), 4),
            skipped=round(k1.mean(), 4), gain=round(d.mean(), 4),
            lo=round(lo, 4), hi=round(hi, 4), sig=int(lo > 0 or hi < 0),
            layers="/".join(str(p[0]) for p in picks),
            lambdas="/".join(f"{p[1]:g}" for p in picks),
            skipped_free=round(k2.mean(), 4), gain_free=round(df.mean(), 4),
            lo_free=round(flo, 4), hi_free=round(fhi, 4),
            sig_free=int(flo > 0 or fhi < 0),
            layers_free="/".join(str(p[0]) for p in fpicks),
            lambdas_free="/".join(f"{p[1]:g}" for p in fpicks)))
        r = rows[-1]
        print(f"  C {method:<10} {tag:<9} {r['k_ext']:.4f} -> {r['skipped']:.4f}"
              f"  gain {r['gain']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]"
              f"{'*' if r['sig'] else ' '}  L {r['layers']}  lam {r['lambdas']}",
              flush=True)
        print(f"    free                 -> {r['skipped_free']:.4f}"
              f"  gain {r['gain_free']:+.4f}"
              f" [{r['lo_free']:+.4f},{r['hi_free']:+.4f}]"
              f"{'*' if r['sig_free'] else ' '}  L {r['layers_free']}"
              f"  lam {r['lambdas_free']}", flush=True)
        if tag == "real":
            perq.append(pd.DataFrame(dict(method=method, question_idx=q,
                                          k0=k0, k1=k1)))
            for L in LAYERS:
                lens_rows.append(dict(
                    method=method, layer=L, normed=int(g is not None),
                    k_lens=round(k_per_q(lens[L], ci, k_all).mean(), 4),
                    k_int_cv=round(np.concatenate(kint_te[L]).mean(), 4),
                    k_ext32=round(k_ext_all, 4)))

    # ---- variant B: add the earlier layer's lens score back into the readout.
    # beta is scale-free: the added term is rescaled to pv's train-fold spread,
    # so beta=1 mixes the two readouts at equal variance and beta=inf is
    # lens-only.
    picks, k0s, k1s = [], [], []
    for tr, va, te in splits:
        sc = {L: float(pv[tr].std() / max(lens[L][tr].std(), 1e-12))
              for L in LAYERS}
        k0v = k_per_q(pv[va], ci, va)
        keep = k0v == 1.0
        best, pick = k0v.mean(), (LAST, 0.0)
        for L in LAYERS:
            for b in BETAS:
                kk = k_per_q(pv[va] + b * sc[L] * lens[L][va], ci, va)
                if keep.any() and kk[keep].mean() < RETAIN:
                    continue
                if kk.mean() > best:
                    best, pick = kk.mean(), (L, b)
        L, b = pick
        picks.append(pick)
        k0s.append(k_per_q(pv[te], ci, te))
        k1s.append(k_per_q(pv[te] + b * sc[L] * lens[L][te], ci, te))
    k0, k1 = np.concatenate(k0s), np.concatenate(k1s)
    d = k1 - k0
    lo, hi = boot_ci(d)
    resid_rows.append(dict(
        method=method, n=len(d), k_ext=round(k0.mean(), 4),
        resid=round(k1.mean(), 4), gain=round(d.mean(), 4),
        lo=round(lo, 4), hi=round(hi, 4), sig=int(lo > 0 or hi < 0),
        layers="/".join(str(p[0]) for p in picks),
        betas="/".join(f"{p[1]:g}" for p in picks)))
    r = resid_rows[-1]
    print(f"  B {method:<10} {'resid':<9} {r['k_ext']:.4f} -> {r['resid']:.4f}"
          f"  gain {r['gain']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]"
          f"{'*' if r['sig'] else ' '}  L {r['layers']}  beta {r['betas']}",
          flush=True)


def _write(rows, perq, lens_rows, resid_rows, sfx):
    if rows:
        pd.DataFrame(rows).to_csv(AV / f"bestlayer_skip{sfx}.csv", index=False)
    if perq:
        pd.concat(perq).to_csv(AV / f"bestlayer_skip_perq{sfx}.csv", index=False)
    if lens_rows:
        pd.DataFrame(lens_rows).to_csv(AV / f"bestlayer_lens{sfx}.csv",
                                       index=False)
    if resid_rows:
        pd.DataFrame(resid_rows).to_csv(AV / f"bestlayer_resid{sfx}.csv",
                                        index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    for a in args:                      # --layers=6,32 : smoke-test grid
        if a.startswith("--layers="):
            LAYERS = sorted({int(x) for x in a.split("=", 1)[1].split(",")}
                            | {LAST})
    ms = [a for a in args if not a.startswith("-")] or (["base"] + list(SLUGS))
    sfx = f"_{ms[0]}" if len(ms) == 1 else ""
    print(f"skip from layer L into the readout | layers {LAYERS}\n"
          f"fit u on TRAIN | select (L, lambda) on VAL | report on TEST\n")
    rows, perq, lens_rows, resid_rows = [], [], [], []
    for m in ms:
        try:
            run(m, rows, perq, lens_rows, resid_rows)
        except Exception as e:
            print(f"  {m}: FAILED {type(e).__name__}: {e}", flush=True)
        _write(rows, perq, lens_rows, resid_rows, sfx)
    print(f"\nwrote {AV}/bestlayer_*{sfx}.csv")
