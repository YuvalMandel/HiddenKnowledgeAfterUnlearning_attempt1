#!/usr/bin/env python3
"""Appendix C (Figure 7): the model's own True/False direction at every layer vs. a probe.

For the base model and the eight final LLM-GAT checkpoints, at all 33 hidden
states (embedding output + 32 blocks):

  A (logit lens)  score a claim by v . RMSNorm(h_l), with v = W_U[" True"] -
                  W_U[" False"] and the model's own final RMSNorm. Nothing is
                  fitted, so every question is scored. h_32 is already
                  post-norm and is used as is, so layer 32 reproduces K_ext.
  B (probe)       K_int's probe with the layer fixed instead of selected: in
                  each fold of knowledge_lens.py, C is chosen by validation AUC,
                  the probe is refitted on train + val and scores the test fold.

Both scores are turned into K exactly as in knowledge_lens.py. Needs the
extracted hidden states (outputs/<model>/bio_hs.npy) and downloads only the
weight shard holding the unembedding and the final norm of each model.

The probes (9 models x 33 layers x 5 folds) are independent and run in
parallel; set LOGIT_LENS_JOBS (default: all cores). Per-layer copies of the
hidden states are staged in LOGIT_LENS_TMP (default: the system temp dir).

Prints K for both readouts at every layer and model (the values of Figure
7) and the numbers quoted in Appendix C. Results are cached in
outputs/logit_lens.csv; pass --recompute to rebuild it.

Usage: python numbers/logit_lens.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import knowledge_lens as kl                                  # noqa: E402
from common import K_CHANCE, METHODS, OUT_DIR  # noqa: E402

MODELS = ["base"] + [f"{m}_ck8" for m in METHODS]
LAST = 32
LAYERS = list(range(LAST + 1))
CACHE = OUT_DIR / "logit_lens.csv"


def readout_weights(model_id):
    """v = W_U[' True'] - W_U[' False'] and the final RMSNorm (weight, eps)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    repo = kl.model_path(model_id)
    t_id, f_id = kl.tf_token_ids(AutoTokenizer.from_pretrained(repo), model_id)
    index = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))
    wmap = index["weight_map"]
    head = "lm_head.weight" if "lm_head.weight" in wmap else "model.embed_tokens.weight"
    with safe_open(hf_hub_download(repo, wmap[head]), framework="pt") as f:
        W = f.get_slice(head)
        v = (W[t_id].float() - W[f_id].float()).numpy()
    with safe_open(hf_hub_download(repo, wmap["model.norm.weight"]), framework="pt") as f:
        g = f.get_tensor("model.norm.weight").float().numpy()
    eps = json.load(open(hf_hub_download(repo, "config.json"))).get("rms_norm_eps", 1e-5)
    return v.astype(np.float32), g.astype(np.float32), float(eps)


def rmsnorm(h, g, eps):
    return h / np.sqrt(np.mean(h * h, -1, keepdims=True) + eps) * g


def probe_fold_k(layer_file, correct_idx, tr, va, te):
    """K_int's probe at one fixed layer, one fold: C by validation AUC, refit on
    train + val, per-question K on the test fold (as knowledge_lens.probe_fold)."""
    H = np.load(layer_file).astype(np.float32)
    X = lambda q: H[q].reshape(-1, H.shape[-1])
    y_tr, y_va = kl.build_labels(correct_idx, tr), kl.build_labels(correct_idx, va)
    best_auc, best_C = -1.0, 1.0
    for C in kl.C_CANDIDATES:
        p = kl.make_probe(C).fit(X(tr), y_tr)
        try:
            auc = roc_auc_score(y_va, p.predict_proba(X(va))[:, 1])
        except ValueError:
            auc = 0.0
        if auc > best_auc:
            best_auc, best_C = auc, C
    trva = np.concatenate([tr, va])
    p = kl.make_probe(best_C).fit(X(trva), kl.build_labels(correct_idx, trva))
    proba = p.predict_proba(X(te))[:, 1].reshape(len(te), kl.N_OPTIONS)
    return kl.pairwise_k(proba, correct_idx[te])


def compute():
    correct_idx = kl.correct_idx_for("bio")
    splits = kl.get_cv_splits(len(correct_idx))
    stage = Path(os.environ.get("LOGIT_LENS_TMP", tempfile.gettempdir())) / "logit_lens_layers"
    stage.mkdir(parents=True, exist_ok=True)

    # A: the lens, and one contiguous file per (model, layer) for the probes.
    lens_k = {}
    for mid in MODELS:
        v, g, eps = readout_weights(mid)
        hs = np.load(OUT_DIR / mid / "bio_hs.npy")
        for L in LAYERS:
            H16 = np.ascontiguousarray(hs[:, :, L, :])
            np.save(stage / f"{mid}_{L}.npy", H16)
            H = H16.astype(np.float32)
            lens = np.einsum("qod,d->qo", H if L == LAST else rmsnorm(H, g, eps), v)
            lens_k[mid, L] = kl.pairwise_k(lens, correct_idx).mean()
        del hs
        print(f"  {mid:<14} lens done, layers staged", flush=True)

    # B: every (model, layer, fold) probe is independent.
    tasks = [(mid, L, f) for mid in MODELS for L in LAYERS for f in range(len(splits))]
    n_jobs = int(os.environ.get("LOGIT_LENS_JOBS", os.cpu_count()))
    print(f"  {len(tasks)} probe tasks on {n_jobs} workers", flush=True)
    ks = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(probe_fold_k)(stage / f"{mid}_{L}.npy", correct_idx, *splits[f])
        for mid, L, f in tasks)
    shutil.rmtree(stage, ignore_errors=True)

    probe_k = {}
    for (mid, L, _), k in zip(tasks, ks):
        probe_k.setdefault((mid, L), []).append(k)
    rows = [dict(model_id=mid, layer=L, k_lens=lens_k[mid, L],
                 k_probe=np.concatenate(probe_k[mid, L]).mean())
            for mid in MODELS for L in LAYERS]
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    return df


def print_numbers(df):
    """Both readouts at every layer, then the numbers quoted in Appendix C.

    Layer 0 (the embedding output) is left out: the four claims of a question
    share their last token there, so their states are identical and every K is 0.
    """
    df = df[df.layer >= 1]
    t = df.pivot(index="layer", columns="model_id", values="k_lens")[MODELS]
    p = df.pivot(index="layer", columns="model_id", values="k_probe")[MODELS]
    for name, tab in (("A: logit lens", t), ("B: K_int's probe, layer fixed", p)):
        print(f"\n{name}")
        print(tab.round(3).to_string())

    un = [m for m in MODELS if m != "base"]
    b = t["base"]
    print(f"\nbase lens: layer 12 {b[12]:.3f} -> layer 16 {b[16]:.3f}; "
          f"layers 16-32 {b.loc[16:].min():.3f}-{b.loc[16:].max():.3f}; "
          f"layer 32 {b[LAST]:.4f} (= K_ext)")
    print(f"unlearned lens maxima {t[un].max().min():.3f}-{t[un].max().max():.3f}")
    late = t[un].loc[12:]
    below = {m: late[m].min() for m in un if late[m].min() < K_CHANCE}
    print(f"below chance beyond layer 12: {len(below)} of 8 "
          + ", ".join(f"{m} (min {v:.3f})" for m, v in below.items()))
    print(f"probe, each unlearned model's best {p[un].max().min():.3f}-"
          f"{p[un].max().max():.3f}; across layers 10-16 "
          f"{p[un].loc[10:16].min().min():.3f}-{p[un].loc[10:16].max().max():.3f}")
    print(f"probe minus lens, unlearned models, layers 10-32: minimum "
          f"{(p[un] - t[un]).loc[10:].min().min():+.3f}")


def main():
    df = compute() if "--recompute" in sys.argv or not CACHE.exists() \
        else pd.read_csv(CACHE)
    print_numbers(df)


if __name__ == "__main__":
    main()
