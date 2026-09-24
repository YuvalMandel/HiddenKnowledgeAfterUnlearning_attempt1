#!/usr/bin/env python3
"""Appendix D: the model's own True/False direction at every depth vs. a probe.

For the base model and the eight final LLM-GAT checkpoints, at eleven layers:

  A (logit lens)  score a claim by v . RMSNorm(h_l), with v = W_U[" True"] -
                  W_U[" False"] and the model's own final RMSNorm. Nothing is
                  fitted, so every question is scored. h_32 is already
                  post-norm and is used as is, so layer 32 reproduces K_ext.
  B (probe)       a logistic-regression probe (C = 1) fitted at the same layer on
                  each fold's training questions and scored on its test fold,
                  with the folds of knowledge_lens.py.

Both scores are turned into K exactly as in knowledge_lens.py. Needs the
extracted hidden states (outputs/<model>/bio_hs.npy) and downloads only the
weight shard holding the unembedding and the final norm of each model.

Results are cached in outputs/logit_lens.csv; the figure and the printed
numbers are drawn from that file. Pass --recompute to rebuild it.

Usage: python plots/logit_lens.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import knowledge_lens as kl                                  # noqa: E402
# Figure 7 is drawn in matplotlib's default style, as in the paper.
from common import K_CHANCE, METHODS, OUT_DIR, label, save  # noqa: E402

MODELS = ["base"] + [f"{m}_ck8" for m in METHODS]
LAYERS = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
LAST = 32
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


def probe_scores(H, correct_idx, tr):
    """Probe fitted on the training questions; its linear score for every claim."""
    p = kl.make_probe(1.0)
    p.set_params(clf__max_iter=2000)
    p.fit(H[tr].reshape(-1, H.shape[-1]), kl.build_labels(correct_idx, tr))
    w = p.named_steps["clf"].coef_.ravel() / p.named_steps["sc"].scale_
    return H @ (w / np.linalg.norm(w))


def compute():
    correct_idx = kl.correct_idx_for("bio")
    rows = []
    for mid in MODELS:
        v, g, eps = readout_weights(mid)
        hs = np.load(OUT_DIR / mid / "bio_hs.npy", mmap_mode="r")
        splits = kl.get_cv_splits(hs.shape[0])
        for L in LAYERS:
            H = np.asarray(hs[:, :, L, :], dtype=np.float32)
            lens = np.einsum("qod,d->qo", H if L == LAST else rmsnorm(H, g, eps), v)
            k_probe = np.concatenate(Parallel(n_jobs=5, prefer="threads")(
                delayed(lambda tr, te: kl.pairwise_k(
                    probe_scores(H, correct_idx, tr)[te], correct_idx[te]))(tr, te)
                for tr, _, te in splits))
            rows.append(dict(model_id=mid, layer=L,
                             k_lens=kl.pairwise_k(lens, correct_idx).mean(),
                             k_probe=k_probe.mean()))
            print(f"  {mid:<14} layer {L:>2}  lens {rows[-1]['k_lens']:.4f}  "
                  f"probe {rows[-1]['k_probe']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    return df


def print_numbers(df):
    t = df.pivot(index="layer", columns="model_id", values="k_lens")
    p = df.pivot(index="layer", columns="model_id", values="k_probe")
    un = [m for m in MODELS if m != "base"]
    b = t["base"]
    print(f"base lens: layer 12 {b[12]:.3f}, 14 {b[14]:.3f}, 16 {b[16]:.3f}, "
          f"32 {b[32]:.3f}; step 12->14 {b[14] - b[12]:.3f}")
    print(f"unlearned lens maxima {t[un].max().min():.3f}-{t[un].max().max():.3f}; "
          f"{(t[un].min() < K_CHANCE).sum()} of 8 fall below chance, "
          f"minimum {t[un].min().min():.3f}")
    steps = t[un].diff()
    m, L = steps.stack().idxmax()[::-1]
    print(f"largest single-step lens gain of an unlearned model: "
          f"{steps[m][L]:.3f} ({m}, to layer {L}, arriving at {t[m][L]:.3f})")
    print(f"probe at the same layers, each unlearned model's best: "
          f"{p[un].max().min():.3f}-{p[un].max().max():.3f}; across layers 10-16: "
          f"{p[un].loc[10:16].min().min():.2f}-{p[un].loc[10:16].max().max():.2f}")


def figure_7(df):
    titles = ["A.  the model's own axis $v$ at layer $\\ell$\n"
              "(logit lens, no fitted parameters)",
              "B.  a probe fitted at layer $\\ell$\n($K_{int}$, cross-validated)"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45), sharey=True)
    for ax, col, ttl in zip(axes, ["k_lens", "k_probe"], titles):
        for mid in MODELS:
            d = df[df.model_id == mid]
            style = (dict(lw=1.5, color="k", ls="--", zorder=5) if mid == "base"
                     else dict(lw=0.9))
            ax.plot(d.layer, d[col], marker="o", ms=1.8,
                    label="base" if mid == "base" else label(mid[:-4]), **style)
        ax.axhline(K_CHANCE, color="k", lw=0.6, ls=":")
        ax.set_xlabel("layer $\\ell$", fontsize=7)
        ax.set_title(ttl, fontsize=7)
        ax.grid(color="0.93", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("$K$ (win fraction over the 3 distractors)", fontsize=7)
    axes[0].text(33, 0.494, "chance", fontsize=6, va="top", ha="right")
    axes[1].legend(fontsize=5.8, ncol=5, loc="lower right", frameon=False,
                   handlelength=1.0, handletextpad=0.3, labelspacing=0.25,
                   columnspacing=0.6, borderaxespad=0.2)
    fig.tight_layout(pad=0.4)
    save(fig, "logit_lens_by_layer")
    plt.close(fig)


def main():
    df = compute() if "--recompute" in sys.argv or not CACHE.exists() \
        else pd.read_csv(CACHE)
    print_numbers(df)
    figure_7(df)


if __name__ == "__main__":
    main()
