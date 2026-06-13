# Which unlearned models are recoverable, and by which vector

We try to make an unlearned (ck8) model **externally output its suppressed answers** by steering it (last token, layers 3/6/9/12/15) along a recovery vector, sweeping coefficient `alpha`, and measuring **K_ext** on a held-out suppressed test set.

- **Base-recoverable** = steering along the **cross-model displacement** `d_S = base_centroid - ck8_centroid` works (needs the original/base model).
- **Unlearned-model-recoverable** = steering along the **within-model expression direction** `r = retained_centroid - suppressed_centroid`, computed *inside the unlearned model only* (no base model), works.
- **"Works" / clean** = recovered K_ext is clearly above **both** the matched-norm **random** control **and** the **forgotten** control (steering forgotten questions). Recovered to *chance* (~0.5) with controls just as high ⇒ not real recovery.
- `HK gap` = mean(K_int − K_ext) over Q\* for that ck8 model (its aggregate hidden-knowledge).

## Classification

| Method | HK gap | Base (`d_S`): K_ext 0→best | Unlearned (`r`): K_ext 0→best | Controls (rand / forgotten) | Verdict |
|---|---|---|---|---|---|
| **RepNoise** | 0.214 | **0.15 → 0.70** ✓ | **0.13 → 0.71** ✓ | rand ~0.13–0.24, forg ~0.41–0.54 | **Recoverable by BOTH** (strong; the clearest case) |
| **PB&J** | 0.120 | 0.17 → 0.50 (~) | 0.18 → 0.52 (~) | rand ~0.22–0.29, forg ~0.43 | **Recoverable by BOTH** (marginal — recovery only ~0.07–0.09 above forgotten) |
| **GradDiff** | 0.171 | **0.22 → 0.63** ✓ | 0.25 → 0.33 ✗ | rand 0.24, forg 0.30 | **Base-recoverable ONLY** (within-`r` fails: GradDiff answers "True" to everything, margin +9, which breaks its internal retained−suppressed axis — needs the base reference) |
| RR | 0.130 | 0.20 → 0.50 ✗ | 0.19 → 0.48 ✗ | rand ~0.43, forg ~0.42 | **Not recoverable** (random/forgotten rise just as much — non-specific) |
| RMU | 0.105 | 0.15 → 0.49 ✗ | 0.15 → 0.36 ✗ | rand ~0.45, forg ~0.52 | **Not recoverable** (forgotten ≥ recovery) |
| RMU-LAT | 0.094 | 0.15 → 0.39 ✗ | 0.16 → 0.32 ✗ | rand ~0.36, forg ~0.31 | **Not recoverable** (≈ random/forgotten) |
| ELM | 0.179 | 0.20 → 0.25 ✗ | 0.21 → 0.24 ✗ | rand ~0.22, forg ~0.19–0.24 | **Not recoverable at all** (barely moves) |
| TAR | 0.197 | 0.18 → 0.25 ✗ | 0.18 → 0.22 ✗ | rand ~0.18, forg ~0.21–0.29 | **Not recoverable at all** (barely moves) |

**Summary:** Recoverable by **both** vectors → **RepNoise** (strong), **PB&J** (marginal). **Base-only** → **GradDiff**. **Not recoverable by either** → RR, RMU, RMU-LAT, ELM, TAR (5/8). The within-model self-contained switch is *narrower* than the cross-model displacement (it loses GradDiff and gains nothing).

## Recoverability vs HK gap (see `hk_vs_recovery.png`)

Directed recoverability (recover@best − random) vs HK gap, per method:
- **cross-model `d_S`:** Pearson r = **0.54 (p=0.17)**, Spearman ρ = **0.60 (p=0.12)** — a *weak-to-moderate positive* trend, **not significant** (n=8).
- **within-model `r`:** Pearson r = **0.17 (p=0.68)** — essentially **no** correlation.

So more hidden knowledge does **not cleanly predict** recoverability. The trend is driven by RepNoise (high HK gap, high recovery) and GradDiff, but **ELM and TAR are clear counterexamples**: they have *large* HK gaps (0.18, 0.20) yet are **not recoverable at all** — substantial hidden knowledge that this steering method cannot bring back. Recoverability depends on *how* the method suppresses (reversible representational shift vs. deep output collapse), not just *how much* it hides.
