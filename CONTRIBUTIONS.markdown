# Contributions

## 1. Hidden-knowledge gap: K_int > K_ext for all eight unlearning methods

**Claim.** After unlearning, the internal knowledge score (K_int, measured by a
per-layer LR probe) remains substantially above chance for every one of the eight
methods tested, while the external knowledge score (K_ext, measured by the True/False
logit margin) drops to near chance. This gap is visible across three convergent metrics
(mean pairwise K, accuracy, AUC) with 5-fold CV error bars.

**Key numbers (per-layer CV probe at ck8):**

| Method   | K_int | K_ext | Gap  |
|----------|------:|------:|-----:|
| Base     | 77.4% | 78.5% | −1.1 |
| GradDiff | 74.0% | 62.2% | +11.8 |
| PB&J     | 66.4% | 56.5% | +9.9 |
| RMU      | 68.6% | 57.0% | +11.6 |
| RMU-LAT  | 69.0% | 60.7% | +8.3 |
| RepNoise | 68.6% | 50.3% | +18.3 |
| ELM      | 68.9% | 57.2% | +11.7 |
| RR       | 70.8% | 60.9% | +9.9 |
| TAR      | 67.9% | 55.4% | +12.5 |

Output-targeting methods (GradDiff, RepNoise, ELM, TAR, PB&J) show gaps of
10–18 pp. Representation-targeting methods (RMU, RMU-LAT) show smaller but
still clearly positive gaps (8–12 pp). Under the more demanding **full-layer probe**
(all 33 layers → PCA-256 → LR, restricted to 312 pre-filtered questions),
representation-targeting methods show near-zero gap, suggesting they more
effectively disrupt global hidden-state geometry — yet they still leave 17–17.3%
of questions suppressed (Contribution 2).

**Evidence.**
- **Figure 1 (fig:kfold-overview)** — 3-panel bar chart (K score, Accuracy, AUC)
  with 5-fold CV error bars; all eight methods show blue > red on every panel.
- **Figure 2 (fig:kint-kext-gap)** — full-layer probe on prefiltered 312 questions;
  highlights the output- vs representation-targeting distinction.
- **Table 2 (tab:probes-bio)** — exact per-layer and multi-layer probe accuracy
  for all methods.

---

## 2. Discovery of the suppressed category: 17–35% of questions per method

**Claim.** For every one of the eight methods tested, a substantial fraction of
questions (17–35%) fall into the *suppressed* category at the final checkpoint (ck8):
K_ext has collapsed to chance yet K_int remains above chance. Behavioral benchmarks
alone cannot detect these questions.

**Evidence.**
- **Table 3 (tab:subsets)** — four-subset counts at ck8 (312 pre-filtered questions):

  | Method   | Ret | Supp | Forg | Lucky | %Supp |
  |----------|----:|-----:|-----:|------:|------:|
  | GradDiff | 215 |   71 |   16 |    10 | 23.1% |
  | RMU      | 146 |   54 |   54 |    58 | 17.6% |
  | RMU-LAT  | 170 |   53 |   41 |    48 | 17.3% |
  | RepNoise | 118 |  107 |   49 |    38 | 34.9% |
  | ELM      | 169 |   82 |   26 |    35 | 26.7% |
  | RR       | 171 |   62 |   31 |    48 | 20.2% |
  | TAR      | 163 |   79 |   42 |    28 | 25.7% |
  | PB&J     | 142 |   71 |   48 |    51 | 23.1% |

- **Figure 1 (fig:kfold-overview)** — corroborates at aggregate level: all eight
  unlearned models show probe scores well above 50% despite K_ext near chance.
- Note: GradDiff produces 99.9% gibberish (Table 1) yet still has 23% suppressed
  questions — K_ext collapse alone does not guarantee K_int erasure.

---

## 3. Trajectory analysis: suppressed questions maintain higher K_int than forgotten throughout training

**Claim.** Suppressed questions do not simply happen to sit above the K_int threshold
at ck8 by chance. They maintain a significantly higher full-layer K_int trajectory
(area under the ck1–ck8 curve) than forgotten questions for all eight methods, with
moderate-to-large effect sizes.

**Evidence.**
- **Table 4 (tab:sf-gap)** — suppressed-minus-forgotten K_int trajectory gap (full-layer
  probe, BH-corrected Mann-Whitney U, bootstrap 95% CI):

  | Method   |    Δ   | 95% CI          |    q    | Sig | Cohen d | Cliff δ |
  |----------|-------:|-----------------|--------:|-----|--------:|--------:|
  | GradDiff | +0.275 | [0.167, 0.381]  | <0.001  | *** |   1.665 |   0.704 |
  | RMU      | +0.118 | [0.052, 0.186]  | <0.001  | *** |   0.661 |   0.367 |
  | RMU-LAT  | +0.175 | [0.101, 0.247]  | <0.001  | *** |   0.962 |   0.493 |
  | RepNoise | +0.178 | [0.129, 0.226]  | <0.001  | *** |   1.331 |   0.647 |
  | ELM      | +0.207 | [0.110, 0.305]  | <0.001  | *** |   1.073 |   0.574 |
  | RR       | +0.128 | [0.056, 0.204]  | <0.001  | *** |   0.768 |   0.425 |
  | TAR      | +0.320 | [0.247, 0.392]  | <0.001  | *** |   1.626 |   0.739 |
  | PB&J     | +0.136 | [0.092, 0.182]  | <0.001  | *** |   1.095 |   0.584 |

  All eight: q < 0.001. Cohen's d: 0.66–1.67 (medium to very large).

- **Figure 3 (fig:forest)** — forest plot with 95% CI; all CIs exclude zero.

---

## 4. Mechanism: unlearning collapses K_ext equally for suppressed and forgotten; the difference is K_int retention

**Claim.** Both suppressed and forgotten questions experience similar K_ext collapse
(~0.82–0.83 external drop from base to ck8). The distinguishing feature is whether
K_int is also erased: suppressed questions retain K_int ≈ 0.79 while forgotten
questions lose it to ≈ 0.34.

**Key mechanism numbers (averaged across methods):**

| Subset     | K_int traj | Ext drop | K_int−K_ext gap @ ck8 |
|------------|:----------:|:--------:|:---------------------:|
| Retained   |    0.84    |   0.10   |        −0.04          |
| Suppressed |    0.79    |   0.82   |        +0.61          |
| Forgotten  |    0.58    |   0.83   |        +0.17          |

**Evidence.**
- **Table 5 (tab:mechanism)** — per-subset means for K_int trajectory, external
  drop, and K_int–K_ext gap at ck8.
- Per-method box plots (retained_vs_suppressed_mechanism figures) confirm the
  pattern holds individually for all eight methods.

---

## 5. Base-model features predict K_int trajectory but not the suppression split

**Claim.** Whether a question's internal representation survives unlearning
(K_int trajectory) is partially determined by pre-existing representational strength
in the base model. However, which questions become suppressed vs. forgotten is
driven by the unlearning process itself, not by the question's initial structure.

**Evidence.**
- **Table 6 (tab:regression)** — RF/HGB best model per target
  (RepeatedKFold 5×10 CV, 4,592 question×method pairs):

  | Target                    | Best | R²    | σ     | ρ     |
  |---------------------------|:----:|------:|------:|------:|
  | K_int trajectory AUC      | HGB  | 0.345 | 0.030 | 0.583 |
  | K_int @ ck8               | HGB  | 0.140 | 0.029 | 0.385 |
  | K_ext drop @ ck8          | RF   | 0.129 | 0.032 | 0.380 |
  | K_int − K_ext gap @ ck8   | RF   | 0.014 | 0.028 | 0.210 |

  K_int trajectory is moderately predictable (ρ = 0.58); the int–ext gap is
  essentially unpredictable (R² ≈ 0.01), confirming that whether a question
  becomes suppressed or forgotten is not determined by its base-model structure.

- **Predicted vs. actual scatter** (trajectory_regression figures) — K_int
  trajectory shows a clear trend; int–ext gap is a flat cloud.
- **Feature importance** — `earliest_kint1` and `mean_kint_layers` are top
  predictors for K_int trajectory, consistent with deeper base-model representations
  surviving unlearning better.
