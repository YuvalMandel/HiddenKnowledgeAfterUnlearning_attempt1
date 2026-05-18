# Contributions

---

## 1. Hidden-knowledge gap: K_int > K_ext for all eight unlearning methods

**Claim.** After unlearning, the internal knowledge score (K_int, per-layer LR probe,
5-fold CV) remains substantially above chance for every one of the eight methods
tested, while the external score (K_ext, True/False logit margin) drops to near chance.
The gap is 8–18 percentage points and is consistent across methods.

**Key numbers (best per-layer CV probe, ck8):**

| Method   | K_int | K_ext | Gap  |
|----------|------:|------:|-----:|
| Base     | 77.4% | 78.5% | −1   |
| GradDiff | 74.0% | 62.2% | +12  |
| PB&J     | 66.4% | 56.5% | +10  |
| RMU      | 68.6% | 57.0% | +12  |
| RMU-LAT  | 69.0% | 60.7% | +8   |
| RepNoise | 68.6% | 50.3% | +18  |
| ELM      | 68.9% | 57.2% | +12  |
| RR       | 70.8% | 60.9% | +10  |
| TAR      | 67.9% | 55.4% | +13  |

![K_int vs K_ext — Mean K pairwise score with 5-fold CV error bars](plots/k_int_vs_ext_kfold_single.png)

Under the more demanding **full-layer probe** (all 33 layers → PCA-256 → LR,
312 pre-filtered questions), the gap is even more visible for output-targeting methods,
while representation-targeting methods (RMU, RMU-LAT) show near-zero gap — consistent
with their direct latent-space intervention disrupting global hidden-state geometry.

![K_int vs K_ext — full-layer probe, 312 pre-filtered questions](plots/kint_kext_gap/kint_kext_gap.png)

---

## 2. Discovery of the suppressed category: 17–35% of questions per method

**Claim.** For every method tested, 17–35% of questions fall into the *suppressed*
category at ck8: K_ext has collapsed to chance but K_int remains above chance.
Standard behavioural benchmarks report only K_ext and cannot detect these questions.

**Four-subset counts at ck8 (312 pre-filtered questions):**

| Method   | Retained | Suppressed | Forgotten | Lucky |
|----------|:--------:|:----------:|:---------:|:-----:|
| GradDiff | 68.9%(215) | 22.8%(71) | 5.1%(16) | 3.2%(10) |
| RMU      | 46.8%(146) | 17.3%(54) | 17.3%(54) | 18.6%(58) |
| RMU-LAT  | 54.5%(170) | 17.0%(53) | 13.1%(41) | 15.4%(48) |
| RepNoise | 37.8%(118) | 34.3%(107) | 15.7%(49) | 12.2%(38) |
| ELM      | 54.2%(169) | 26.3%(82) | 8.3%(26)  | 11.2%(35) |
| RR       | 54.8%(171) | 19.9%(62) | 9.9%(31)  | 15.4%(48) |
| TAR      | 52.2%(163) | 25.3%(79) | 13.5%(42) | 9.0%(28)  |
| PB&J     | 45.5%(142) | 22.8%(71) | 15.4%(48) | 16.3%(51) |

GradDiff produces 99.9% gibberish output yet still leaves 22.8% suppressed — K_ext
collapse alone does not guarantee K_int erasure.

---

## 3. Trajectory analysis: suppressed questions maintain higher K_int throughout training

**Claim.** Suppressed questions maintain a significantly higher full-layer K_int
trajectory (AUC over ck1–ck8) than forgotten questions for all eight methods,
with large effect sizes (Cohen's d = 0.66–1.67, all q < 0.001).

The four subsets show distinct K_int trajectory patterns across checkpoints and layers.
Below is GradDiff as a representative example:

- **Retained** (top-left): K_int stays high (green) across all layers and all ck1–ck8.
- **Suppressed** (top-right): K_int also stays high throughout — these questions are
  internally stable from the start and never decay.
- **Forgotten** (bottom-left): K_int is high at base but collapses by ck3–ck8,
  especially in early/middle layers.
- **Lucky** (bottom-right): K_int is low throughout (the base model did not know these
  internally) yet K_ext happens to be recoverable at ck8.

![All 4 subset K_int trajectories — GradDiff](plots/ckpt_k_subset_heatmap/ckpt_k_subset_GradDiff.png)

**Forest plot — suppressed − forgotten K_int trajectory gap, all 8 methods:**

![Forest plot](plots/ckpt_layer_trajectory_metrics/suppressed_vs_forgotten_forest_plot.png)

| Method   |    Δ   | 95% CI          |   q    | Sig | Cohen d | Cliff δ |
|----------|-------:|-----------------|-------:|-----|--------:|--------:|
| GradDiff | +0.275 | [0.167, 0.381]  | <0.001 | *** |   1.665 |   0.704 |
| TAR      | +0.320 | [0.247, 0.392]  | <0.001 | *** |   1.626 |   0.739 |
| RepNoise | +0.178 | [0.129, 0.226]  | <0.001 | *** |   1.331 |   0.647 |
| ELM      | +0.207 | [0.110, 0.305]  | <0.001 | *** |   1.073 |   0.574 |
| PB&J     | +0.136 | [0.092, 0.182]  | <0.001 | *** |   1.095 |   0.584 |
| RMU-LAT  | +0.175 | [0.101, 0.247]  | <0.001 | *** |   0.962 |   0.493 |
| RR       | +0.128 | [0.056, 0.204]  | <0.001 | *** |   0.768 |   0.425 |
| RMU      | +0.118 | [0.052, 0.186]  | <0.001 | *** |   0.661 |   0.367 |

---

## 4. Mechanism: unlearning collapses K_ext equally for suppressed and forgotten; the difference is K_int retention

**Claim.** Both suppressed and forgotten questions experience a large K_ext drop
(~0.70–0.80 units). The distinguishing feature is K_int: suppressed questions retain
it (mean K_int trajectory ≈ 0.79), while forgotten questions lose it (≈ 0.34).
The K_int − K_ext gap at ck8 is ~+0.61 for suppressed vs ~+0.17 for forgotten.

![Mechanism summary: suppressed − retained mean differences across methods](plots/retained_vs_suppressed_mechanism/retained_vs_suppressed_summary_bars.png)

*Left panel (blue, leftward):* suppressed questions have slightly lower K_int
trajectory than retained — showing they are not simply the same as retained.
*Middle panel (red, rightward):* all methods show large K_ext drop for suppressed
(~0.70–0.80 units, all \*\*\*), nearly matching the forgotten subset's drop.
*Right panel (red, rightward):* all methods show a large K_int − K_ext gap for
suppressed (~0.60–0.75 units, all \*\*\*), confirming internal knowledge survives
even as external access collapses.

**Averaged across methods:**

| Subset     | K_int traj | Ext drop | K_int−K_ext gap @ ck8 |
|------------|:----------:|:--------:|:---------------------:|
| Retained   |    0.84    |   0.10   |        −0.04          |
| Suppressed |    0.79    |   0.82   |        +0.61          |
| Forgotten  |    0.58    |   0.83   |        +0.17          |

---

## 5. Base-model features predict K_int trajectory but not the suppression split

**Claim.** Questions with stronger base-model representational structure tend, across
the population, to have higher K_int trajectories after unlearning (ρ = 0.58 at
the question×method level). This is a distributional regularity — it does not mean
we can predict for a specific question whether it will become suppressed or forgotten.
If that were possible, the K_int − K_ext gap would be predictable from base features,
but it is essentially not (R² ≈ 0.01). The suppression/forgetting split is driven by
the unlearning algorithm applied to each question, not by the question's pre-existing
structure.

![Predicted vs actual K_int trajectory AUC (HGB, ρ = 0.583)](plots/trajectory_regression/predicted_vs_actual_k_int_traj_auc.png)

**Regression results (RepeatedKFold 5×10, 4,592 question×method pairs):**

| Target                    | Best | R²    | σ     | ρ     |
|---------------------------|:----:|------:|------:|------:|
| K_int trajectory AUC      | HGB  | 0.345 | 0.030 | 0.583 |
| K_int @ ck8               | HGB  | 0.140 | 0.029 | 0.385 |
| K_ext drop @ ck8          | RF   | 0.129 | 0.032 | 0.380 |
| K_int − K_ext gap @ ck8   | RF   | 0.014 | 0.028 | 0.210 |

The near-zero R² for the int–ext gap is the key result: even though K_int trajectory
is moderately predictable (ρ = 0.58), the gap that defines suppressed vs. forgotten
is not. A question with high base-model K_int may end up retained, suppressed, or
forgotten depending on how the unlearning algorithm interacts with it — the outcome
cannot be read off from the base model alone.
