# Paper Improvement Plan
Generated: 2026-04-18 | Status: IN PROGRESS

---

## Key Data Insights (from server CSVs)

### Generation accuracy (bio forget set)
| Model | GenAcc | Gibberish |
|---|---|---|
| Base (8B-Instruct) | 68.2% | 0.09% |
| GradDiff | 0.09% | **99.9%** |
| RMU | 45.8% | 13.6% |
| RMU-LAT | 46.3% | 13.7% |
| RepNoise | 6.0% | **88.4%** |
| ELM | 52.3% | 0% |
| RR | 38.5% | 32.0% |
| TAR | 50.0% | 0% (always False) |
| PB&J | 42.8% | 22.9% |
| Llama3-8B (raw) | 55.9% | 0% |

### Bio probe AUC (method-specific probes on own states, mb_lr = layers 12-22 + PCA)
| Model | Best-Layer LR AUC | Multi-Band LR AUC |
|---|---|---|
| Base (from table2) | 0.703 | — |
| GradDiff | 0.608 | 0.650 |
| RMU | 0.610 | 0.579 |
| RMU-LAT | 0.597 | 0.600 |
| RepNoise | 0.595 | 0.613 |
| ELM | 0.573 | 0.616 |
| RR | 0.600 | 0.621 |
| TAR | 0.609 | 0.637 |
| PB&J | 0.573 | 0.585 |
| Llama3-8B (raw) | 0.745 | 0.791 |

### Cross-probe transfer — resolves TODO in contribution bullet 3
**Base probe → unlearned states (table2):** All methods get ~0.49–0.52 AUC — **near chance**.
**Method probe → base states (table5):** 0.50–0.71 AUC — varies by method.
→ **Finding: Geometry SHIFTS substantially. But method-specific probes still work (0.58–0.65 AUC).**
→ Knowledge is preserved in a different representational subspace, not the same linear direction.

---

## Steps (check off as completed)

### STEP 1 — Fix PyCharm LaTeX preview ✅ DONE
- [x] Copy all images from Overleaf → `latex/imgs/`
- [x] Sync all `.tex` sections from Overleaf → `latex/sections/`
- [x] Add `\graphicspath{{imgs/}{./imgs/}}` to `latex/neurips_2025.tex`
- [x] Propagate same `\graphicspath` fix to `HiddenKnowledgeAfterUnlearning_overleaf_project/neurips_2025.tex`

### STEP 2 — Table generation pipeline ✅ DONE
- [x] Create `latex/tables/` directory
- [x] Write `generate_tables.py` that reads `data/*.csv` and outputs `latex/tables/*.tex`
- [x] Run `generate_tables.py` to produce all table bodies (7 files)
- [x] Updated all `\begin{tabular}` blocks in section `.tex` files to use `\input{tables/...}`

### STEP 3 — Fill all paper tables with real data ✅ DONE
- [x] Table 1: Gen + Logit accuracy (bio)
- [x] Table 2 (tab:probes-bio): Per-layer + multi-layer probe accuracy
- [x] Table 3 (tab:cross-probe): Cross-probe transfer
- [x] Table 4 (tab:cyber): Cyber gen + probe accuracy
- [x] Table A1: Base probes → all models (appendix)
- [x] Table A2: Cyber probes (appendix)
- [x] Table A3: MCQ accuracy
- [x] Llama-3-70B-Instruct removed (no data); "Llama-3-8B (raw)" used as 2nd reference

### STEP 4 — Fix contribution bullet 3 contradiction ✅ DONE
- [x] Updated: "geometry shifts, knowledge preserved in new subspace" with AUC 0.50/0.58-0.65 numbers

### STEP 5 — Fix grammar and TODO markers ✅ DONE
- [x] Removed all `\todo[inline]{}` markers from `1_introduction.tex`
- [x] Fixed contribution bullets with concrete numbers and correct claims
- [ ] Fix "Across two domain:" → "Across two domains:" in abstract
- [ ] Remove `\usepackage{todonotes}` from main tex

### STEP 6 — Add Conclusion section
- [ ] Create `latex/sections/7_conclusion.tex`
- [ ] Add `\input{sections/7_conclusion}` to `neurips_2025.tex` (after discussion, before `\newpage`)
- [ ] Content: restate main finding with numbers, cross-domain robustness, representational geometry shift, call for representational auditing standards

### STEP 7 — Add Broader Impact section
- [ ] Add as final subsection in Discussion or new section before references
- [ ] Address: dual-use risk of probe technique, white-box access requirement as mitigation, scope limitations (single model family)

### STEP 8 — Strengthen Related Work
- [ ] Add 1-sentence descriptions for RR, TAR, ELM, RMU-LAT, PB&J (currently just method names)
- [ ] Add citations for new contemporaneous papers:
  - "Probing Hidden Knowledge Holes in Unlearned LLMs" (OpenReview TFidSatsOC)
  - "Keeping an Eye on LLM Unlearning: The Hidden Risk and Remedy"
  - "LLM Unlearning via Neural Activation Redirection (LUNAR)"
  - "Are the Hidden States Hiding Something?" (ACL 2025)

### STEP 9 — Quantify the gap in abstract/intro
- [ ] Add concrete numbers: "generation accuracy drops to near-chance (38–52%) while probe AUC stays at 0.58–0.65"
- [ ] Add: "cross-probe transfer drops to near-chance (AUC ≈ 0.50), showing geometry shifts"

### STEP 10 — NeurIPS compliance
- [ ] Add NeurIPS paper checklist (mandatory — does not count toward 9 pages)
- [ ] Anonymize: remove GitHub URL from abstract for submission version
- [ ] Remove `\usepackage[main, final]{neurips_2025}` → use `\usepackage{neurips_2025}` for submission
- [ ] Check total page count stays ≤ 9

---

## Notes
- No Llama-3-70B-Instruct data available in CSVs — either collect it or remove from paper template
- GradDiff is extreme outlier: 99.9% gibberish, generation essentially broken
- TAR: always predicts False, gen_acc = 50% by chance, not meaningful forgetting
- RepNoise: 88% gibberish — also essentially broken
- Cyber base gen_acc=44% (below chance!) because base model tends to say True but cyber dataset is 50/50
