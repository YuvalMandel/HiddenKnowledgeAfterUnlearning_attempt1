# Knowledge-Recovery via Reversing the Unlearning Displacement

**Question.** After an unlearning method *suppresses* a fact (the model stops outputting it externally, yet a probe shows the knowledge is still internally decodable), can we *causally recover* the suppressed answer — i.e. make the unlearned model output it again — by steering its activations back toward where the **base** model had them?

**Where the steering happens.** We steer the **unlearned checkpoint-8 model of the method itself** (e.g. `LLM-GAT/llama-3-8b-instruct-repnoise-checkpoint-8`), by **adding a fixed vector to the last-token residual stream at a set of layers** `L = {3, 6, 9, 12, 15}` during the forward pass. The vector is built from *stored* base-vs-ck8 hidden states; the model we *run* is the ck8 model.

---

## 1. Notation

- Question $q$ has 4 options; correct-option index $c(q)$.
- $H_m[q,o,\ell]\in\mathbb{R}^{4096}$ — the **last-token hidden state** at layer $\ell\in\{0,\dots,32\}$ produced by model $m$ on the True/False **verify-prompt** of (question $q$, option $o$). Stored in `inside_out_out/<m>/bio_hs.npy`, shape $(1273,4,33,4096)$.
  - $m=\text{base}$ → original Llama-3-8B-Instruct
  - $m=M\text{ck8}$ → method $M$'s unlearned checkpoint 8
- $\mathcal{Q}^\*$ = 701 questions the base model knows ($K_\text{int}=K_\text{ext}=1$).
- Subsets, from **method $M$'s ck8** scores, for $q\in\mathcal{Q}^\*$:

$$\text{suppressed }\;\mathcal{S}_M=\{q: K_\text{int}^{M}(q)=1,\;K_\text{ext}^{M}(q)=0\},\qquad \text{forgotten }\;\mathcal{F}_M=\{q: K_\text{int}^{M}(q)=0,\;K_\text{ext}^{M}(q)=0\}.$$

Each subset is split into **train** (build the vector) and **held-out test** (measure recovery), 60/40.

---

## 2. Per-model "suppressed" centroid (one vector per layer)

For each layer $\ell\in L$, average the **correct-option** hidden state over the **suppressed train** questions, separately for each model:

$$u^{\text{base}}_\ell=\frac{1}{|\mathcal{S}_M^{\text{train}}|}\sum_{q\in \mathcal{S}_M^{\text{train}}} H_{\text{base}}[q,\,c(q),\,\ell],
\qquad
u^{\text{ck8}}_\ell=\frac{1}{|\mathcal{S}_M^{\text{train}}|}\sum_{q\in \mathcal{S}_M^{\text{train}}} H_{M\text{ck8}}[q,\,c(q),\,\ell].$$

Same questions, same correct option — **two models** ⇒ two centroids in the shared $\mathbb{R}^{4096}$ space.

---

## 3. The displacement (diff) vector

$$\boxed{\,d_S(\ell)=u^{\text{base}}_\ell-u^{\text{ck8}}_\ell\,}\qquad \in\mathbb{R}^{4096},\ \text{for each }\ell\in L.$$

This is the **average shift the correct-option representation underwent from base → ck8** for suppressed questions. It points **from the unlearned (suppressed) representation toward the base (expressed) one** — i.e. "what unlearning removed."

We use the **raw** diff (not normalized), so $\alpha=1$ means "undo exactly the displacement unlearning introduced."

---

## 4. The intervention (what we add back)

Run the **ck8 model** on a held-out test question. At each layer $\ell\in L$, with a forward hook on decoder layer $\ell-1$ (whose output is $\texttt{hidden\_states}[\ell]$), add the diff to the **last token** only:

$$\boxed{\,h^{(\ell)}_{\text{last}} \;\leftarrow\; h^{(\ell)}_{\text{last}} \;+\; \alpha\cdot d_S(\ell)\,}\qquad\text{simultaneously for all }\ell\in L.$$

A single fixed set of vectors, added to every option's prompt (a white-box steering attack). We sweep $\alpha$:

| $\alpha$ | meaning |
|---|---|
| $0$ | plain ck8 (suppressed; baseline) |
| $+1$ | **full displacement reversal** → maps the avg ck8 rep onto the avg base rep (the principled value) |
| $+0.5,+2$ | dose-response around 1 |
| $-1$ | push *further into* suppression (sign control) |

Because the residual stream carries each layer's edit forward, the per-layer edits **compound** — so $\alpha=1$ is "≈ undo unlearning," and the small sweep maps the dose-response.

---

## 5. What we measure

For each option $o$ of a test question, read the True/False logit margin at the last token:

$$s_\text{ext}(q,o\mid\alpha)=\text{logit}(\text{“True”})-\text{logit}(\text{“False”})\quad\text{(under steering }\alpha).$$

Then the **external** pairwise knowledge score (recovery target):

$$K_\text{ext}^{M}(q\mid\alpha)=\frac{1}{3}\sum_{o\neq c(q)}\mathbb{1}\!\left[s_\text{ext}(q,c(q)\mid\alpha)>s_\text{ext}(q,o\mid\alpha)\right].$$

**Recovery** = mean $K_\text{ext}$ over held-out suppressed test questions climbs from $\approx 0$ (suppressed) toward $1$ as $\alpha\to 1$.

---

## 6. Controls (the double dissociation)

| condition | steer set | applied to | prediction |
|---|---|---|---|
| **`supp_dS`** | $d_S$ | held-out **suppressed** | $K_\text{ext}$ **recovers** (above chance) ✓ |
| **`forg_dF`** | $d_F(\ell)=u^{\text{base},F}_\ell-u^{\text{ck8},F}_\ell$ | held-out **forgotten** | no real recovery (≤ chance) |
| **`supp_random`** | $r(\ell),\ \lVert r(\ell)\rVert=\lVert d_S(\ell)\rVert$ | suppressed | no recovery (direction matters) |
| **$\alpha<0$** | $-d_S$ | suppressed | no recovery / more suppressed |

A clean result is: **suppressed recovers above chance, forgotten only to chance, random not at all** — causal proof the suppressed knowledge was *hidden, not erased*.

---

## 7. Code / outputs

- Script: `plots/causal_recover.py <METHOD>` (per method). SLURM: `plots/slurm_causal_recover*.sh`.
- Per-method outputs: `plots/activation_vectors/causal_recover_<METHOD>.csv` and `.png`.
- Requires the method's ck8 weights (ungated `LLM-GAT/llama-3-8b-instruct-<slug>-checkpoint-8`).

### RepNoise result (first run)
$K_\text{ext}$ on held-out suppressed: baseline $0.15 \to 0.70$ at $\alpha=1$ (plateau); random control $0.24$; forgotten control $0.54$ (≈ chance). → suppressed knowledge **recovered, direction-specific**, forgotten only to chance. Recovery is substantial but partial ($0.70$, not $\approx 1$).
