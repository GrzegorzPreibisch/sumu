# Where this method sits in the 2026 literature

Notes from a literature sweep run August 2026, for positioning the paper. Numbers are
quoted from the cited papers; anything derived is marked as such.

---

## 1. Somebody now benchmarks on exactly our four networks

**SciNO** — Kang et al., "Score-informed Neural Operator for Enhancing Ordering-based
Causal Discovery", NeurIPS 2025, arXiv:2508.12650 (LG AI Research / Korea University).
No public code.

They use *"the BNLearn collection of four real-world graphs — MAGIC-NIAB (44/66),
ECOLI70 (46/70), MAGIC-IRRI (64/102), ARTH150 (107/150)"*, at **n = 10,000**, 5 runs,
linear and nonlinear SEMs. They report **OD (order divergence), SHD and SID**.

OD counts true edges the estimated order forbids — i.e. exactly our notion of an
unreachable edge. So `max recall = (E − OD)/E` converts their OD into our ceiling.
**This conversion is our arithmetic, not printed in their paper:**

| network (edges) | DiffAN | DiffAN + SciNO | CaPS (linear) |
|---|---|---|---|
| MAGIC-NIAB (66) | 0.848 | 0.942 | 0.455 |
| **ECOLI70 (70)** | **0.677** | **0.794** | **0.643** |
| MAGIC-IRRI (102) | 0.888 | 0.902 | 0.588 |
| ARTH150 (150) | 0.764 | 0.861 | 0.673 |

**Our layering ceiling on ECOLI70 is 0.74 (n=300) to 0.82 (n=1000)** — it brackets
SciNO's 0.794 and beats DiffAN's 0.677 and CaPS's 0.643.

Two caveats that must be stated if this comparison goes in the paper:
- **They use n = 10,000; we use n = 300 and 1,000.** Getting a comparable order cap
  with 10x less data is the actual claim, and it has to be phrased that way.
- Their raw SHD/SID are on a different scale from ours because they solve a harder
  (nonlinear, or equal-variance-free) problem. Do not put their SHD next to ours.
- The paper never defines OD formally and never says whether its SHD is DAG- or
  CPDAG-based. Since these are order+pruning methods the output is fully oriented, so
  DAG-level is near-certain, but they cannot be cited as saying so.

**Others who used these four networks:** PARNI-DAG (arXiv 2023, n=100, SHD only);
Gadget itself (NeurIPS 2020 supplement Fig. D.1, but scored by causal-effect MSE and
mixing, not structure recovery); Scutari, Vitolo & Tucker (IJAR 2019, scaled CPDAG SHD,
no order-based method); TriOpt (arXiv 2026, ARTH150 only).

So the honest framing is **not** "nobody uses these networks". It is: *no one has
reported power/FDR structure recovery for a Bayesian layering pipeline on them, and no
one has connected the ordering cap to what the selection stage then attains.*

---

## 2. The layer-partition family — the closest relatives

A statistics line, largely disjoint from the score-matching literature, that produces
**partitions into layers** rather than total permutations. This is our setting and
these are the citations we cannot omit.

- **Grassi & Tarantino, "SEMdag", PLOS ONE 20(1):e0317283, 2025** (R package `SEMgraph`).
  **The closest published architecture to ours**: linear Gaussian SEM with equal error
  variances, topological layers (*"L₀ denotes the set of the root (source) nodes … Lⱼ is
  the set of all the source nodes in the subgraph G[V − Lⱼ₋₁]"*), then *"LASSO
  regressions of the j-th outcome variable on the predictor (ancestor) variables"*.
  **But it is never validated against a known DAG** — four RNA-seq datasets, SHD used
  only as a pairwise distance between methods for clustering, quality judged by
  downstream random-forest MCC. Cite it as the nearest architecture and note the
  absence of ground-truth validation; our contribution survives precisely there.
- **Zhao, He & Wang, JMLR 23(269):1–34, 2022** — linear non-Gaussian DAG with diverging
  node count. States our constraint verbatim: *"for each node k ∈ 𝒜_t … pa_k ⊂ 𝒮_{t+1}"*,
  parents in strictly earlier layers. Bottom-up, graphical lasso + distance-covariance
  tests, avoids faithfulness.
- **Zhou, He, Zhong & Wang, JCGS 31(4):1269–1279, 2022** (`TLDAG`) — topological layers
  + ℓ₁-penalised GLM per node on accumulated earlier layers. Non-Gaussian. Does not
  report layer accuracy separately from edge accuracy.
- **Chen, Drton & Wang, Biometrika** (arXiv:1807.03419) — equal-variance ordering by
  minimal conditional variance, then CV-lasso on predecessors. The one paper reporting
  an ordering metric (Kendall's τ) beside edge metrics, though τ is not a cap.

---

## 3. Our decomposition appears to be new

`D_top` (order divergence) is standard across SCORE, NoGAM, DiffAN, TriOpt; DiffAN and
Vo et al. (KDD 2026) both state it *"provides a lower bound on the SHD of the final
algorithm (irrespective of the pruning method)"*. SciNO prints OD and SHD side by side
for every dataset **but never connects them**.

**No paper reports what fraction of the ceiling the selection stage attains.** Our
framing — the layering caps power at 0.74–0.82 and the selection reaches 92–96% of that
cap — is new. SciNO's Table 2 is the natural thing to position it against.

Nearby precedents to acknowledge so a referee cannot say we missed them: **TriOpt**
(arXiv:2605.17465) compares pruning methods *given the true ordering* and shows
structurally meaningful order errors hurt far more than positional ones; **SSTS**
(arXiv:2604.25295) proves Kendall's τ degrades quadratically in layer width even at
zero edge violations — a citable argument that a total order over-specifies the target
and **layers are the right object**. Both 2026 preprints, internals unverified.

---

## 4. The varsortability objection, and why it does not land here

Reisach, Seiler & Weichwald (NeurIPS 2021): the standard simulation — ER graph, iid
random weights, unit noise — has varsortability > 0.94, so sorting by marginal variance
nearly recovers a causal order and least-squares methods exploit it.

**Measured analytically for our four networks** (`sortability.R`, closed form from the
published coefficients, no simulation):

| network | varsortability | R²-sortability |
|---|---|---|
| ecoli70 | 0.730 | 0.393 |
| magic-niab | **0.308** | 0.773 |
| magic-irri | 0.630 | 0.528 |
| arth150 | 0.631 | **0.239** |

0.5 is chance. These fitted networks are far less sortable than the simulations the
critique targets; magic-niab is anti-varsortable. `sortcheck.R` validates the
implementation by reproducing the near-1 value (0.982) on the standard iid-weight ER-2
setting. **These numbers do not appear to have been published.**

Supporting citations:
- **Deng, Bello, Ravikumar & Aragam, NeurIPS 2024** prove the Gaussian log-likelihood
  is **scale-invariant**: the DAG structures derived from raw and standardised data
  coincide even in finite samples. NOTEARS's failure is its least-squares surrogate,
  not continuous optimisation as such.
- **Reisach et al. §3.4 explicitly exempts** score-equivalent criteria and scale-free CI
  tests. Our criterion is likelihood/BIC-type, so the critique is structurally
  inapplicable — and that can be said with a citation.
- **CausalRegNet** (Kovačević et al., arXiv:2407.06015) measures varsortability 0.477–0.492
  for parameters *fitted to real expression data* versus 0.986 for iid sampled weights —
  published support for preferring fitted networks.
- Do **not** quote raw-vs-standardised SHD numbers from Reisach et al.; that paper has
  only boxplots. Use **Yi et al., ICLR 2025** instead: NOTEARS SHD 1.5 → 18.0 and GOLEM
  1.4 → 17.5 after standardisation on ER-2, d=10, n=2000 — i.e. worse than PC and GES.

---

## 5. Consequences for the paper

1. **Add SciNO as a comparison point on the ordering stage**, converting OD to a recall
   cap, with the n=10,000-vs-1,000 caveat stated plainly.
2. **Cite SEMdag as the closest architecture** and note it was never validated against a
   known DAG.
3. **Report the analytic sortability table** — it is cheap, novel, and pre-empts the
   most likely referee objection.
4. **Keep the ceiling-vs-attained decomposition front and centre**; it is the new part.
5. Still missing: a comparison against `sparsebn`/CCDr (Aragam & Zhou), the closest
   competitor by method, whose papers sit unread in `DAG/`.
