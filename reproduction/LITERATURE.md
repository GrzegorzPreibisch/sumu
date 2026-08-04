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

---

## 6. Addendum — three more papers on these networks

### Kitson & Constantinou (2025): our exact algorithm set, our exact metric basis

**"Stable Structure Learning with HC-Stable and Tabu-Stable Algorithms", arXiv:2504.01740.**
Seven continuous networks including **all four of ours**; n = 10², 10³, 10⁴, 10⁵;
**25 replications** with variable order, names and row order randomised; ten algorithms —
HC, Tabu, HC-Stable, Tabu-Stable, FGES, PC-Stable, GS, Inter-IAMB, **MMHC, H2PC** — a
strict superset of our four. Metric basis verbatim: *"The structural metrics compare the
CPDAG of the true graph and with that of the learned graph."* Same basis as
`bnlearn::shd()`.

**They report F1 and BSF, not SHD, and aggregate the structural results into Figure 4
rather than tabulating them per network.** That makes our contribution precisely
nameable: *same algorithm family and same CPDAG basis as Kitson & Constantinou (2025),
but per-network SHD, power and FDR at fixed n, which they do not report.* Cite them and
say exactly that.

Their normalised-BIC table is worth borrowing for a different point — every learned
graph **beats the true graph on BIC** (Ecoli70: Tabu −42.267 vs true −42.382; Arth150:
Tabu-Stable −37.858 vs true −38.655). A clean illustration that fit and structural
accuracy diverge.

Their Table 9 also records outright failures at a 3-hour limit: on **ARTH150 at n=10,000
FGES, MMHC and H2PC all fail**, and at n=100,000 six of ten algorithms fail. Useful when
we justify our own sample sizes.

### Zuo (2024), Purdue PhD thesis — F1 on all four, 50 replications

C. Zuo, *Scalable Bayesian Methods for Probabilistic Graphical Models*, Purdue, 2024,
DOI 10.25394/PGS.25674651.v1. n = 0.5p, p, 2p; **Recall, Precision, F1 only — no SHD.**

ECOLI70 F1: HC 0.203 / 0.364 / 0.452 across the three sizes, MMHC 0.292 / 0.408 / 0.473,
HPC 0.390 / 0.480 / 0.522, their own method 0.521 / 0.542 / 0.564.
ARTH150 F1: HC 0.136 / 0.305 / 0.429, MMHC 0.310 / 0.415 / 0.523, HPC 0.357 / 0.448 / 0.509.

Useful as a prior when sanity-checking our own numbers: HC is worst at small n and
catches up by n = 2p; MMHC and HPC are the strongest classical methods.

### Sato, Scutari & Imoto (2025) — the one source we could not retrieve

*"Causal assessment of gene regulatory network in single-cell transcriptomics data based
on Bayesian networks"*, bioRxiv, 19 Dec 2025, DOI 10.64898/2025.12.17.695014, package
`github.com/noriakis/scstruc`. **ECOLI70 and ARTH150 at n = 100, 500, 1000** — our exact
range — with HC, MMHC, GS, H2PC, TABU **plus LASSO, CCDr, GES, L0/L0L1/L0L2, MCP, SCAD,
LiNGAM**, reporting **SHD, SID and F1**, data generated with `rbn()`.

**This is the single most valuable comparison point in the literature for us, and it is
paywalled to automated access** — bioRxiv returns 403 to every route tried. The numbers
are in Figure 2A and **Supplementary Table S1**:
https://www.biorxiv.org/content/10.64898/2025.12.17.695014v1.supplementary-material
**Download it by hand.** Confirmed qualitatively: at ECOLI70 n=100 a hurdle model with
zBIC had the lowest SHD while GES won on F1; at n=500 LASSO was best on SHD.

### TriOpt — ARTH150 SHD at n = 10,000, exact table

Joy & Zheleva, arXiv:2605.17465 (2026), linear SEM, **DAG-based** SHD:
DAGMA 138.0, GOLEM 163.0, NOTEARS 140.4 ± 0.89, TriOpt 135.0.

For scale: ARTH150 has 150 arcs, so these methods make roughly as many structural errors
as there are true edges. The published `wyniki.ods` figure for our method on ARTH150 at
**n = 1,000** is SHD 81 (GIC1) and 72 (GIC2). The comparison flatters us by a wide margin
even at a tenth of the sample size — **but their SHD is DAG-based and ours is CPDAG-based,
so this must not be presented as a like-for-like number.** It is a motivating observation,
not a result, until someone runs both on the same footing.

---

## 7. The (power, FDR) operating point — our strongest claim

**Aragam & Zhou, "Concave Penalized Estimation of Sparse Gaussian Bayesian Networks",
JMLR 16:2273–2328 (2015), Table 1.** ER random DAGs, weights U[0.5,2], unit error
variance, 50 replications. Their FDR charges reversed edges as false discoveries, same
as ours. SHD is DAG- and skeleton-based, **not CPDAG**.

| p = 200 (T = 185) | CCDr-MCP | CCDr-ℓ1 | GES | HC | MMHC | PC |
|---|---|---|---|---|---|---|
| TPR | 0.45 | 0.40 | 0.86 | 0.69 | 0.49 | 0.48 |
| FDR | 0.44 | 0.48 | 0.71 | 0.78 | 0.33 | 0.31 |

The pattern holds at p = 50 and p = 100 too: **no method in that table attains
TPR ≥ 0.7 at FDR below 0.69.** GES buys recall 0.71–0.86 at FDR 0.69–0.71; HC gets
0.59–0.69 at FDR 0.76–0.78; the methods with decent FDR (PC 0.31–0.37, MMHC 0.33–0.40)
sit at TPR 0.36–0.49. CCDr publishes **TPR 0.31–0.45 at FDR 0.44–0.46** as its own
headline.

**Our ECOLI70 n=1000 point — power 0.79 at FDR 0.22 — dominates every row.** That is the
claim to lead with, and it should be stated as a Pareto point, e.g. *"no method in
Aragam & Zhou (2015, Table 1) attains TPR ≥ 0.7 at FDR < 0.69."*

The closest published observational analogue is Fu & Zhou (JASA 2013) Table 2, topping
out at TPR 0.815 / FDR 0.245. Their better numbers (TPR 0.75–0.86 at FDR 0.09–0.23) are
**interventional** — one intervention per node — a materially easier problem.
Ghoshal & Honorio's FDR = 0 is a theory-regime demonstration with n set by their own
sample-complexity bound, not a benchmark result.

**Who even reports FDR:** Aragam & Zhou ✅, Fu & Zhou ✅, Ghoshal & Honorio ✅ (as
precision). NOTEARS reports it in a figure only. **GOLEM, NoCurl and DAGMA do not report
FDR at all.** van de Geer & Bühlmann (2013) and Loh & Bühlmann (2014) contain no
simulations whatsoever.

### Do NOT claim SHD superiority against the synthetic benchmarks

Per-edge difficulty differs by an order of magnitude and this would not survive review:

| | SHD | true edges | errors per true edge |
|---|---|---|---|
| NOTEARS, ER3 d=50 | 8.39 | 150 | **0.056** |
| DAGMA, d=50 (ER4/SF4 avg) | 12.03 | ~200 | ~0.06 |
| **ours, ECOLI70 n=1000** | **22.9** | **70** | **0.33** |
| tabu, ECOLI70 n=1000 | 48.8 | 70 | 0.70 |

A headline like "our SHD 23 beats DAGMA's 12" is meaningless — their graphs are far
easier per edge. **The honest SHD comparison is against our own tabu baseline on the
same graph, where we are 2.1x better.** Compounding it, all of NOTEARS/GOLEM/DAGMA/
NoCurl compute *directed* SHD while ours is CPDAG-based.

### Nobody has run these methods on our networks

Confirmed: **none** of sparsebn/CCDr, Fu & Zhou, NOTEARS, GOLEM, DAGMA, NoCurl,
Ghoshal & Honorio evaluated on ECOLI70, MAGIC-NIAB, MAGIC-IRRI or ARTH150. sparsebn uses
`pathfinder`, Sachs and LOAD; the rest use ER/SF synthetics.

And **Scutari, Graafland & Gutiérrez (IJAR 2019)** — the paper that established this
four-network suite — reports scaled SHD in **scatter plots, not tables**. So there is no
published per-algorithm ECOLI70 SHD number to cite against, which means our own tabu
baseline is the right and only comparator. That is a defensible position, not a gap.

**Open item worth checking before citing:** one extraction suggested Aragam & Zhou
*"selected the DAG with smallest SHD from each algorithm's solution path"* — i.e. oracle
tuning. Unconfirmed. If true it weakens their Table 1 as a baseline and strengthens us
further, so it is worth verifying in the PDF before leaning on it.
