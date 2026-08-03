# Reproducing the MCMC + thresholded-lasso results

Branch `reproduction-2026`. Written August 2026 while reconstructing the project in order
to finish the paper.

**Two published result sets exist and they disagree:** the tables in Maryia's dissertation
(Ch. 4, p. 76) and the numbers in `wyniki.ods`. The baseline columns (hc / tabu / mmhc / h2pc)
are identical in both, so only our method's column was ever recomputed.

This branch reproduces **both**, and identifies what separates them.

---

## The finding

The difference is one line in `Gadget.TL`. It used to read:

```python
thresholds = beta_bic[beta_bic > 0]     # grid built from POSITIVE coefficients only
...
beta_thres[beta_thres < delta] = 0      # signed comparison, not |.|
```

Any `delta > 0` therefore zeroes **every negative coefficient**. No negative-weight edge can
ever be recovered, and a node whose coefficients are all negative gets no parents at all.
Roughly half the edges in these networks carry negative weights, which is about what it costs.

Both lines now use `abs()`. `TL(..., legacy_threshold=True)` restores the old behaviour so the
two can be run side by side.

**Running the old code reproduces the dissertation. Fixing that one line reproduces `wyniki.ods`.**

| network | n | power, legacy | dissertation | power, fixed | wyniki GIC1 / GIC2 |
|---|---|---|---|---|---|
| ecoli70 | 300 | 0.46 | 0.49 | 0.63 | 0.69 / 0.67 |
| ecoli70 | 1000 | 0.54 | 0.55 | 0.74 | 0.76 / 0.74 |
| magic-niab | 300 | 0.20 | – | 0.36 | 0.35 / 0.21 |
| magic-niab | 1000 | 0.48 | – | 0.77 | – (cell missing from wyniki) |
| magic-irri | 300 | 0.19 | 0.24 | 0.31 | 0.35 / 0.25 |
| magic-irri | 1000 | 0.37 | 0.38 | 0.66 | 0.62 / 0.54 |

**Note for Błażej:** Grzegorz remembered the bug as the layering list coming back in reverse
order. That does not appear to be it. `PartitionMCMC._pi` scores layer-0 nodes with an empty
parent set, `_valid` requires each node in `R[i]` to have a candidate parent in `R[i-1]`, and
`bnet.partition()` builds layers by peeling parentless nodes — sumu is root-first throughout,
and `generate_final_dag` consumes it correctly. **If you remember fixing something else as
well, that is exactly what we are missing.**

---

## How closely it matches

Reproduced vs `wyniki.ods`, 10 replications per cell (6 for magic-irri n=1000):

| network | n | GIC1 repro (power/fdr/shd) | GIC1 published | GIC2 repro | GIC2 published |
|---|---|---|---|---|---|
| ecoli70 | 300 | 0.67 / 0.32 / 35.4 | 0.69 / 0.26 / 30 | 0.65 / 0.24 / 29.2 | 0.67 / 0.20 / 21 |
| ecoli70 | 1000 | 0.75 / 0.27 / 27.1 | 0.76 / 0.23 / 23 | 0.75 / 0.21 / 22.0 | 0.74 / 0.17 / 15 |
| magic-niab | 300 | 0.39 / 0.30 / 42.4 | 0.35 / 0.27 / 45 | 0.20 / 0.32 / 53.0 | 0.21 / 0.27 / 53 |
| magic-irri | 300 | 0.35 / 0.33 / 73.7 | 0.35 / 0.26 / 73 | 0.25 / 0.28 / 77.3 | 0.25 / 0.23 / 80 |
| magic-irri | 1000 | 0.70 / 0.21 / 39.5 | 0.62 / 0.24 / 48 | 0.53 / 0.16 / 49.3 | 0.54 / 0.19 / 49 |

- **power**: mean signed error **+0.006**, mean |error| 0.020 — unbiased.
- **fdr**: mean signed error **+0.031** — we consistently find slightly *more* false positives.
- **shd**: magic-niab and magic-irri land within 1 point; **ecoli70 is 4–8 too high in all four
  cells**, which is the one thing that does not fit.

Two network-specific quirks do come back, which is why we think the configuration is close to
right: magic-niab's unusually wide GIC1→GIC2 power gap (0.39→0.20 vs 0.35→0.21 published), and
magic-irri being the only network where GIC2 has *worse* SHD than GIC1 (77.3 vs 73.7 here,
80 vs 73 published).

---

## Questions we need you to answer

1. **What exactly are GIC1 and GIC2?** We could not find the R script anywhere. Our best fit is
   two multipliers on the GIC penalty, `c * log(k_0 + ... + k_{j-1})` with **c = 2** for GIC1 and
   **c = 5** for GIC2, used with the dissertation's scale-free criterion
   `n*log(RSS/n) + pen*k` and an **OLS refit on the retained support**. The committed Python does
   none of that — it uses `RSS + pen*k` and scores the shrunk lasso coefficients directly. Which
   is right?

2. **How was SHD computed for our method in `wyniki.ods`?** The Python `compute_stats` counts
   directed edges with a reversal charged as 1. `bnlearn::shd`, used for the baselines in
   `comparison.R`, compares **CPDAGs** and is more forgiving. If our column used `bnlearn::shd`
   too, that would explain part of the ecoli70 gap — though not why only ecoli70 is affected.

3. **Did you fix anything besides the threshold sign?** See the note above.

4. **Do you still have the R script and the per-replication outputs?** That would settle 1 and 2
   immediately.

---

## Running it

```bash
python3.11 -m venv venv
./venv/bin/pip install "numpy<2" scipy scikit-learn pandas "Cython<3"
./venv/bin/pip install -e .          # builds sumu; python-glmnet NOT required any more

cd reproduction
../venv/bin/python run_one.py ecoli70 1000 0        # layering, ~7 min for 46 nodes
../venv/bin/python evaluate_one.py cache/ecoli70_1000_rep0.npz
../venv/bin/python aggregate.py                      # comparison against both published sets
```

To run a whole set:

```bash
for net in ecoli70 magic-niab magic-irri; do for n in 300 1000; do for r in $(seq 0 9); do
  echo "$net $n $r"; done; done; done | xargs -P 6 -n 3 ../venv/bin/python run_one.py
ls cache/*.npz | xargs -P 6 -n 1 ../venv/bin/python evaluate_one.py
../venv/bin/python aggregate.py
```

`arth150` is in the benchmark set but was not run: its layering costs over an hour per
replication (vs ~7 min for ecoli70), so a full set is a multi-hour job.

### Files

| file | what |
|---|---|
| `bnsim.py` | network loading, Gaussian refit, simulation, layering, thresholded lasso, scoring |
| `run_one.py` | one (network, n, replication) → cached layering + data |
| `evaluate_one.py` | sweeps penalty configurations over one cached layering |
| `aggregate.py` | averages and prints the comparison against both published sets |
| `raw_runs.csv` | every replication we ran, one row per (network, n, rep, config) |
| `report.txt` | the output of `aggregate.py` from our run |

`bnsim.threshold_lasso` has switches for each thing we were unsure about, so you can flip them
and see which combination matches your R:

- `buggy_sign=True` — the original threshold bug
- `log_crit=False` — `RSS + pen*k` (what the Python did) vs `n*log(RSS/n) + pen*k` (dissertation)
- `refit=False` — score the shrunk lasso coefficients instead of refitting by OLS

### Two substitutions we had to make

- **No R and no bnlearn `.rda` files here**, so replications could not be drawn with `rbn`.
  Each network is instead refitted by OLS from its 1000-row pool in `datasets/`, given the true
  arc list, and fresh replications simulated from that. Coefficients should be very close to
  bnlearn's but are not identical — **this is the most likely source of the residual SHD gap,
  and you can remove it by generating the data in R.**
- **`python-glmnet` no longer builds** (its setup.py uses the removed `numpy.distutils`), so
  `sumu/_lasso_compat.py` supplies the same lasso path via scikit-learn. If you do have
  python-glmnet installed, sumu imports that instead and the shim is ignored.

---

## One result worth keeping regardless

Every run records how many true edges are *reachable at all* under the MAP layering — an edge
is only recoverable if its parent lands in a strictly earlier layer. That ceiling is **0.80** on
ecoli70 at n=1000, **0.63** on magic-niab, **0.83** on magic-irri at n=1000. The method reaches
0.75 of the 0.80 — about **94% of what the layering permits**.

Power is limited almost entirely by layering error, not by the lasso, which is exactly the gap
Corollary 4.2 assumes away. That seems worth a figure and a paragraph in the paper.
