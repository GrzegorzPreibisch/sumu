"""Minimal demonstration of the Gadget.TL threshold bug. Runs in a second, no MCMC.

    python reproduction/test_threshold_bug.py

Two predictors matter: x0 with coefficient +1.5 and x1 with coefficient -1.5.
The legacy threshold rule recovers only x0.  Everything with a negative weight
is deleted, whatever its magnitude.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sumu.gadget import Gadget, ElasticNet

print("lasso backend:", ElasticNet.__module__)

rng = np.random.default_rng(1)
n, p = 400, 6
X = rng.normal(size=(n, p))
truth = np.zeros(p)
truth[0], truth[1] = 1.5, -1.5
y = X @ truth + rng.normal(0, 0.3, n)

g = Gadget.__new__(Gadget)          # TL touches no instance state
ok = True
for legacy in (True, False):
    beta, _ = Gadget.TL(g, X, y, np.log(n), np.log(p), legacy_threshold=legacy)
    picked = np.nonzero(beta)[0].tolist()
    print(f"legacy_threshold={legacy!s:<5} selected={picked} "
          f"coefficients={np.round(beta[beta != 0], 2).tolist()}")
    if legacy and picked != [0]:
        ok = False
        print("  ! expected the legacy rule to drop the negative predictor")
    if not legacy and picked != [0, 1]:
        ok = False
        print("  ! expected the fixed rule to recover both predictors")

print("\ntrue support = [0, 1] with coefficients [1.5, -1.5]")
print("PASS - bug reproduced and fix verified" if ok else "UNEXPECTED RESULT")
raise SystemExit(0 if ok else 1)
