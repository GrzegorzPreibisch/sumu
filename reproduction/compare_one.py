"""Factorial comparison of selection variants on one cached layering.

usage: compare_one.py CACHEFILE

Isolates three choices, holding the layering fixed:
  * the penalty schedule  (2022 final config vs log(n) + c*log(A_j))
  * the loss form         (RSS + pen*k   vs   n*log(RSS/n) + pen*k)
  * refit                 (score shrunk lasso coefs vs OLS on retained support)
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bnsim

path = sys.argv[1]
base = os.path.basename(path)[:-4]
net, n, rep = base.rsplit("_", 2)
n, rep = int(n), int(rep.replace("rep", ""))

out = f"cmp/{base}.csv"
os.makedirs("cmp", exist_ok=True)
if os.path.exists(out):
    sys.exit(0)

d = np.load(path, allow_pickle=True)
X = d["X"].astype(float)
layers = json.loads(str(d["layers"]))
_, mat_t, _ = bnsim.load_network(net)

rows = []


def run(label, pb, pg, log_crit, refit):
    m = bnsim.edges_from_layering(X, layers, pb, pg,
                                  log_crit=log_crit, refit=refit)
    s = bnsim.score(m, mat_t)
    rows.append([net, n, rep, label, s["power"], s["fdr"], s["shd"]])


# --- the configuration the 2022 experiments actually ran -------------------
# division_decay overwrites BOTH penalties with 0.5*log(A_j); the log(n) and
# log(p) values computed at the call site are discarded.
half = lambda nn, p, a: 0.5 * np.log(max(a, 2))
run("2022_default__RSS_norefit", half, half, log_crit=False, refit=False)
run("2022_default__log_refit",   half, half, log_crit=True,  refit=True)

# --- 2x2: loss form x refit, at a fixed sensible penalty -------------------
pb = lambda nn, p, a: np.log(nn)
pg = lambda nn, p, a: 2.0 * np.log(max(a, 2))
for log_crit in (False, True):
    for refit in (False, True):
        run(f"logcrit={int(log_crit)}_refit={int(refit)}", pb, pg, log_crit, refit)

# --- constant vs layer-adaptive GIC penalty (both log-crit + refit) --------
run("gic_const_logp", pb, lambda nn, p, a: 2.0 * np.log(p), log_crit=True, refit=True)

with open(out, "w") as f:
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")
print(f"{base} done", flush=True)
