"""Evaluate every penalty config on one cached layering.  usage: evaluate_one.py CACHEFILE"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bnsim

path = sys.argv[1]
base = os.path.basename(path)[:-4]
net, n, rep = base.rsplit("_", 2)
n, rep = int(n), int(rep.replace("rep", ""))

out = f"eval/{base}.csv"
os.makedirs("eval", exist_ok=True)
if os.path.exists(out):
    sys.exit(0)

d = np.load(path, allow_pickle=True)
X = d["X"].astype(float)
layers = json.loads(str(d["layers"]))
_, mat_t, _ = bnsim.load_network(net)

# ceiling: an edge is recoverable only if the parent is in a strictly earlier layer
depth = {v: i for i, l in enumerate(layers) for v in l}
reach = sum(1 for c in range(mat_t.shape[0]) for pa in np.nonzero(mat_t[c])[0]
            if depth.get(int(pa), 1 << 30) < depth.get(c, -1))
ceiling = reach / mat_t.sum()

pb = lambda nn, p, a: np.log(nn)
rows = []


def run(label, pg, **kw):
    m = bnsim.edges_from_layering(X, layers, pb, pg, **kw)
    s = bnsim.score(m, mat_t)
    rows.append([net, n, rep, label, ceiling, s["power"], s["fdr"], s["shd"],
                 s["TP"], s["FP"], s["FN"], s["n_est"]])


# the 2022 code exactly as written: RSS criterion, no refit, sign-bugged threshold
run("code2022_buggy", lambda nn, p, a: np.log(p),
    log_crit=False, refit=False, buggy_sign=True)
# same but with the sign bug fixed - isolates the effect of the bug alone
run("code2022_signfixed", lambda nn, p, a: np.log(p),
    log_crit=False, refit=False, buggy_sign=False)

# thesis form: scale-free criterion, OLS refit on the retained support
for c in (1, 2, 3, 4, 5, 6, 8):
    run(f"fixed_c{c}", lambda nn, p, a, c=c: c * np.log(max(a, 2)),
        log_crit=True, refit=True, buggy_sign=False)

with open(out, "w") as f:
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")
print(f"{base}: ceiling={ceiling:.2f}", flush=True)
