"""Run the faithful SS port on one cached layering. usage: ss_one.py CACHEFILE"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bnsim, ss_select, cpdag

path = sys.argv[1]; base = os.path.basename(path)[:-4]
net, n, rep = base.rsplit("_", 2); n = int(n); rep = int(rep.replace("rep",""))
out = f"ss/{base}.csv"; os.makedirs("ss", exist_ok=True)
if os.path.exists(out): sys.exit(0)
d = np.load(path, allow_pickle=True)
X = d["X"].astype(float); layers = json.loads(str(d["layers"]))
_, mat_t, _ = bnsim.load_network(net)
rows = []
for gic in (1, 2):
    m = ss_select.edges_ss(X, layers, variant="one", gic=gic)
    s = bnsim.score(m, mat_t)
    rows.append([net, n, rep, f"GIC{gic}", s["power"], s["fdr"],
                 s["shd"], cpdag.shd_cpdag(m, mat_t)])
with open(out, "w") as f:
    for r in rows: f.write(",".join(str(x) for x in r) + "\n")
print(base, "ok", flush=True)
