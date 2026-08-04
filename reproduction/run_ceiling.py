"""Layering + oracle ceiling + GIC1/GIC2, all from the same run. usage: NET N REP"""
import json, os, sys, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bnsim, ss_select, cpdag

net, n, rep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = f"ceil/{net}_{n}_rep{rep}.csv"
os.makedirs("ceil", exist_ok=True)
if os.path.exists(out): sys.exit(0)

df = pd.read_csv(f"rbn/{net}_{n}_rep{rep}.csv", sep=";")
X = np.array(df, dtype=float); X = (X - X.mean(0)) / X.std(0, ddof=0)
_, mat_t, cols = bnsim.load_network(net)
order = [cols.index(c) for c in df.columns]
mat_t = mat_t[np.ix_(order, order)]

t = time.time()
layers = bnsim.map_layering(X, seed=1000 * n + rep)

# oracle: exactly the true edges whose parent sits in a strictly earlier layer
depth = {v: i for i, l in enumerate(layers) for v in l}
mat_o = np.zeros_like(mat_t)
for c in range(mat_t.shape[0]):
    for pa in np.nonzero(mat_t[c])[0]:
        if depth.get(int(pa), 1 << 30) < depth.get(c, -1):
            mat_o[c, int(pa)] = 1.0
so = bnsim.score(mat_o, mat_t)
rows = [[net, n, rep, "ORACLE", so["power"], 0.0, cpdag.shd_cpdag(mat_o, mat_t)]]

for gic in (1, 2):
    m = ss_select.edges_ss(X, layers, variant="one", gic=gic)
    s = bnsim.score(m, mat_t)
    rows.append([net, n, rep, f"GIC{gic}", s["power"], s["fdr"],
                 cpdag.shd_cpdag(m, mat_t)])
with open(out, "w") as f:
    for r in rows: f.write(",".join(str(x) for x in r) + "\n")
print(f"{net} {n} rep{rep} ceiling={so['power']:.2f} {time.time()-t:.0f}s", flush=True)
