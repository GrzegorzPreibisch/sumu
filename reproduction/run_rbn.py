"""sumu layering + SS selection on an rbn replication. usage: run_rbn.py NET N REP"""
import json, os, sys, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bnsim, ss_select, cpdag

net, n, rep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = f"ours/{net}_{n}_rep{rep}.csv"
os.makedirs("ours", exist_ok=True)
if os.path.exists(out): sys.exit(0)

df = pd.read_csv(f"rbn/{net}_{n}_rep{rep}.csv", sep=";")
X = np.array(df, dtype=float)
X = (X - X.mean(0)) / X.std(0, ddof=0)

# ground truth in the SAME column order as the rbn file
_, mat_t, cols = bnsim.load_network(net)
order = [cols.index(c) for c in df.columns]
mat_t = mat_t[np.ix_(order, order)]

t = time.time()
layers = bnsim.map_layering(X, seed=1000 * n + rep)
rows = []
for gic in (1, 2):
    m = ss_select.edges_ss(X, layers, variant="one", gic=gic)
    s = bnsim.score(m, mat_t)
    rows.append([net, n, rep, f"GIC{gic}", s["power"], s["fdr"],
                 cpdag.shd_cpdag(m, mat_t), s["shd"]])
with open(out, "w") as f:
    for r in rows: f.write(",".join(str(x) for x in r) + "\n")
print(f"{net} n={n} rep={rep} {time.time()-t:.0f}s", flush=True)
