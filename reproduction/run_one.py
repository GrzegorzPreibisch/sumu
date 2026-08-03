"""Compute and cache one (network, n, replication) layering.  usage: run_one.py NET N REP"""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bnsim

net, n, rep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs("cache", exist_ok=True)
out = f"cache/{net}_{n}_rep{rep}.npz"
if os.path.exists(out):
    sys.exit(0)

pool, mat_t, _ = bnsim.load_network(net)
B, b0, sd = bnsim.fit_gaussian_bn(pool, mat_t)
order = bnsim.topo_order(mat_t)

seed = abs(hash((net, n, rep))) % (2 ** 31)
X = bnsim.simulate(B, b0, sd, order, n, np.random.default_rng(seed))
X = (X - X.mean(0)) / X.std(0, ddof=0)

t = time.time()
layers = bnsim.map_layering(X, seed=seed % (2 ** 31))
np.savez_compressed(out, X=X.astype(np.float32),
                    layers=json.dumps(layers), secs=time.time() - t)
print(f"{net} n={n} rep={rep}: {time.time()-t:.0f}s, {len(layers)} layers", flush=True)
