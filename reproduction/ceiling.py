"""How much of the remaining error is the layering (sumu) and how much is selection?

For a fixed layering, the best any selector can do is pick exactly those true edges
whose parent sits in a strictly earlier layer. That oracle has FDR 0 and
SHD = (number of true edges that are unreachable under the layering).

Note adding the *reverse* of an unreachable edge never helps: a missing edge and a
reversed edge both cost 1 SHD, and the reversal also costs a false positive. So the
strict oracle is simultaneously optimal for power, FDR and SHD.
"""
import csv, glob, json, os, collections, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bnsim

COLS = ["net", "n", "rep", "label", "ceiling", "power", "fdr", "shd",
        "TP", "FP", "FN", "n_est"]

# measured runs
runs = collections.defaultdict(list)
for f in glob.glob("eval/*.csv"):
    for r in csv.reader(open(f)):
        d = dict(zip(COLS, r))
        for k in COLS[4:]:
            d[k] = float(d[k])
        runs[(d["net"], int(d["n"]), d["label"])].append(d)

# oracle, computed straight from the cached layerings
oracle = collections.defaultdict(list)
truth = {}
for path in sorted(glob.glob("cache/*.npz")):
    base = os.path.basename(path)[:-4]
    net, n, rep = base.rsplit("_", 2)
    n = int(n)
    if net not in truth:
        truth[net] = bnsim.load_network(net)[1]
    mat_t = truth[net]
    layers = json.loads(str(np.load(path, allow_pickle=True)["layers"]))
    depth = {v: i for i, l in enumerate(layers) for v in l}

    mat_o = np.zeros_like(mat_t)
    for c in range(mat_t.shape[0]):
        for pa in np.nonzero(mat_t[c])[0]:
            if depth.get(int(pa), 1 << 30) < depth.get(c, -1):
                mat_o[c, int(pa)] = 1.0
    oracle[(net, n)].append(bnsim.score(mat_o, mat_t))

NTRUE = {"ecoli70": 70, "magic-niab": 66, "magic-irri": 102}
CELLS = [("ecoli70", 300), ("ecoli70", 1000), ("magic-niab", 300),
         ("magic-niab", 1000), ("magic-irri", 300), ("magic-irri", 1000)]

CS = [1, 2, 3, 4, 5, 6, 8]


def best_cfg(cell, key, better=min):
    """The GIC penalty that does best on `key` for this cell (tuning c alone)."""
    out = []
    for c in CS:
        v = runs.get((cell[0], cell[1], f"fixed_c{c}"))
        if v:
            out.append((np.mean([d[key] for d in v]), c))
    return better(out) if out else (None, None)


print("Power reachable by tuning the GIC penalty alone, vs the oracle ceiling\n")
hh = f"{'network':<11}{'n':>5} | " + "".join(f"{'c='+str(c):>7}" for c in CS) + f" | {'ORACLE':>8}"
print(hh); print("-" * len(hh))
for cell in CELLS:
    if not oracle.get(cell):
        continue
    row = ""
    for c in CS:
        v = runs.get((cell[0], cell[1], f"fixed_c{c}"))
        row += f"{np.mean([d['power'] for d in v]):>7.2f}" if v else f"{'-':>7}"
    print(f"{cell[0]:<11}{cell[1]:>5} | {row} | "
          f"{np.mean([o['power'] for o in oracle[cell]]):>8.2f}")

print("\n\nORACLE = perfect selection given the layering sumu actually produced\n")
h = (f"{'network':<11}{'n':>5} | {'power':>6}{'oracle':>7}{'ceiling%':>9} | "
     f"{'shd':>6}{'oracle':>7} | {'SHD lost to':>12}{'SHD lost to':>13}")
print(h)
print(f"{'':<11}{'':>5} | {'now':>6}{'max':>7}{'reached':>9} | {'now':>6}{'min':>7} | "
      f"{'LAYERING':>12}{'SELECTION':>13}")
print("-" * len(h))

tot_lay, tot_sel = [], []
for cell in CELLS:
    orc = oracle.get(cell)
    bs, bc = best_cfg(cell, "shd", min)
    got = runs.get((cell[0], cell[1], f"fixed_c{bc}"))
    if not orc or not got:
        continue
    p_max = np.mean([o["power"] for o in orc])
    s_min = np.mean([o["shd"] for o in orc])
    p_now = np.mean([d["power"] for d in got])
    s_now = np.mean([d["shd"] for d in got])
    lay, sel = s_min, s_now - s_min
    tot_lay.append(lay); tot_sel.append(sel)
    print(f"{cell[0]:<11}{cell[1]:>5} | {p_now:>6.2f}{p_max:>7.2f}{p_now/p_max*100:>8.0f}% | "
          f"{s_now:>6.1f}{s_min:>7.1f} | {lay:>11.1f} ({lay/s_now*100:>3.0f}%)"
          f"{sel:>7.1f} ({sel/s_now*100:>3.0f}%)")

print(f"\noverall: {np.sum(tot_lay)/(np.sum(tot_lay)+np.sum(tot_sel))*100:.0f}% of SHD "
      f"is the layering, {np.sum(tot_sel)/(np.sum(tot_lay)+np.sum(tot_sel))*100:.0f}% is selection")

print("\n\nPOWER BUDGET  (share of all true edges)\n")
h2 = (f"{'network':<11}{'n':>5} | {'found':>7}{'lost: bad':>11}{'lost: sel':>11} | "
      f"{'headroom if selection were perfect':>36}")
print(h2); print("-" * len(h2))
for cell in CELLS:
    orc = oracle.get(cell)
    bs, bc = best_cfg(cell, "shd", min)
    got = runs.get((cell[0], cell[1], f"fixed_c{bc}"))
    if not orc or not got:
        continue
    p_max = np.mean([o["power"] for o in orc])
    p_now = np.mean([d["power"] for d in got])
    print(f"{cell[0]:<11}{cell[1]:>5} | {p_now:>6.0%} {1-p_max:>10.0%} {p_max-p_now:>10.0%} | "
          f"{'+' + format(p_max - p_now, '.0%') + ' power, then hard-capped at ' + format(p_max, '.0%'):>36}")
