import json, warnings, numpy as np, sys
warnings.filterwarnings("ignore")
import bnsim, ss_select

TGT = {("ecoli70",1000):{1:(.76,.23,23), 2:(.74,.17,15)},
       ("ecoli70",300): {1:(.69,.26,30), 2:(.67,.20,21)}}
for net, n in [("ecoli70",1000),("ecoli70",300)]:
    _, mat_t, _ = bnsim.load_network(net)
    accum = {}
    for rep in range(3):
        d = np.load(f"cache/{net}_{n}_rep{rep}.npz", allow_pickle=True)
        X = d["X"].astype(float); layers = json.loads(str(d["layers"]))
        for gic in (1,2):
            for var in ("bez","one","iter"):
                m = ss_select.edges_ss(X, layers, variant=var, gic=gic)
                s = bnsim.score(m, mat_t)
                accum.setdefault((gic,var), []).append(s)
    print(f"\n=== {net} n={n}  (3 powt.) ===")
    print(f"  {'wariant':<16}{'moc':>7}{'fdr':>7}{'shd':>7}   | cel (wyniki.ods)")
    for gic in (1,2):
        t = TGT[(net,n)][gic]
        for var in ("bez","one","iter"):
            v = accum[(gic,var)]
            p,f,s = (np.mean([x[k] for x in v]) for k in ("power","fdr","shd"))
            print(f"  GIC{gic} {var:<11}{p:>7.2f}{f:>7.2f}{s:>7.1f}   | GIC{gic} {t[0]:.2f}/{t[1]:.2f}/{t[2]}")
