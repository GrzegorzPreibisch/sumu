"""Aggregate eval/*.csv and compare against wyniki.ods and the thesis tables."""
import csv, glob, collections, sys
import numpy as np

COLS = ["net", "n", "rep", "label", "ceiling", "power", "fdr", "shd",
        "TP", "FP", "FN", "n_est"]

rows = []
for f in glob.glob("eval/*.csv"):
    for r in csv.reader(open(f)):
        d = dict(zip(COLS, r))
        for k in COLS[4:]:
            d[k] = float(d[k])
        d["n"] = int(d["n"])
        rows.append(d)

if not rows:
    sys.exit("no eval rows yet")

agg = collections.defaultdict(list)
for d in rows:
    agg[(d["net"], d["n"], d["label"])].append(d)

# --- published numbers -------------------------------------------------------
WYNIKI = {  # (net, n) -> {GIC1: (power,fdr,shd), GIC2: ...}
    ("ecoli70", 300):    {"GIC1": (.69, .26, 30), "GIC2": (.67, .20, 21)},
    ("ecoli70", 1000):   {"GIC1": (.76, .23, 23), "GIC2": (.74, .17, 15)},
    ("magic-irri", 300): {"GIC1": (.35, .26, 73), "GIC2": (.25, .23, 80)},
    ("magic-irri", 1000):{"GIC1": (.62, .24, 48), "GIC2": (.54, .19, 49)},
    ("arth150", 300):    {"GIC1": (.53, .32, 93), "GIC2": (.48, .24, 91)},
    ("arth150", 1000):   {"GIC1": (.65, .29, 81), "GIC2": (.64, .23, 72)},
    ("magic-niab", 300): {"GIC1": (.35, .27, 45), "GIC2": (.21, .27, 53)},
}
THESIS = {  # (net, n) -> (power, shd) as printed in the dissertation
    ("ecoli70", 300): (.49, 39.2),   ("ecoli70", 1000): (.55, 35.1),
    ("magic-irri", 300): (.24, 85.9),("magic-irri", 1000): (.38, 79.3),
    ("arth150", 300): (.38, 103.0),  ("arth150", 1000): (.46, 98.7),
}

NETS = ["ecoli70", "magic-niab", "magic-irri", "arth150"]
order = [(nt, n) for nt in NETS for n in (300, 1000)]


def mean(key, cell, label):
    v = agg.get((cell[0], cell[1], label))
    return (np.mean([d[key] for d in v]), len(v)) if v else (None, 0)


print("=" * 96)
print("A. DOES THE BUGGY CODE REPRODUCE THE THESIS TABLES?")
print("   config: RSS criterion, no refit, threshold from positive coefficients only")
print("=" * 96)
print(f"{'network':<12}{'n':>6}{'reps':>6} | {'power repro':>12}{'power thesis':>14} | "
      f"{'shd repro':>11}{'shd thesis':>12}")
for cell in order:
    if cell not in THESIS:
        continue
    p, k = mean("power", cell, "code2022_buggy")
    s, _ = mean("shd", cell, "code2022_buggy")
    if p is None:
        continue
    tp, ts = THESIS[cell]
    print(f"{cell[0]:<12}{cell[1]:>6}{k:>6} | {p:>12.2f}{tp:>14.2f} | {s:>11.1f}{ts:>12.1f}")

print()
print("=" * 96)
print("B. DOES THE FIXED CODE REPRODUCE wyniki.ods?")
print("=" * 96)
for gic, label in (("GIC1", "fixed_c2"), ("GIC2", "fixed_c5")):
    print(f"\n--- {gic}  (reproduced with {label}) ---")
    print(f"{'network':<12}{'n':>6}{'reps':>6} | {'power':>7}{'target':>8} | "
          f"{'fdr':>7}{'target':>8} | {'shd':>7}{'target':>8} | {'ceiling':>8}")
    for cell in order:
        if cell not in WYNIKI or gic not in WYNIKI[cell]:
            continue
        p, k = mean("power", cell, label)
        if p is None:
            continue
        fd, _ = mean("fdr", cell, label)
        sh, _ = mean("shd", cell, label)
        ce, _ = mean("ceiling", cell, label)
        tp, tf, ts = WYNIKI[cell][gic]
        print(f"{cell[0]:<12}{cell[1]:>6}{k:>6} | {p:>7.2f}{tp:>8.2f} | "
              f"{fd:>7.2f}{tf:>8.2f} | {sh:>7.1f}{ts:>8.0f} | {ce:>8.2f}")

print()
print("=" * 96)
print("C. EFFECT OF THE SIGN BUG ALONE (identical code path either way)")
print("=" * 96)
print(f"{'network':<12}{'n':>6} | {'power buggy':>12}{'power fixed':>12}{'delta':>8} | "
      f"{'shd buggy':>11}{'shd fixed':>11}")
for cell in order:
    pb, k = mean("power", cell, "code2022_buggy")
    pf, _ = mean("power", cell, "code2022_signfixed")
    if pb is None or pf is None:
        continue
    sb, _ = mean("shd", cell, "code2022_buggy")
    sf, _ = mean("shd", cell, "code2022_signfixed")
    print(f"{cell[0]:<12}{cell[1]:>6} | {pb:>12.2f}{pf:>12.2f}{pf-pb:>+8.2f} | "
          f"{sb:>11.1f}{sf:>11.1f}")

print()
print("=" * 96)
print("D. FULL GIC PENALTY SWEEP (fixed code)")
print("=" * 96)
labels = [f"fixed_c{c}" for c in (1, 2, 3, 4, 5, 6, 8)]
for cell in order:
    got = [l for l in labels if agg.get((cell[0], cell[1], l))]
    if not got:
        continue
    tgt = WYNIKI.get(cell, {})
    t1 = tgt.get("GIC1"); t2 = tgt.get("GIC2")
    extra = ""
    if t1 and t2:
        extra = f"   [GIC1 {t1[0]:.2f}/{t1[1]:.2f}/{t1[2]}  GIC2 {t2[0]:.2f}/{t2[1]:.2f}/{t2[2]}]"
    print(f"\n{cell[0]} n={cell[1]}{extra}")
    for l in got:
        p, k = mean("power", cell, l)
        fd, _ = mean("fdr", cell, l)
        sh, _ = mean("shd", cell, l)
        print(f"   {l:<10} power={p:.2f}  fdr={fd:.2f}  shd={sh:6.1f}   (reps={k})")
