"""CPDAG conversion and CPDAG-based SHD, matching what bnlearn::shd computes.

bnlearn converts both graphs to their CPDAG (completed partially directed acyclic
graph, i.e. the Markov-equivalence class representative) and then counts the pairs
whose edge state differs. Edges whose orientation is not identifiable from the data
therefore cost nothing, which makes it strictly more forgiving than the directed
SHD the 2022 Python compute_stats produced.

Input adjacency convention is the project's: mat[child, parent] == 1.
"""

import numpy as np

NONE, UNDIR, FWD, BWD = 0, 1, 2, 3      # state of pair (i, j), FWD = i->j


def dag_to_cpdag(mat):
    """mat[child, parent] -> dict {(i,j): state} with i < j."""
    p = mat.shape[0]
    parents = [set(np.nonzero(mat[v])[0].tolist()) for v in range(p)]
    adj = [set() for _ in range(p)]
    for v in range(p):
        for u in parents[v]:
            adj[v].add(u)
            adj[u].add(v)

    # start from the skeleton, everything undirected
    state = {}
    for i in range(p):
        for j in adj[i]:
            if i < j:
                state[(i, j)] = UNDIR

    def orient(a, b):
        """record a -> b; return True if this changed anything"""
        key = (a, b) if a < b else (b, a)
        want = FWD if a < b else BWD
        if state.get(key) == UNDIR:
            state[key] = want
            return True
        return False

    def directed(a, b):
        key = (a, b) if a < b else (b, a)
        s = state.get(key, NONE)
        return s == (FWD if a < b else BWD)

    def undirected(a, b):
        key = (a, b) if a < b else (b, a)
        return state.get(key, NONE) == UNDIR

    # v-structures: a -> c <- b with a, b non-adjacent
    for c in range(p):
        pa = sorted(parents[c])
        for x in range(len(pa)):
            for y in range(x + 1, len(pa)):
                a, b = pa[x], pa[y]
                if b not in adj[a]:
                    orient(a, c)
                    orient(b, c)

    # Meek's rules R1-R3 (sufficient for DAG -> CPDAG)
    changed = True
    while changed:
        changed = False
        for (i, j), s in list(state.items()):
            if s != UNDIR:
                continue
            for a, b in ((i, j), (j, i)):
                # R1: c -> a, a - b, c and b non-adjacent  =>  a -> b
                if any(directed(c, a) and b not in adj[c]
                       for c in adj[a] if c != b):
                    changed |= orient(a, b)
                    break
                # R2: a -> c -> b, a - b  =>  a -> b
                if any(directed(a, c) and directed(c, b)
                       for c in adj[a] if c != b):
                    changed |= orient(a, b)
                    break
                # R3: a - c, a - d, c -> b, d -> b, c and d non-adjacent => a -> b
                cs = [c for c in adj[a] if c != b and undirected(a, c) and directed(c, b)]
                if any(d not in adj[c] for x, c in enumerate(cs) for d in cs[x + 1:]):
                    changed |= orient(a, b)
                    break
    return state


def shd_cpdag(mat_est, mat_true):
    """Number of pairs whose CPDAG edge state differs (bnlearn::shd)."""
    a = dag_to_cpdag(mat_est)
    b = dag_to_cpdag(mat_true)
    return float(sum(1 for k in set(a) | set(b)
                     if a.get(k, NONE) != b.get(k, NONE)))


def shd_directed(mat_est, mat_true):
    """The 2022 Python metric: symmetrised difference, reversal counted once."""
    diff = np.abs(mat_true - mat_est)
    diff = diff + diff.T
    diff[diff > 1] = 1
    return float(np.sum(diff) / 2)


if __name__ == "__main__":
    # chain 0->1->2 : no v-structure, so the whole chain is reversible
    m = np.zeros((3, 3)); m[1, 0] = 1; m[2, 1] = 1
    assert set(dag_to_cpdag(m).values()) == {UNDIR}, dag_to_cpdag(m)
    # collider 0->2<-1 with 0,1 non-adjacent : both edges compelled
    c = np.zeros((3, 3)); c[2, 0] = 1; c[2, 1] = 1
    assert set(dag_to_cpdag(c).values()) == {FWD}, dag_to_cpdag(c)
    # 0->1->2 vs 0<-1<-2 are Markov equivalent -> CPDAG SHD 0, directed SHD 2
    r = np.zeros((3, 3)); r[0, 1] = 1; r[1, 2] = 1
    assert shd_cpdag(m, r) == 0 and shd_directed(m, r) == 2
    print("cpdag.py self-checks pass")
