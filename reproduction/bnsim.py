"""Shared pieces: load a benchmark network, fit its Gaussian parameters, simulate,
run the layering stage, run the thresholded-lasso stage, score against ground truth."""

import os
import sys
import contextlib
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "datasets")

NETWORKS = {                    # name: (n_nodes, n_arcs) as documented
    "ecoli70":    (46, 70),
    "magic-niab": (44, 66),
    "magic-irri": (64, 102),
    "arth150":    (107, 150),
}


# --------------------------------------------------------------------------- #
# network + ground truth
# --------------------------------------------------------------------------- #

def load_network(name):
    """-> (pool array (N,p), true adjacency mat_t[child, parent], colnames)"""
    df = pd.read_csv(os.path.join(DATA_DIR, name + ".csv"), sep=";")
    colnames = list(df.columns)
    idx = {c: i for i, c in enumerate(colnames)}
    p = len(colnames)

    arcs = pd.read_csv(os.path.join(DATA_DIR, name + "_arcs.csv"), sep=";")
    mat_t = np.zeros((p, p))
    for frm, to in np.array(arcs, dtype=str):
        mat_t[idx[to], idx[frm]] = 1.0          # [child, parent]

    return np.array(df, dtype=float), mat_t, colnames


def topo_order(mat_t):
    """Kahn's algorithm on mat_t[child, parent]."""
    p = mat_t.shape[0]
    n_par = mat_t.sum(axis=1).astype(int)
    order, ready = [], [v for v in range(p) if n_par[v] == 0]
    while ready:
        v = ready.pop()
        order.append(v)
        for w in np.nonzero(mat_t[:, v])[0]:
            n_par[w] -= 1
            if n_par[w] == 0:
                ready.append(int(w))
    assert len(order) == p, "true graph is not a DAG"
    return order


def fit_gaussian_bn(pool, mat_t):
    """OLS-fit X_v = b0 + sum_{u in pa(v)} b_vu X_u + N(0, s_v^2) on the data pool.

    bnlearn ships these networks already fitted to real data; we recover an
    equivalent parameterisation so fresh replications can be simulated without R.
    """
    N, p = pool.shape
    B = np.zeros((p, p))            # B[child, parent]
    b0 = np.zeros(p)
    sd = np.zeros(p)
    for v in range(p):
        pa = np.nonzero(mat_t[v])[0]
        y = pool[:, v]
        if len(pa) == 0:
            b0[v] = y.mean()
            sd[v] = y.std(ddof=1)
            continue
        X = np.column_stack([np.ones(N), pool[:, pa]])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        b0[v] = coef[0]
        B[v, pa] = coef[1:]
        sd[v] = np.sqrt(resid @ resid / max(N - len(pa) - 1, 1))
    return B, b0, sd


def simulate(B, b0, sd, order, n, rng):
    p = len(b0)
    X = np.zeros((n, p))
    for v in order:
        pa = np.nonzero(B[v])[0]
        mu = b0[v] + (X[:, pa] @ B[v, pa] if len(pa) else 0.0)
        X[:, v] = mu + rng.normal(0.0, sd[v], n)
    return X


# --------------------------------------------------------------------------- #
# stage 1 - MAP layering via sumu's partition MCMC
# --------------------------------------------------------------------------- #

def map_layering(sample, seed=None):
    """Run Gadget and return its best-scoring partition as a list of lists."""
    import sumu
    if seed is not None:
        np.random.seed(seed)
    devnull = open(os.devnull, "w")
    with contextlib.redirect_stdout(devnull):
        data = sumu.Data(sample)
        g = sumu.Gadget(data=data, recurring=False,
                        logging={"logfile": devnull})
        g.sample()
        layers = [sorted(int(v) for v in part) for part in g.dags[0]]
    devnull.close()
    return layers


# --------------------------------------------------------------------------- #
# stage 2 - thresholded lasso, restricted to earlier layers
# --------------------------------------------------------------------------- #

def _lasso_path(X, y, n_lambda=100):
    """Lasso path, standardised design, intercept recovered per lambda."""
    from sklearn.linear_model import enet_path
    n, p = X.shape
    xm, ym = X.mean(0), y.mean()
    Xc = X - xm
    sdv = np.sqrt((Xc ** 2).mean(0))
    sdv[sdv < 1e-12] = 1.0
    eps = 1e-4 if n > p else 1e-2
    _, coefs, _ = enet_path(Xc / sdv, y - ym, l1_ratio=1.0,
                            eps=eps, n_alphas=n_lambda)
    coefs = coefs / sdv[:, None]
    return coefs, ym - xm @ coefs


def _rss(X, y, beta, intercept):
    r = y - X @ beta - intercept
    return float(r @ r)


def threshold_lasso(X, y, pen_bic, pen_gic, log_crit=True, refit=True,
                    buggy_sign=False):
    """Return the selected coefficient vector.

    log_crit  True  -> n*log(RSS/n) + pen*k   (thesis, scale-free)
              False -> RSS + pen*k            (as the 2022 Python code wrote it)
    refit     re-estimate by OLS on the retained support before scoring a
              threshold, which is what "thresholded lasso" means in
              Pokarowski & Mielniczuk; the 2022 code scored the shrunk
              coefficients instead.
    buggy_sign reproduces the 2022 defect: thresholds taken from positive
              coefficients only and compared without abs(), which deletes every
              negative coefficient.
    """
    n = len(y)
    coefs, intercepts = _lasso_path(X, y)

    def crit(rss, k, pen):
        if rss <= 0:
            rss = 1e-300
        return (n * np.log(rss / n) + pen * k) if log_crit else (rss + pen * k)

    # --- BIC over the lasso path -> beta_bic ---
    best, beta_bic, inter_bic = np.inf, np.zeros(X.shape[1]), 0.0
    for i in range(coefs.shape[1]):
        b, c = coefs[:, i], intercepts[i]
        v = crit(_rss(X, y, b, c), int(np.sum(b != 0)), pen_bic)
        if v < best:
            best, beta_bic, inter_bic = v, b, c

    # --- GIC over thresholds of beta_bic ---
    if buggy_sign:
        thresholds = np.sort(beta_bic[beta_bic > 0])
    else:
        thresholds = np.sort(np.abs(beta_bic[beta_bic != 0]))
    if len(thresholds) == 0:
        return np.zeros(X.shape[1])

    best, beta_gic = np.inf, np.zeros(X.shape[1])
    for delta in thresholds:
        b = beta_bic.copy()
        if buggy_sign:
            b[b < delta] = 0.0
        else:
            b[np.abs(b) < delta] = 0.0
        keep = np.nonzero(b)[0]
        if refit and len(keep):
            Z = np.column_stack([np.ones(n), X[:, keep]])
            coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
            b = np.zeros(X.shape[1])
            b[keep] = coef[1:]
            c = coef[0]
        else:
            c = inter_bic
        v = crit(_rss(X, y, b, c), len(keep), pen_gic)
        if v < best:
            best, beta_gic = v, b
    return beta_gic


def edges_from_layering(X, layers, pen_bic_fn, pen_gic_fn, **tl_kw):
    """Walk the layering; each node regresses on every node in earlier layers."""
    p = X.shape[1]
    mat_e = np.zeros((p, p))
    available = list(layers[0])
    for i in range(1, len(layers)):
        if not available:
            available = sorted(set(available) | set(layers[i]))
            continue
        Z = X[:, available]
        pb = pen_bic_fn(len(X), p, len(available))
        pg = pen_gic_fn(len(X), p, len(available))
        for v in layers[i]:
            beta = threshold_lasso(Z, X[:, v], pb, pg, **tl_kw)
            for k, u in enumerate(available):
                if beta[k] != 0:
                    mat_e[v, u] = 1.0
        available = sorted(set(available) | set(layers[i]))
    return mat_e


# --------------------------------------------------------------------------- #
# scoring - definitions taken from the 2022 compute_stats()
# --------------------------------------------------------------------------- #

def score(mat_e, mat_t):
    TP = float(np.sum(mat_t * mat_e))
    FP = float(np.sum((1 - mat_t) * mat_e))
    FN = float(np.sum(mat_t * (1 - mat_e)))
    power = TP / max(mat_t.sum(), 1)
    fdr = FP / max(TP + FP, 1)
    diff = np.abs(mat_t - mat_e)
    diff = diff + diff.T
    diff[diff > 1] = 1
    shd = float(np.sum(diff) / 2)
    return dict(power=power, fdr=fdr, shd=shd,
                TP=TP, FP=FP, FN=FN, n_est=float(mat_e.sum()))
