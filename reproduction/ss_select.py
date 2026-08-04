"""Faithful Python port of Wojciech Rejchel's SSnet4lm_2.R (screening-selection).

This is NOT "pick lambda by BIC, then threshold". The procedure is:

  1. lasso path, 100 lambdas log-spaced from lmax = max|2 X'y|/n down by 1e-4
  2. for EVERY lambda on the path, order its selected predictors by |beta|
     descending and build the nested family top-1, top-2, ... top-k
  3. QR-orthogonalise to get the exact OLS R^2 of every nested model
  4. for each model SIZE, keep the best R^2 achieved by ANY lambda
  5. GIC over that size-indexed family picks the final model

Penalty, matching Wojtek's mail:
    GIC1  lambda^2 = 2.5 * sigma^2 * log(#regressors)                 -> pen = 1
    GIC2  lambda^2 = 2.5 * sigma^2 * log(#regressors * L * |l_j|)     -> pen = min(L*|l_j|, d)

Three sigma treatments, all returned by the R function:
    'bez'  sigma unknown, scale-free:  n*log(1-R2) + k*2.5*log(p*pen)
    '1'    sigma from the largest model, one shot
    'iter' sigma re-estimated, up to `iter` rounds
"""

import numpy as np
from sklearn.linear_model import lasso_path


def ss_select(X, y, pen=1.0, n_lambda=100, nmin=1e-4, iters=10, kappa=2.5):
    """Return dict of supports (index arrays into X's columns) for each variant."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    y = y - y.mean()
    n, p = X.shape
    if p == 0:
        return {k: np.array([], int) for k in ("bez", "one", "iter")}

    # Xg: columns rescaled so that ||x||^2 = n  (R: sqrt(n/sum(x^2))*x)
    nrm = np.sqrt((X ** 2).sum(axis=0))
    nrm[nrm < 1e-12] = 1.0
    Xg = X * (np.sqrt(n) / nrm)

    lmax = np.abs(2.0 * Xg.T @ y).max() / n
    if lmax <= 0:
        return {k: np.array([], int) for k in ("bez", "one", "iter")}
    lambdas = np.exp(np.linspace(np.log(lmax), np.log(lmax * nmin), n_lambda))

    # glmnet(alpha=1, intercept=FALSE) with an explicit lambda grid
    _, coefs, _ = lasso_path(Xg, y, alphas=lambdas)          # (p, n_lambda)

    df = (coefs != 0).sum(axis=0)
    keep = np.where((df < n) & (df > 0))[0]
    if len(keep) == 0:
        return {k: np.array([], int) for k in ("bez", "one", "iter")}
    B = np.abs(coefs[:, keep])

    # drop lambdas giving a duplicate support (R: duplicated(t(BB>0)))
    seen, cols = set(), []
    for j in range(B.shape[1]):
        key = tuple(np.nonzero(B[:, j])[0])
        if key not in seen:
            seen.add(key)
            cols.append(j)
    B = B[:, cols]

    tss = float(y @ y)
    if tss <= 0:
        return {k: np.array([], int) for k in ("bez", "one", "iter")}

    # nested OLS R^2 along each lambda's |beta|-ordering, via QR
    models = []
    for j in range(B.shape[1]):
        b = B[:, j]
        sel = np.nonzero(b)[0]
        order = sel[np.argsort(-b[sel])]
        Q, _ = np.linalg.qr(X[:, order])
        q2 = (Q.T @ y) ** 2
        rss = tss - np.cumsum(q2)
        models.append((1.0 - rss / tss, order))

    maxl = max(len(m[0]) for m in models)
    r2 = np.zeros((maxl, len(models)))
    for j, (R2j, _) in enumerate(models):
        r2[:len(R2j), j] = R2j

    which = r2.argmax(axis=1)
    R2 = r2.max(axis=1)

    s = int(min(p, np.ceil(n / 2), len(models), len(R2)))
    if s < 1:
        return {k: np.array([], int) for k in ("bez", "one", "iter")}
    R2 = R2[:s]
    supports = [np.sort(models[which[i]][1][:i + 1]) for i in range(s)]

    k = np.arange(1, s + 1)
    penalty = kappa * np.log(max(p * pen, 2.0))
    R2 = np.clip(R2, None, 1 - 1e-12)

    # sigma unknown, scale-free; the appended 0 is the empty model
    crit_bez = np.append(n * np.log(1 - R2) + k * penalty, 0.0)
    k_bez = int(crit_bez.argmin())

    # sigma from the largest model
    denom = max(n - s, 1)
    crit_one = np.append((1 - R2) + k * penalty / denom * (1 - R2[s - 1]), 1.0)
    k_one = int(crit_one.argmin())

    # sigma re-estimated
    k_it, R2i, si = k_one, R2, s
    for _ in range(iters):
        c = np.append((1 - R2i) + np.arange(1, si + 1) * penalty / max(n - si, 1)
                      * (1 - R2i[si - 1]), 1.0)
        kk = int(c.argmin())
        if kk == si:                       # empty model chosen -> stop
            k_it = kk
            break
        if kk == k_it:
            break
        k_it, si, R2i = kk, kk, R2i[:kk]

    def sup(idx):
        return np.array([], int) if idx >= s else supports[idx]

    return {"bez": sup(k_bez), "one": sup(k_one), "iter": sup(k_it)}


def edges_ss(X, layers, variant="one", gic=1, kappa=2.5):
    """Walk the layering; regress each node on all strictly earlier layers."""
    p = X.shape[1]
    n_layers = len(layers)
    mat_e = np.zeros((p, p))
    available = list(layers[0])
    for i in range(1, n_layers):
        if available:
            Z = X[:, available]
            pen = 1.0 if gic == 1 else float(min(n_layers * len(layers[i]), p))
            for v in layers[i]:
                if len(available) == 1:
                    # R: single regressor -> t-test at 0.01, no lasso
                    x = X[:, available[0]] - X[:, available[0]].mean()
                    yv = X[:, v] - X[:, v].mean()
                    denom = float(x @ x)
                    if denom > 0:
                        beta = float(x @ yv) / denom
                        resid = yv - beta * x
                        dof = max(len(yv) - 2, 1)
                        se = np.sqrt(float(resid @ resid) / dof / denom)
                        if se > 0:
                            from scipy import stats
                            pval = 2 * (1 - stats.t.cdf(abs(beta / se), dof))
                            if pval < 0.01:
                                mat_e[v, available[0]] = 1.0
                else:
                    Zs = (Z - Z.mean(0)) / np.where(Z.std(0) < 1e-12, 1, Z.std(0))
                    sel = ss_select(Zs, X[:, v], pen=pen, kappa=kappa)[variant]
                    for c in sel:
                        mat_e[v, available[int(c)]] = 1.0
        available = sorted(set(available) | set(layers[i]))
    return mat_e
