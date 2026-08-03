"""Fallback for `glmnet.ElasticNet`, used when python-glmnet is not installed.

python-glmnet no longer builds on current toolchains: its setup.py depends on
numpy.distutils, which was removed in numpy 1.26.  Rather than pin the whole
project to an old numpy, this provides the small slice of the glmnet API that
`Gadget.TL` uses -- `fit`, `coef_path_`, `intercept_path_` -- on top of
scikit-learn's coordinate-descent path.

The parameterisations agree: glmnet and sklearn both minimise
    (1/2n)||y - Xb||^2 + lam*||b||_1
so the lambda grids coincide.  Defaults mirror python-glmnet's: pure lasso,
100 log-spaced lambdas down from the smallest one giving b=0, min_lambda_ratio
1e-4 when n>p else 1e-2, internal standardisation, intercept per lambda.

If python-glmnet IS installed, sumu uses it and this module is ignored, so
results can be checked against the original solver.
"""

import numpy as np
from sklearn.linear_model import enet_path

__all__ = ["ElasticNet"]


class ElasticNet:
    def __init__(self, alpha=1.0, n_lambda=100, min_lambda_ratio=None,
                 standardize=True, fit_intercept=True, **kwargs):
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.min_lambda_ratio = min_lambda_ratio
        self.standardize = standardize
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, p = X.shape

        x_mean = X.mean(axis=0) if self.fit_intercept else np.zeros(p)
        y_mean = y.mean() if self.fit_intercept else 0.0
        Xc = X - x_mean

        if self.standardize:
            x_sd = np.sqrt((Xc ** 2).mean(axis=0))
            x_sd[x_sd < 1e-12] = 1.0
        else:
            x_sd = np.ones(p)

        eps = self.min_lambda_ratio
        if eps is None:
            eps = 1e-4 if n > p else 1e-2

        lambdas, coefs, _ = enet_path(Xc / x_sd, y - y_mean,
                                      l1_ratio=self.alpha, eps=eps,
                                      n_alphas=self.n_lambda)

        coefs = coefs / x_sd[:, None]            # back to the original scale
        self.lambda_path_ = lambdas
        self.coef_path_ = coefs                  # (p, n_lambda)
        self.intercept_path_ = y_mean - x_mean @ coefs
        return self
