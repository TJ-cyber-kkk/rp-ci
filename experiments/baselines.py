"""Baseline CI tests in the configurations of the paper (Appendix C.2).

Each test takes X (n, dx), Y (n, dy), Z (n, dz) and an integer seed, and returns a p-value,
or (p-value, statistic) with return_statistic=True. An empty Z (dz = 0) gives the
unconditional version of the test.
"""
from __future__ import annotations

import numpy as np

from rp_ci.gcm import ols_residuals


def _standardize(v):
    return (v - v.mean(axis=0)) / (v.std(axis=0, ddof=1) + 1e-10)


def _nonconstant(v):
    """Drop constant columns, which causal-learn's standardization would divide by zero."""
    keep = v.std(axis=0) > 1e-12
    return v if keep.all() else v[:, keep]


def _with_global_seed(seed, fn):
    # causal-learn draws from NumPy's global generator.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return fn()
    finally:
        np.random.set_state(state)


def _output(pvalue, statistic, return_statistic):
    return (float(pvalue), float(statistic)) if return_statistic else float(pvalue)


def kci(x, y, z, seed, return_statistic=False):
    """KCI: Gaussian kernels, median bandwidth, Gamma approximation."""
    from causallearn.utils.KCI.KCI import KCI_CInd, KCI_UInd

    if z.shape[1] == 0:
        test = KCI_UInd(kernelX="Gaussian", kernelY="Gaussian", est_width="median", approx=True)
        return _output(*test.compute_pvalue(x, y), return_statistic)
    test = KCI_CInd(kernelX="Gaussian", kernelY="Gaussian", kernelZ="Gaussian",
                    est_width="median", approx=True, nullss=1000)
    return _output(*_with_global_seed(seed, lambda: test.compute_pvalue(x, y, z)), return_statistic)


def rcit(x, y, z, seed, return_statistic=False):
    """RCIT: 100 conditioning features, 10 tested-variable features, lpd4; RIT if Z is empty."""
    from causallearn.utils.RCIT.RCIT import RCIT, RIT

    x, y, z = _nonconstant(x), _nonconstant(y), _nonconstant(z)
    if z.shape[1] == 0:
        test, args = RIT(approx="lpd4"), (x, y)
    else:
        test, args = RCIT(approx="lpd4", num_f=100, num_f2=10, rcit=True), (x, y, z)
    return _output(*_with_global_seed(seed, lambda: test.compute_pvalue(*args)), return_statistic)


def cmiknn(x, y, z, seed, return_statistic=False):
    """CMIknn: 10 nearest neighbors, 5 shuffle neighbors, 100 shuffles, standardized inputs."""
    from tigramite.data_processing import DataFrame
    from tigramite.independence_tests.cmiknn import CMIknn

    test = CMIknn(knn=10, shuffle_neighbors=5, significance="shuffle_test",
                  transform="standardize", workers=1, permute="Y", sig_samples=100, seed=seed)
    test.set_dataframe(DataFrame(data=_standardize(np.hstack([x, y, z]))))
    dx, dy, dz = x.shape[1], y.shape[1], z.shape[1]
    cols = [[(j, 0) for j in range(a, b)] for a, b in
            ((0, dx), (dx, dx + dy), (dx + dy, dx + dy + dz))]
    statistic, pvalue = test.run_test(X=cols[0], Y=cols[1], Z=cols[2])[:2]
    return _output(pvalue, statistic, return_statistic)


def max_gcm(x, y, z, seed, draws=999, block=4096, return_statistic=False):
    """Coordinate-max GCM: in-sample OLS residuals, Gaussian multiplier calibration."""
    rx, ry = ols_residuals(x, z), ols_residuals(y, z)
    n, dy, pairs = rx.shape[0], ry.shape[1], rx.shape[1] * ry.shape[1]
    multipliers = np.random.default_rng(seed).normal(size=(draws, n))
    statistic, bootstrap = -np.inf, np.zeros(draws)
    # Blocks of coordinate pairs bound memory; all blocks share the multipliers.
    for start in range(0, pairs, block):
        columns = np.arange(start, min(start + block, pairs))
        products = rx[:, columns // dy] * ry[:, columns % dy]
        centered = products - products.mean(axis=0)
        scales = np.sqrt(np.mean(centered**2, axis=0))
        valid = np.isfinite(scales) & (scales > 1e-12)
        if valid.any():
            standardized = centered[:, valid] / scales[valid]
            statistic = max(statistic, np.max(np.abs(np.sqrt(n) * products.mean(axis=0)[valid] / scales[valid])))
            bootstrap = np.maximum(bootstrap, np.max(np.abs(multipliers @ standardized), axis=1) / np.sqrt(n))
    if statistic == -np.inf:
        return _output(1.0, 0.0, return_statistic)
    return _output((1.0 + np.sum(bootstrap >= statistic)) / (draws + 1.0), statistic, return_statistic)


BASELINES = {"kci": kci, "cmiknn": cmiknn, "rcit": rcit, "max_gcm": max_gcm}
