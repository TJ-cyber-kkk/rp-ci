"""Scalar generalized covariance measure (GCM) test."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from ..projections import _as_2d
from .regression import ols_residuals

_RTOL = 1e-8
Residualizer = Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]


@dataclass(frozen=True)
class GCMResult:
    reject: bool
    pvalue: float
    statistic: float
    alpha: float


def _residuals(residualize: Residualizer, V, Z) -> NDArray[np.floating] | None:
    """Residual vector, or None if it is numerically zero relative to V."""
    R = _as_2d(residualize(V, Z))
    if R.shape != V.shape or not np.isfinite(R).all():
        raise ValueError("residualize must return finite residuals with V's shape.")
    if R.std() <= _RTOL * V.std():
        return None
    return R[:, 0]


def gcm_test(
    X: ArrayLike,
    Y: ArrayLike,
    Z: ArrayLike,
    *,
    alpha: float = 0.05,
    residualize: Residualizer = ols_residuals,
) -> GCMResult:
    """GCM test of scalar X independent of scalar Y given Z (Shah and Peters, 2020).

    T = sqrt(n) mean(RS) / sd(RS) for residuals R, S of X, Y on Z, with a
    two-sided normal p-value. p = 1 when a residual is numerically zero
    relative to its variable (e.g. dz >= n - 1) or RS is constant.
    X and Y are (n,) or (n,1); Z is (n,) or (n,dz), possibly with dz=0.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1).")
    X, Y, Z = _as_2d(X), _as_2d(Y), _as_2d(Z)
    n = X.shape[0]
    if Y.shape[0] != n or Z.shape[0] != n:
        raise ValueError("X, Y, Z must have the same number of rows (n).")
    if n < 3:
        raise ValueError("Need at least n=3 observations.")
    if X.shape[1] != 1 or Y.shape[1] != 1:
        raise ValueError("gcm_test requires scalar X and Y.")

    R, S = _residuals(residualize, X, Z), _residuals(residualize, Y, Z)
    if R is None or S is None:
        return GCMResult(False, 1.0, 0.0, float(alpha))
    product = R * S
    mean = product.mean()
    sd = np.sqrt(np.mean((product - mean) ** 2))
    if sd == 0.0:
        return GCMResult(False, 1.0, 0.0, float(alpha))
    statistic = float(np.sqrt(n) * mean / sd)
    pvalue = float(2.0 * norm.sf(abs(statistic)))
    return GCMResult(pvalue <= alpha, pvalue, statistic, float(alpha))
