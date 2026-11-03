"""Nuisance regression for the GCM base test."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import lstsq

from ..projections import _as_2d


def ols_residuals(V: ArrayLike, Z: ArrayLike) -> NDArray[np.floating]:
    """In-sample OLS residuals of each column of V on Z, with an intercept.

    A Z without columns gives centered V. Z is standardized before the solve,
    which leaves the residuals unchanged.
    """
    V = _as_2d(V)
    Z = _as_2d(Z)
    if Z.shape[0] != V.shape[0]:
        raise ValueError("V and Z must have the same number of rows.")
    if not (np.isfinite(V).all() and np.isfinite(Z).all()):
        raise ValueError("V and Z must be finite.")
    v_mean = V.mean(axis=0)
    if Z.shape[1] == 0:
        return V - v_mean
    scale = Z.std(axis=0)
    scale[scale == 0.0] = 1.0
    Zs = (Z - Z.mean(axis=0)) / scale
    zs_mean = Zs.mean(axis=0)
    coef = lstsq(Zs - zs_mean, V - v_mean)[0]
    return V - (Zs @ coef + (v_mean - zs_mean @ coef))
