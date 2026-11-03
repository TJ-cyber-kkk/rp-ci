"""Single scalar CIT test with plus-one reference calibration."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import ArrayLike

from . import calibration
from .rosenblatt import _validate_cit_matrix, _validate_h_scale, rosenblatt_transform
from .statistic import rho_hat_multivariate


@dataclass(frozen=True)
class CITResult:
    """statistic and critical_value are on the n*rho_hat scale."""

    reject: bool
    pvalue: float
    statistic: float
    critical_value: float
    alpha: float
    reference_size: int


def cit_test(
    X: ArrayLike,
    Y: ArrayLike,
    Z: ArrayLike,
    *,
    alpha: float = 0.05,
    h_scale: float = 0.8,
    null_pool: ArrayLike | None = None,
    B: int = 2000,
    seed: int | None = None,
    n_jobs: int = 1,
) -> CITResult:
    """Scalar CIT with plus-one calibration; no random projections.

    X and Y are (n,) or (n,1); Z is (n,) or (n,dz), with n>=2.
    A supplied 1-D pool overrides B, seed and n_jobs; it must be simulated for
    the same n and dz, independently of the data. Otherwise seed controls the
    reference simulation.
    Plus-one ranking alone does not ensure finite-sample validity.
    """
    level = np.asarray(alpha, dtype=float)
    if level.ndim != 0 or not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("alpha must be a finite scalar in (0,1).")
    alpha = float(level)
    _validate_h_scale(h_scale)
    X = _validate_cit_matrix(X, "X")
    Y = _validate_cit_matrix(Y, "Y")
    Z = _validate_cit_matrix(Z, "Z")
    n = X.shape[0]
    if X.shape[1] != 1 or Y.shape[1] != 1:
        raise ValueError("cit_test requires scalar X and Y; use rp_test for vectors.")
    if Y.shape[0] != n or Z.shape[0] != n:
        raise ValueError("X, Y, Z must have the same number of rows (n).")

    if null_pool is None:
        if not isinstance(B, (int, np.integer)) or isinstance(B, (bool, np.bool_)) or B < 1:
            raise ValueError("B must be a positive integer.")
        if seed is not None and (
            not isinstance(seed, (int, np.integer))
            or isinstance(seed, (bool, np.bool_)) or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None.")
        if not isinstance(n_jobs, (int, np.integer)) or isinstance(n_jobs, (bool, np.bool_)) or n_jobs == 0:
            raise ValueError("n_jobs must be a nonzero integer.")
        reference_size = int(B)
    else:
        null_pool = np.asarray(null_pool, dtype=float)
        if null_pool.ndim != 1 or null_pool.size == 0 or not np.isfinite(null_pool).all():
            raise ValueError("null_pool must be a nonempty, finite 1-D array.")
        reference_size = int(null_pool.size)

    if 1.0 / (reference_size + 1) > alpha:
        warnings.warn(
            "The minimum plus-one p-value 1/(B+1) exceeds alpha; "
            "this test cannot reject. The reference budget is unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )

    U, V, W = rosenblatt_transform(X, Y, Z, h_scale=h_scale)
    statistic = rho_hat_multivariate(U, V, W=W)
    if not np.isfinite(statistic):
        raise ValueError("Observed statistic must be finite.")
    if null_pool is None:
        null_pool = calibration.simulate_null_pool(
            n, 1, 1, Z.shape[1], B=B, seed=seed, n_jobs=n_jobs
        )
    pvalue = float(calibration.plus_one_pvalues(statistic, null_pool))
    critical_value = calibration.plus_one_critical_value(null_pool, alpha)
    return CITResult(pvalue <= alpha, pvalue, statistic, critical_value, alpha, reference_size)
