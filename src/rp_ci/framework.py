from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .aggregation import TOPR_RULES, bonferroni_max, topr
from .projections import _as_2d, batch_unit_random_projection


@dataclass
class RPTestResult:
    """min_pvalue and rth_pvalue are unadjusted; threshold applies to rth_pvalue."""

    reject: bool
    min_pvalue: float
    threshold: float
    pvalues: NDArray[np.floating]
    rule: str = "max"
    r: int = 1
    rth_pvalue: float = float("nan")

    def __post_init__(self):
        if self.rule == "max" and np.isnan(self.rth_pvalue):
            self.rth_pvalue = self.min_pvalue


def _check_aggregation(aggregation: str, r: int | None, k: int | None = None) -> None:
    if aggregation == "max":
        if r not in (None, 1):
            raise ValueError("r applies only to top-r aggregation.")
    elif aggregation in TOPR_RULES:
        if r is None:
            raise ValueError("top-r aggregation requires r.")
        if k is not None and not 1 <= r <= k:
            raise ValueError("r must satisfy 1 <= r <= k.")
    else:
        raise ValueError(f"aggregation must be 'max' or one of {TOPR_RULES}.")


def aggregate(
    pvalues: NDArray[np.floating],
    *,
    alpha: float,
    aggregation: str = "max",
    r: int | None = None,
    lam: float = 1.5,
) -> RPTestResult:
    """Aggregate projection p-values by 'max' (Bonferroni) or top-r ('beta', 'markov')."""
    _check_aggregation(aggregation, r)
    pvalues = np.asarray(pvalues, dtype=float)
    if aggregation == "max":
        return RPTestResult(*bonferroni_max(pvalues, alpha=alpha), pvalues)
    reject, rth, threshold = topr(pvalues, alpha=alpha, r=r, rule=aggregation, lam=lam)
    return RPTestResult(
        reject, float(np.min(pvalues)), threshold, pvalues, aggregation, r, rth
    )


def rp_test(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    Z: NDArray[np.floating],
    base_test: Callable[
        [NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]], float
    ],
    *,
    k: int = 10,
    alpha: float = 0.05,
    seed: int | None = None,
    aggregation: str = "max",
    r: int | None = None,
    lam: float = 1.5,
) -> RPTestResult:
    """base_test(x, y, z) receives (n, 1), (n, 1), and unprojected 2-D Z.

    It must return a finite scalar p-value in [0, 1] without modifying inputs.
    seed controls projections only; base_test handles calibration.
    aggregation: 'max' (default), or 'beta'/'markov' with r for top-r.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    X = _as_2d(X)
    Y = _as_2d(Y)
    Z = _as_2d(Z)
    n = X.shape[0]
    if Y.shape[0] != n or Z.shape[0] != n:
        raise ValueError("X, Y, Z must have the same number of rows (n).")
    if n == 0:
        raise ValueError("Inputs must contain at least one sample.")
    if X.shape[1] == 0 or Y.shape[1] == 0:
        raise ValueError("X and Y must each contain at least one feature.")
    if k < 1:
        raise ValueError("k must be >= 1.")
    _check_aggregation(aggregation, r, k)

    rng = np.random.default_rng(seed)
    xp = batch_unit_random_projection(X, rng, n_proj=k, target_dim=1)
    yp = batch_unit_random_projection(Y, rng, n_proj=k, target_dim=1)

    pvalues = np.empty(k, dtype=float)
    for index in range(k):
        p = np.asarray(base_test(xp[:, index, :], yp[:, index, :], Z))
        if p.ndim != 0 or p.dtype.kind not in "fiu" or not (0.0 <= p <= 1.0):
            raise ValueError("base_test must return a finite scalar p-value in [0, 1].")
        pvalues[index] = p

    return aggregate(pvalues, alpha=alpha, aggregation=aggregation, r=r, lam=lam)
