from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta

TOPR_RULES = ("beta", "markov")


def bonferroni_max(
    pvalues: NDArray[np.floating],
    *,
    alpha: float,
) -> tuple[bool, float, float]:
    """Return (reject, unadjusted_min_pvalue, alpha / len(pvalues)).

    Requires nonempty 1-D finite p-values in [0, 1] and 0 < alpha < 1.
    """
    pvals = np.asarray(pvalues, dtype=float)
    k = pvals.size
    min_p = float(np.min(pvals))
    threshold = float(alpha / k)
    return bool(min_p <= threshold), min_p, threshold


def topr_threshold(
    alpha: float,
    k: int,
    r: int,
    *,
    rule: str = "beta",
    lam: float = 1.5,
) -> float:
    """Return the threshold for the r-th smallest of k p-values.

    'beta': Beta(r, k-r+1) alpha-quantile / lam.
    'markov': alpha * r / k.
    """
    level = np.asarray(alpha, dtype=float)
    if level.ndim != 0 or not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("alpha must be a finite scalar in (0,1).")
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1:
        raise ValueError("k must be an integer >= 1.")
    if (not isinstance(r, (int, np.integer)) or isinstance(r, (bool, np.bool_))
            or not 1 <= r <= k):
        raise ValueError("r must be an integer with 1 <= r <= k.")
    if rule == "beta":
        if not np.isfinite(lam) or lam < 1.0:
            raise ValueError("lam must be finite and >= 1.")
        return float(beta.ppf(alpha, r, k - r + 1)) / lam
    if rule == "markov":
        return float(alpha * r / k)
    raise ValueError(f"rule must be one of {TOPR_RULES}.")


def topr(
    pvalues: NDArray[np.floating],
    *,
    alpha: float,
    r: int,
    rule: str = "beta",
    lam: float = 1.5,
) -> tuple[bool, float, float]:
    """Return (reject, r-th smallest p-value, threshold)."""
    pvals = np.asarray(pvalues, dtype=float)
    if pvals.ndim != 1 or pvals.size == 0:
        raise ValueError("pvalues must be a nonempty 1-D array.")
    if not (np.isfinite(pvals).all() and (pvals >= 0.0).all() and (pvals <= 1.0).all()):
        raise ValueError("pvalues must be finite and lie in [0, 1].")
    threshold = topr_threshold(alpha, pvals.size, r, rule=rule, lam=lam)
    rth = float(np.partition(pvals, r - 1)[r - 1])
    return bool(rth <= threshold), rth, threshold
