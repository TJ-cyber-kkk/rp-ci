"""Uniform reference simulation and Monte Carlo calibration for n*rho_hat."""
from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from .statistic import rho_hat_multivariate


def plus_one_pvalues(
    statistics: NDArray[np.floating],
    null_pool: NDArray[np.floating],
    *,
    assume_sorted: bool = False,
) -> NDArray[np.floating]:
    """Return (1 + count{reference >= T})/(B+1), with T's shape.

    assume_sorted requires an ascending pool.
    """
    stats = np.asarray(statistics, dtype=float)
    if not np.isfinite(stats).all():
        raise ValueError("Observed statistics must contain only finite values.")
    pool = np.asarray(null_pool, dtype=float).reshape(-1)
    if pool.size < 1:
        raise ValueError("null_pool must contain at least one statistic.")
    if not np.isfinite(pool).all():
        raise ValueError("null_pool must contain only finite values.")

    sorted_pool = pool if assume_sorted else np.sort(pool)
    number_strictly_below = np.searchsorted(sorted_pool, stats, side="left")
    number_greater_equal = sorted_pool.size - number_strictly_below
    return np.asarray(
        (1.0 + number_greater_equal) / (sorted_pool.size + 1.0),
        dtype=float,
    )


def plus_one_critical_value(
    null_pool: NDArray[np.floating],
    alpha_single: float,
) -> float:
    """Return c such that p(T)<=alpha_single iff T>c, including ties.

    Returns +inf if the level is below the minimum p-value 1/(B+1).
    """
    pool = np.asarray(null_pool, dtype=float).reshape(-1)
    if pool.size < 1:
        raise ValueError("null_pool must contain at least one statistic.")
    if not np.isfinite(pool).all():
        raise ValueError("null_pool must contain only finite values.")
    if not (0.0 < alpha_single < 1.0):
        raise ValueError("alpha_single must be in (0,1).")

    # Search the p-value grid to avoid rounding in (B+1)*alpha_single.
    low, high = 0, pool.size
    denominator = pool.size + 1.0
    while low < high:
        middle = (low + high + 1) // 2
        if middle / denominator <= alpha_single:
            low = middle
        else:
            high = middle - 1
    if low == 0:
        return float("inf")

    sorted_pool = np.sort(pool)
    cutoff_index = pool.size - low
    return float(sorted_pool[cutoff_index])


def simulate_null_pool(
    n: int,
    du: int,
    dv: int,
    dw: int,
    B: int = 200000,
    seed: int | None = None,
    n_jobs: int = -1,
    batch: int = 1000
) -> NDArray[np.floating]:
    """Return B reference n*rho_hat values from independent Uniform U,V,W.

    Each replicate has n rows and dimensions du,dv,dw. Indexed seeds
    reproduce across batch sizes and workers in the same numerical environment.
    """
    if n < 2:
        raise ValueError("n must be >= 2.")
    if du < 1 or dv < 1 or dw < 1:
        raise ValueError("du, dv, dw must be >= 1.")
    if not isinstance(B, (int, np.integer)) or isinstance(B, (bool, np.bool_)) or B < 1:
        raise ValueError("B must be a positive integer.")
    if batch < 1:
        raise ValueError("batch must be >= 1.")

    base_seed = (
        int(np.random.SeedSequence().entropy) if seed is None else int(seed)
    )
    if base_seed < 0:
        raise ValueError("seed must be non-negative or None.")
    seed_words = (
        base_seed & 0xFFFFFFFF,
        (base_seed >> 32) & 0xFFFFFFFF,
    )

    def _one(index: int) -> float:
        child = np.random.SeedSequence([*seed_words, int(index)])
        r = np.random.default_rng(child)
        U = r.uniform(size=(n, du))
        V = r.uniform(size=(n, dv))
        W = r.uniform(size=(n, dw))
        return float(rho_hat_multivariate(U, V, W=W))

    if n_jobs == 1:
        return np.fromiter((_one(index) for index in range(B)), dtype=float, count=B)

    stats_out: list[float] = []
    parallel_backend = "loky"
    for i in range(0, B, batch):
        chunk = range(i, min(i + batch, B))
        try:
            vals = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
                delayed(_one)(index) for index in chunk
            )
        except (NotImplementedError, PermissionError) as exc:
            warnings.warn(
                f"process parallelism unavailable ({type(exc).__name__}); "
                "falling back to deterministic threads",
                RuntimeWarning,
                stacklevel=2,
            )
            parallel_backend = "threading"
            vals = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(_one)(index) for index in chunk
            )
        stats_out.extend(vals)

    return np.asarray(stats_out, dtype=float)
