"""Batched scalar CIT statistics for fixed projected data."""
from __future__ import annotations

import numpy as np

from .rosenblatt import band_width, rosenblatt_Z_only
from .statistic import _C0, compute_KW
from ._utils import _as_2d


def scalar_projected_statistics(
    xp: np.ndarray,
    yp: np.ndarray,
    z: np.ndarray,
    *,
    h_scale: float = 0.8,
    batch_size: int = 32,
) -> np.ndarray:
    """Return k RP-CIT statistics n*rho_hat for xp,yp of shape (n,k,1).

    z: (n,dz) or (n,). Self-including, with diagonal terms; no RNG draws.
    Batch size and memory layout can affect roundoff.
    """
    xp, yp, z = np.asarray(xp, dtype=float), np.asarray(yp, dtype=float), _as_2d(z)
    if xp.ndim != 3 or xp.shape[-1] != 1 or yp.shape != xp.shape:
        raise ValueError("xp and yp must have the same shape (n, k, 1).")
    n, k, _ = xp.shape
    if n < 2 or k < 1 or z.shape[0] != n or z.shape[1] < 1:
        raise ValueError("Need n >= 2, k >= 1 and matching, nonempty Z.")
    if not all(np.isfinite(a).all() for a in (xp, yp, z)):
        raise ValueError("Projected data and Z must be finite.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    # Cap the estimated per-batch working set, not total process memory.
    batch_size = min(batch_size, max(1, (64 * 1024**2) // (48 * n * n)))

    h = band_width(z, h_scale)
    weights = np.exp(-0.5 * np.sum(((z[:, None, :] - z[None, :, :]) / h) ** 2, axis=2))
    denominator = weights.sum(axis=1)  # Self-weight is one.
    kw = compute_KW(rosenblatt_Z_only(z, h_scale=h_scale))
    result = np.empty(k, dtype=float)
    for start in range(0, k, batch_size):
        stop = min(k, start + batch_size)
        terms = []
        for projected in (xp, yp):
            values = projected[:, start:stop, 0].T
            comparisons = values[:, None, :] <= values[:, :, None]
            uv = np.sum(comparisons * weights[None, :, :], axis=2) / denominator
            bracket = 2.0 - np.exp(-uv) - np.exp(uv - 1.0)
            terms.append(
                np.exp(-np.abs(uv[:, :, None] - uv[:, None, :]))
                + 2.0 * np.exp(-1.0) - bracket[:, :, None] - bracket[:, None, :]
            )
        result[start:stop] = n * _C0 * np.mean(terms[0] * terms[1] * kw, axis=(1, 2))
    if not np.isfinite(result).all():
        raise ValueError("Observed statistics must contain only finite values.")
    return result
