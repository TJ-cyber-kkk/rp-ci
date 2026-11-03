"""Exact matrix implementation for direct multivariate CIT."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._utils import _as_2d
from .rosenblatt import _validate_h_scale, band_width, conditional_cdf
from .statistic import rho_hat_multivariate


def _matrix(value: NDArray[np.floating], name: str) -> NDArray[np.floating]:
    value = _as_2d(value)
    if value.shape[0] < 2 or value.shape[1] < 1:
        raise ValueError(f"{name} must have at least two rows and one column.")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _add_distances(cumulative: NDArray[np.floating], values: NDArray[np.floating]) -> None:
    values = _matrix(values, "conditioning values")
    scales = values.std(axis=0, ddof=1)
    if not np.isfinite(scales).all():
        raise ValueError("Conditioning scales must be finite.")
    scales = np.where(scales < 1e-8, 1.0, scales)
    for column, scale in zip(values.T, scales):
        difference = (column[:, None] - column[None, :]) / scale
        cumulative += difference * difference


def _cdf_rows(
    target: NDArray[np.floating],
    cumulative: NDArray[np.floating],
    conditioning_dimension: int,
    h_scale: float,
) -> NDArray[np.floating]:
    target = np.asarray(target, dtype=float).reshape(-1)
    n = target.size
    bandwidth = 1.06 * n ** (-1.0 / (4.0 + conditioning_dimension)) * h_scale
    weights = np.exp(-0.5 * cumulative / bandwidth**2)
    denominator = weights.sum(axis=1)
    if not np.isfinite(denominator).all():
        raise ValueError("Conditional CDF weights must be finite.")
    less_equal = target[None, :] <= target[:, None]
    numerator = np.sum(weights * less_equal, axis=1)
    result = np.empty(n, dtype=float)
    regular = denominator >= 1e-10
    result[regular] = numerator[regular] / denominator[regular]
    result[~regular] = less_equal.mean(axis=1)[~regular]
    return result


def rosenblatt_target_given_z_fast(
    target: NDArray[np.floating],
    z: NDArray[np.floating],
    h_scale: float = 0.8,
) -> NDArray[np.floating]:
    """Return the self-including transform using cached distances."""
    _validate_h_scale(h_scale)
    target = _matrix(target, "target")
    z = _matrix(z, "Z")
    if target.shape[0] != z.shape[0]:
        raise ValueError("target and Z must have the same number of rows.")

    n, dimension = target.shape
    conditioning_columns = np.column_stack((z, target[:, :-1]))
    scales = conditioning_columns.std(axis=0, ddof=1)
    if not np.isfinite(scales).all():
        raise ValueError("Conditioning scales must be finite.")
    scales = np.where(scales < 1e-8, 1.0, scales)
    cumulative = np.zeros((n, n), dtype=float)
    _add_distances(cumulative, z)
    transformed = np.empty_like(target)
    conditioning_dimension = z.shape[1]

    for coordinate in range(dimension):
        raw_bandwidth = (
            1.06
            * scales[:conditioning_dimension]
            * n ** (-1.0 / (4.0 + conditioning_dimension))
            * h_scale
        )
        if np.any(raw_bandwidth < 1e-8) or not np.isfinite(raw_bandwidth).all():
            conditioning = conditioning_columns[:, :conditioning_dimension]
            bandwidth = band_width(conditioning, h_scale)
            transformed[:, coordinate] = [
                conditional_cdf(
                    target[:, coordinate], conditioning, target[i, coordinate],
                    conditioning[i], bandwidth,
                )
                for i in range(n)
            ]
        else:
            transformed[:, coordinate] = _cdf_rows(
                target[:, coordinate], cumulative, conditioning_dimension, h_scale
            )
        _add_distances(cumulative, target[:, coordinate : coordinate + 1])
        conditioning_dimension += 1
    return transformed


def direct_cit_statistic_fast(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    h_scale: float = 0.8,
) -> tuple[float, NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return the direct-CIT statistic and transforms without approximation."""
    x = _matrix(x, "X")
    y = _matrix(y, "Y")
    z = _matrix(z, "Z")
    if x.shape[0] != y.shape[0] or x.shape[0] != z.shape[0]:
        raise ValueError("X, Y, and Z must have the same number of rows.")
    u = rosenblatt_target_given_z_fast(x, z, h_scale=h_scale)
    v = rosenblatt_target_given_z_fast(y, z, h_scale=h_scale)
    ranks = np.searchsorted(np.sort(z[:, 0]), z[:, 0], side="right") / float(len(z))
    if z.shape[1] == 1:
        w = ranks[:, None]
    else:
        from .rosenblatt import rosenblatt_Z_only
        w = rosenblatt_Z_only(z, h_scale=h_scale)
    statistic = rho_hat_multivariate(u, v, W=w)
    return statistic, u, v, w
