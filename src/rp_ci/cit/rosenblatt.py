"""Self-including Rosenblatt estimates with Gaussian kernel weights."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._utils import _as_2d

ROSENBLATT_VERSION = "self_including_ecdf_n_v3_guarded"


def _validate_h_scale(h_scale: float) -> None:
    if np.ndim(h_scale) != 0 or not np.isfinite(h_scale) or h_scale <= 0:
        raise ValueError("h_scale must be finite and strictly positive.")


def _validate_cit_matrix(value: NDArray[np.floating], name: str) -> NDArray[np.floating]:
    value = _as_2d(value)
    if value.shape[0] < 2 or value.shape[1] < 1:
        raise ValueError(f"{name} must have at least two rows and one column.")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value


def band_width(A: NDArray[np.floating], h_scale: float = 0.8) -> NDArray[np.floating]:
    """Coordinate rule-of-thumb bandwidths for conditioning data A of shape (n,d).

    SDs and final bandwidths below 1e-8 are replaced by 1.
    """
    _validate_h_scale(h_scale)
    A = _as_2d(A)
    n, d = A.shape
    if n < 2 or not np.isfinite(A).all():
        raise ValueError("Bandwidth data must have at least two rows and be finite.")
    if d == 0:
        return np.array([1.0], dtype=float)

    std = A.std(axis=0, ddof=1)
    if not np.isfinite(std).all():
        raise ValueError("Nonfinite bandwidth scale; check the data magnitude.")
    std = np.where(std < 1e-8, 1.0, std)

    h = 1.06 * std * (n ** (-1.0 / (4.0 + d))) * h_scale
    h = np.where(h < 1e-8, 1.0, h)
    if not np.isfinite(h).all():
        raise ValueError("Computed bandwidth must be finite.")
    return h.astype(float)


def conditional_cdf(
    T: NDArray[np.floating],
    C: NDArray[np.floating],
    t: float,
    c: NDArray[np.floating],
    h_c: NDArray[np.floating],
    exclude: int | None = None,
) -> float:
    """Gaussian-kernel estimate of F(T<=t | C=c).

    T has n entries; C is (n, d); c and h_c have d entries.
    Uses all rows unless exclude specifies a row to omit.
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    C = _as_2d(C)
    c = np.asarray(c, dtype=float).reshape(-1)
    h_c = np.asarray(h_c, dtype=float).reshape(-1)

    if T.size != C.shape[0] or T.size == 0:
        raise ValueError("T and C must have matching, nonempty rows.")
    if not all(np.isfinite(a).all() for a in (T, C, c)) or not np.isfinite(t):
        raise ValueError("Conditional CDF inputs must be finite.")
    if not np.isfinite(h_c).all() or np.any(h_c <= 0):
        raise ValueError("h_c bandwidth must be finite and strictly positive.")
    if C.shape[1] != c.size:
        raise ValueError(f"Dimension mismatch: C has d={C.shape[1]} but c has size {c.size}.")
    if C.shape[1] != h_c.size:
        raise ValueError(f"Dimension mismatch: C has d={C.shape[1]} but h_c has size {h_c.size}.")

    if exclude is not None:
        if not isinstance(exclude, (int, np.integer)) or not 0 <= exclude < T.size or T.size < 2:
            raise ValueError("exclude must select a row and leave at least one observation.")
        mask = np.ones(C.shape[0], dtype=bool)
        mask[exclude] = False
        T_use = T[mask]
        C_use = C[mask]
    else:
        T_use, C_use = T, C

    diff = (C_use - c) / h_c
    w = np.exp(-0.5 * np.sum(diff**2, axis=1))
    den = float(np.sum(w))
    if not np.isfinite(den):
        raise ValueError("Conditional CDF weights must be finite.")

    # Fall back to the empirical CDF for negligible weights.
    if den < 1e-10:
        return float(np.mean(T_use <= t))

    num = float(np.sum(w * (T_use <= t)))
    return float(num / den)


def rosenblatt_Z_only(Z: NDArray[np.floating], h_scale: float = 0.8) -> NDArray[np.floating]:
    """Return W of shape (n, dz): ECDF first, then sequential conditional CDFs."""
    _validate_h_scale(h_scale)
    Z = _validate_cit_matrix(Z, "Z")
    n, dz = Z.shape
    W = np.empty((n, dz), dtype=float)

    # Right-continuous ECDF: #{j: z_j <= z_i}/n.
    W[:, 0] = np.searchsorted(np.sort(Z[:, 0]), Z[:, 0], side="right") / float(n)

    if dz > 1:
        for k in range(1, dz):
            C = Z[:, :k]
            hC = band_width(C, h_scale)
            for i in range(n):
                W[i, k] = conditional_cdf(Z[:, k], C, float(Z[i, k]), Z[i, :k], hC)

    return W


def rosenblatt_XY_given_Z(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    Z: NDArray[np.floating],
    h_scale: float = 0.8,
    hZ: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Self-including transforms U,V with the shapes of X,Y.

    Each coordinate conditions on Z and its own preceding coordinates.
    Optional hZ has shape (dz,) and supplies bandwidths for Z alone.
    """
    _validate_h_scale(h_scale)
    X = _validate_cit_matrix(X, "X")
    Y = _validate_cit_matrix(Y, "Y")
    Z = _validate_cit_matrix(Z, "Z")

    n = X.shape[0]
    if Y.shape[0] != n or Z.shape[0] != n:
        raise ValueError("X, Y, Z must have the same number of rows (n).")

    dx = X.shape[1]
    dy = Y.shape[1]

    U = np.empty((n, dx), dtype=float)
    V = np.empty((n, dy), dtype=float)

    if hZ is None:
        hZ = band_width(Z, h_scale)

    for i in range(n):
        U[i, 0] = conditional_cdf(X[:, 0], Z, float(X[i, 0]), Z[i], hZ)

    for k in range(1, dx):
        C = np.column_stack([Z, X[:, :k]])
        hC = band_width(C, h_scale)
        for i in range(n):
            c = np.concatenate([Z[i], X[i, :k]])
            U[i, k] = conditional_cdf(X[:, k], C, float(X[i, k]), c, hC)

    for i in range(n):
        V[i, 0] = conditional_cdf(Y[:, 0], Z, float(Y[i, 0]), Z[i], hZ)

    for k in range(1, dy):
        C = np.column_stack([Z, Y[:, :k]])
        hC = band_width(C, h_scale)
        for i in range(n):
            c = np.concatenate([Z[i], Y[i, :k]])
            V[i, k] = conditional_cdf(Y[:, k], C, float(Y[i, k]), c, hC)

    return U, V


def rosenblatt_transform(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    Z: NDArray[np.floating],
    h_scale: float = 0.8,
    W_precomputed: NDArray[np.floating] | None = None,
    hZ_precomputed: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return U,V,W in [0,1], shaped like the matrix forms of X,Y,Z.

    Cached W and hZ must match Z, row order and bandwidth policy;
    provenance is not checked.
    """
    if W_precomputed is None:
        W = rosenblatt_Z_only(Z, h_scale=h_scale)
    else:
        W = np.asarray(W_precomputed, dtype=float)
        if W.shape != _as_2d(Z).shape or not np.isfinite(W).all():
            raise ValueError("W_precomputed must be finite and have the same shape as Z.")
        if np.any(W < 0.0) or np.any(W > 1.0):
            raise ValueError("W_precomputed must contain transformed values in [0, 1].")

    U, V = rosenblatt_XY_given_Z(X, Y, Z, h_scale=h_scale, hZ=hZ_precomputed)
    return U, V, W
