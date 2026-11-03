"""Kernel statistic n * rho_hat on transformed CIT data."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._utils import _as_2d

_C0_INV = 2.0 * np.exp(-1.0) * (6.5 * np.exp(-2.0) - 20.0 * np.exp(-1.0) + 6.5)
_C0 = 1.0 / _C0_INV


def _validate_unit_matrix(M: NDArray[np.floating], name: str) -> NDArray[np.floating]:
    M = _as_2d(M)
    if M.shape[0] < 2 or M.shape[1] < 1:
        raise ValueError(f"{name} must have at least two rows and one column.")
    if not np.isfinite(M).all():
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(M < 0.0) or np.any(M > 1.0):
        raise ValueError(f"{name} must contain transformed values in [0, 1].")
    return M


def _validate_KW(K_W: NDArray[np.floating], n: int) -> NDArray[np.floating]:
    """Check necessary kernel properties, not PSD or cache provenance."""
    K_W = np.asarray(K_W, dtype=float)
    if K_W.shape != (n, n):
        raise ValueError("K_W must have shape (n,n) matching U/V sample size.")
    if not np.isfinite(K_W).all():
        raise ValueError("K_W must contain only finite values.")
    if np.any(K_W < 0.0) or np.any(K_W > 1.0):
        raise ValueError("K_W entries must lie in [0, 1].")
    if not np.allclose(K_W, K_W.T, rtol=0.0, atol=1e-12):
        raise ValueError("K_W must be symmetric.")
    if not np.allclose(np.diag(K_W), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("K_W must have unit diagonal.")
    return K_W


def dist_mat_l1(M: NDArray[np.floating]) -> NDArray[np.floating]:
    """Pairwise L1 distances between rows of M."""
    M = _as_2d(M)
    return np.abs(M[:, None, :] - M[None, :, :]).sum(axis=2).astype(float)


def prod_bracket(M: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return row products of 2-exp(-M)-exp(M-1), evaluated via logs."""
    M = _as_2d(M)
    x = 2.0 - np.exp(-M) - np.exp(M - 1.0)
    x = np.clip(x, 1e-300, None)  # Prevent log(0).
    return np.exp(np.sum(np.log(x), axis=1)).astype(float)


def compute_KW(W: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return exp(-L1 distances) for finite transformed W in [0,1]."""
    W = _validate_unit_matrix(W, "W")
    DW = dist_mat_l1(W)
    return np.exp(-DW).astype(float)


def rho_hat_multivariate(
    U: NDArray[np.floating],
    V: NDArray[np.floating],
    W: NDArray[np.floating] | None = None,
    K_W: NDArray[np.floating] | None = None,
) -> float:
    """Return n*rho_hat, including diagonal terms.

    U,V,W are finite [0,1] matrices with n>=2 rows, possibly different widths.
    Supplied (n,n) K_W overrides W; it must match the W kernel and row order.
    """
    U = _validate_unit_matrix(U, "U")
    V = _validate_unit_matrix(V, "V")

    n_u, d_u = U.shape
    n_v, d_v = V.shape
    if n_u != n_v:
        raise ValueError("U and V must have the same number of rows (n).")

    DU = dist_mat_l1(U)
    DV = dist_mat_l1(V)
    K_U = np.exp(-DU)
    K_V = np.exp(-DV)

    if K_W is None:
        if W is None:
            raise ValueError("rho_hat_multivariate requires either W or K_W.")
        W = _validate_unit_matrix(W, "W")
        if W.shape[0] != n_u:
            raise ValueError("W must have the same number of rows (n) as U and V.")
        K_W = compute_KW(W)
    else:
        K_W = _validate_KW(K_W, n_u)

    PU = prod_bracket(U)
    PV = prod_bracket(V)

    c_const_u = (2.0 * np.exp(-1.0)) ** d_u
    c_const_v = (2.0 * np.exp(-1.0)) ** d_v

    term_U = K_U + c_const_u - PU[:, None] - PU[None, :]
    term_V = K_V + c_const_v - PV[:, None] - PV[None, :]
    term_W = K_W

    stat = float(np.mean(term_U * term_V * term_W))
    return float((n_u * _C0) * stat)


class RhoHatCachedW:
    """Cache the W kernel; cached(U,V) returns n*rho_hat.

    Rebuild if W or row order changes; modifying W does not refresh K_W.
    """

    def __init__(self, W: NDArray[np.floating]) -> None:
        self.W = _validate_unit_matrix(W, "W")
        self.K_W = compute_KW(self.W)

    def __call__(self, U: NDArray[np.floating], V: NDArray[np.floating]) -> float:
        U = _validate_unit_matrix(U, "U")
        V = _validate_unit_matrix(V, "V")

        n_u, d_u = U.shape
        n_v, d_v = V.shape
        if n_v != n_u:
            raise ValueError("U and V must have the same number of rows (n).")
        K_W = _validate_KW(self.K_W, n_u)

        DU = dist_mat_l1(U)
        DV = dist_mat_l1(V)
        K_U = np.exp(-DU)
        K_V = np.exp(-DV)

        PU = prod_bracket(U)
        PV = prod_bracket(V)

        c_const_u = (2.0 * np.exp(-1.0)) ** d_u
        c_const_v = (2.0 * np.exp(-1.0)) ** d_v

        term_U = K_U + c_const_u - PU[:, None] - PU[None, :]
        term_V = K_V + c_const_v - PV[:, None] - PV[None, :]
        term_W = K_W

        stat = float(np.mean(term_U * term_V * term_W))
        return float((n_u * _C0) * stat)
