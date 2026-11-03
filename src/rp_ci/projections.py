from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _as_2d(x: NDArray[np.floating]) -> NDArray[np.floating]:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("Input must be 1D or 2D array.")
    return x


def unit_random_projection(
    X: NDArray[np.floating],
    rng: np.random.Generator,
    out_dim: int = 1,
) -> NDArray[np.floating]:
    """Project X of shape (n, d) or (n,) to (n, out_dim).

    Uses normalized Gaussian directions and advances rng in place.
    """
    X = _as_2d(X)
    if out_dim < 1:
        raise ValueError("out_dim must be >= 1.")

    n, d = X.shape
    U = rng.normal(size=(d, out_dim))
    U /= (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
    return (X @ U).astype(float)


def batch_unit_random_projection(
    X: NDArray[np.floating],
    rng: np.random.Generator,
    n_proj: int,
    target_dim: int = 1,
) -> NDArray[np.floating]:
    """Project X of shape (n, d) or (n,) to (n, n_proj, target_dim).

    Uses normalized Gaussian directions and advances rng in place.
    """
    X = _as_2d(X)
    if n_proj < 1:
        raise ValueError("n_proj must be >= 1.")
    if target_dim < 1:
        raise ValueError("target_dim must be >= 1.")

    n, d = X.shape
    R = rng.normal(size=(d, n_proj * target_dim))
    R /= (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
    Xp = X @ R
    return Xp.reshape(n, n_proj, target_dim).astype(float)
