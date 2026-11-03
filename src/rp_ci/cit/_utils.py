import numpy as np
from numpy.typing import NDArray


def _as_2d(x: NDArray[np.floating]) -> NDArray[np.floating]:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("Input must be 1D or 2D array.")
    return x
