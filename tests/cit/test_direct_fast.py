import inspect

import numpy as np
import pytest

from rp_ci.cit import (
    direct_cit_statistic_fast,
    rho_hat_multivariate,
    rosenblatt_target_given_z_fast,
    rosenblatt_transform,
)


@pytest.mark.parametrize("dims", [(9, 1, 1, 1), (11, 3, 4, 1), (8, 2, 3, 2)])
@pytest.mark.parametrize("h_scale", [.8, 1.2])
def test_fast_direct_matches_ordinary_path(dims, h_scale):
    n, dx, dy, dz = dims
    rng = np.random.default_rng(1307 + dx + 10 * dy + 100 * dz)
    x, y, z = (
        rng.normal(size=(n, dx)),
        rng.normal(size=(n, dy)),
        rng.normal(size=(n, dz)),
    )
    u, v, w = rosenblatt_transform(x, y, z, h_scale=h_scale)
    expected = rho_hat_multivariate(u, v, W=w)
    actual, u_fast, v_fast, w_fast = direct_cit_statistic_fast(
        x, y, z, h_scale=h_scale
    )
    np.testing.assert_allclose(u_fast, u, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(v_fast, v, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(w_fast, w)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_fast_direct_default_is_0_8():
    assert inspect.signature(
        direct_cit_statistic_fast
    ).parameters["h_scale"].default == .8
    assert inspect.signature(
        rosenblatt_target_given_z_fast
    ).parameters["h_scale"].default == .8


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, [1.0]])
def test_fast_direct_rejects_invalid_bandwidth(bad):
    rng = np.random.default_rng(41)
    x, y, z = (rng.normal(size=(7, 2)), rng.normal(size=(7, 3)),
               rng.normal(size=(7, 1)))
    with pytest.raises(ValueError, match="h_scale"):
        direct_cit_statistic_fast(x, y, z, h_scale=bad)


def test_fast_direct_rejects_mismatched_rows():
    rng = np.random.default_rng(51)
    with pytest.raises(ValueError, match="same number"):
        direct_cit_statistic_fast(
            rng.normal(size=(7, 2)), rng.normal(size=(6, 2)),
            rng.normal(size=(7, 1)),
        )
