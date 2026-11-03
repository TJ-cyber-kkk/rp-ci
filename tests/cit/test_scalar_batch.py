import numpy as np
import pytest

from rp_ci.cit import scalar_batch as batch
from rp_ci.cit.calibration import plus_one_critical_value, plus_one_pvalues, simulate_null_pool
from rp_ci.cit.rosenblatt import rosenblatt_transform
from rp_ci.cit.statistic import rho_hat_multivariate


def _data(n=13, k=7, dz=1):
    rng = np.random.default_rng(291013)
    return (rng.normal(size=(n, k, 1)), rng.normal(size=(n, k, 1)),
            rng.normal(size=(n, dz)))


def _ordinary_statistics(xp, yp, z, h_scale):
    return np.array([
        rho_hat_multivariate(*rosenblatt_transform(
            xp[:, j, :], yp[:, j, :], z, h_scale=h_scale,
        ))
        for j in range(xp.shape[1])
    ])


@pytest.mark.parametrize("dims", [(2, 1, 1), (13, 7, 1), (15, 5, 2)])
@pytest.mark.parametrize("h_scale", [.7, .8, 1.2, 1e-9])
@pytest.mark.parametrize("batch_size", [1, 3, 32])
def test_matches_ordinary_statistics(dims, h_scale, batch_size):
    arrays = _data(*dims)
    before = [a.copy() for a in arrays]
    expected = _ordinary_statistics(*arrays, h_scale)
    actual = batch.scalar_projected_statistics(*arrays, h_scale=h_scale, batch_size=batch_size)
    assert actual.shape == (dims[1],)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    for a, original in zip(arrays, before):
        np.testing.assert_array_equal(a, original)


@pytest.mark.parametrize("kind", ["ties", "constant_z", "constant_x", "tiny_z"])
def test_edge_data_matches_ordinary_path(kind):
    xp, yp, z = _data()
    if kind == "ties":
        xp, yp, z = (np.round(a) for a in (xp, yp, z))
    elif kind == "constant_z":
        z[:] = 0.
    elif kind == "constant_x":
        xp[:] = 2.
    else:
        z *= 1e-10
    np.testing.assert_allclose(
        batch.scalar_projected_statistics(xp, yp, z),
        _ordinary_statistics(xp, yp, z, .8), rtol=2e-13, atol=2e-13,
    )


def test_default_bandwidth_scale_is_explicitly_0_8():
    arrays = _data()
    np.testing.assert_array_equal(
        batch.scalar_projected_statistics(*arrays),
        batch.scalar_projected_statistics(*arrays, h_scale=.8),
    )


def test_vector_z_and_projection_order():
    xp, yp, z = _data()
    original = batch.scalar_projected_statistics(xp, yp, z)
    order = np.array([4, 0, 6, 1, 3, 5, 2])
    np.testing.assert_array_equal(original, batch.scalar_projected_statistics(xp, yp, z[:, 0]))
    np.testing.assert_allclose(
        original[order], batch.scalar_projected_statistics(xp[:, order], yp[:, order], z),
        rtol=2e-13, atol=2e-13,
    )
    np.testing.assert_array_equal(original, batch.scalar_projected_statistics(yp, xp, z))


def test_row_permutation():
    arrays = _data()
    order = np.random.default_rng(82).permutation(len(arrays[0]))
    np.testing.assert_allclose(
        batch.scalar_projected_statistics(*(a[order] for a in arrays)),
        batch.scalar_projected_statistics(*arrays), rtol=2e-13, atol=2e-13,
    )


def test_batch_size_preserves_results_within_roundoff():
    arrays = _data(n=18, k=39)
    expected = batch.scalar_projected_statistics(*arrays, batch_size=1)
    for size in [3, 32, 100]:
        np.testing.assert_allclose(
            batch.scalar_projected_statistics(*arrays, batch_size=size), expected,
            rtol=2e-13, atol=2e-13,
        )


def test_memory_cap_splits_large_requested_batch(monkeypatch):
    arrays = _data(n=400, k=19)
    expected = batch.scalar_projected_statistics(*arrays, batch_size=1)
    mean = np.mean
    shapes = []

    def recording_mean(a, *args, **kwargs):
        shapes.append(a.shape)
        return mean(a, *args, **kwargs)

    monkeypatch.setattr(batch.np, "mean", recording_mean)
    actual = batch.scalar_projected_statistics(*arrays, batch_size=1000)
    assert shapes == [(8, 400, 400), (8, 400, 400), (3, 400, 400)]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_no_random_draws(monkeypatch):
    arrays = _data()
    expected = batch.scalar_projected_statistics(*arrays)

    def unexpected_rng(*args, **kwargs):
        pytest.fail("Fixed projected data must not require random draws")

    for name in ["default_rng", "normal", "uniform", "random", "standard_normal"]:
        monkeypatch.setattr(np.random, name, unexpected_rng)
    np.testing.assert_array_equal(batch.scalar_projected_statistics(*arrays), expected)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_data_rejected(index, bad):
    arrays = list(_data())
    arrays[index].flat[0] = bad
    with pytest.raises(ValueError, match="finite"):
        batch.scalar_projected_statistics(*arrays)


@pytest.mark.parametrize("kind", ["xp_2d", "yp_rows", "multivariate_projection", "no_rows",
                                  "one_row", "no_projections", "z_rows", "z_empty", "z_3d"])
def test_invalid_shapes_rejected(kind):
    xp, yp, z = _data()
    if kind == "xp_2d":
        xp = xp[:, :, 0]
    elif kind == "yp_rows":
        yp = yp[:-1]
    elif kind == "multivariate_projection":
        xp, yp = (np.repeat(a, 2, axis=2) for a in (xp, yp))
    elif kind in {"no_rows", "one_row"}:
        n = 0 if kind == "no_rows" else 1
        xp, yp, z = xp[:n], yp[:n], z[:n]
    elif kind == "no_projections":
        xp, yp = xp[:, :0], yp[:, :0]
    elif kind == "z_rows":
        z = z[:-1]
    elif kind == "z_empty":
        z = z[:, :0]
    else:
        z = z[:, :, None]
    with pytest.raises(ValueError):
        batch.scalar_projected_statistics(xp, yp, z)


@pytest.mark.parametrize("scale", [0., -1., np.nan, np.inf, [1.2]])
def test_invalid_bandwidth_scale_rejected(scale):
    with pytest.raises(ValueError, match="h_scale"):
        batch.scalar_projected_statistics(*_data(), h_scale=scale)


@pytest.mark.parametrize("size", [0, -1])
def test_nonpositive_batch_size_rejected(size):
    with pytest.raises(ValueError, match="batch_size"):
        batch.scalar_projected_statistics(*_data(), batch_size=size)


def test_nonfinite_output_is_not_returned(monkeypatch):
    monkeypatch.setattr(batch, "_C0", np.nan)
    with pytest.raises(ValueError, match="Observed statistics"):
        batch.scalar_projected_statistics(*_data())


def test_batched_results_feed_existing_calibration():
    arrays = _data()
    batched = batch.scalar_projected_statistics(*arrays)
    ordinary = _ordinary_statistics(*arrays, .8)
    pool = simulate_null_pool(len(arrays[0]), 1, 1, 1, B=99, seed=872, n_jobs=1)
    pvalues = plus_one_pvalues(batched, pool)
    np.testing.assert_array_equal(pvalues, plus_one_pvalues(ordinary, pool))
    for level in [.01, .05, .2]:
        np.testing.assert_array_equal(batched > plus_one_critical_value(pool, level), pvalues <= level)
