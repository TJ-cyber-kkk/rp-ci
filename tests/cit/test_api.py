from dataclasses import FrozenInstanceError
import inspect

import numpy as np
import pytest

from rp_ci import rp_test
from rp_ci.cit import (
    CITResult, band_width, cit_test, plus_one_critical_value, plus_one_pvalues,
    rho_hat_multivariate, rosenblatt_transform, scalar_projected_statistics,
    simulate_null_pool,
)
from rp_ci.cit import api, calibration
from rp_ci.projections import batch_unit_random_projection


@pytest.fixture
def data():
    rng = np.random.default_rng(918)
    z = rng.normal(size=12)
    return z + rng.normal(size=12), z + rng.normal(size=12), z


@pytest.mark.parametrize("dz", [1, 2])
@pytest.mark.parametrize("h_scale", [.7, 1.2, 1.4])
def test_matches_explicit_computation(dz, h_scale):
    rng = np.random.default_rng(51)
    x, y, z = rng.normal(size=9), rng.normal(size=9), rng.normal(size=(9, dz))
    pool = simulate_null_pool(9, 1, 1, dz, B=23, seed=512, n_jobs=1)
    before = [a.copy() for a in (x, y, z, pool)]
    result = cit_test(x, y, z, null_pool=pool, alpha=.2, h_scale=h_scale)
    u, v, w = rosenblatt_transform(x, y, z, h_scale=h_scale)
    stat = rho_hat_multivariate(u, v, W=w)
    p = (1 + np.count_nonzero(pool >= stat)) / (len(pool) + 1)
    assert result == CITResult(p <= .2, p, stat, plus_one_critical_value(pool, .2), .2, 23)
    assert result.reject == (result.statistic > result.critical_value)
    for actual, saved in zip((x, y, z, pool), before):
        np.testing.assert_array_equal(actual, saved)


def test_internal_pool_matches_explicit_pool_and_is_reproducible(data):
    pool = simulate_null_pool(12, 1, 1, 1, B=19, seed=38, n_jobs=1)
    expected = cit_test(*data, null_pool=pool)
    assert cit_test(*data, B=19, seed=38) == expected
    assert cit_test(*data, B=19, seed=38) == expected


def test_supplied_pool_ignores_simulation_options(data, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Supplied pool must not trigger simulation")
    monkeypatch.setattr(calibration, "simulate_null_pool", forbidden)
    pool = np.arange(19.)
    expected = cit_test(*data, null_pool=pool)
    assert cit_test(*data, null_pool=pool, B=0, seed=-1, n_jobs=0) == expected


def test_vectors_columns_and_lists_are_equivalent(data):
    kwargs = dict(null_pool=np.arange(19.))
    expected = cit_test(*data, **kwargs)
    assert cit_test(*(a[:, None] for a in data), **kwargs) == expected
    assert cit_test(*(a.tolist() for a in data), **kwargs) == expected


@pytest.mark.parametrize("B", [1, 2, 7, 18])
def test_insufficient_resolution_warns_without_changing_budget(data, B):
    with pytest.warns(RuntimeWarning, match="cannot reject"):
        result = cit_test(*data, B=B, seed=1)
    assert result.reference_size == B
    assert result.pvalue >= 1 / (B + 1)
    assert result.critical_value == np.inf
    assert not result.reject


def test_resolution_boundary_and_ties(data):
    stat = cit_test(*data, null_pool=np.arange(19.)).statistic
    low = cit_test(*data, null_pool=np.full(19, stat - 1))
    tied = cit_test(*data, null_pool=np.full(19, stat))
    assert low.reject and low.pvalue == .05
    assert not tied.reject and tied.pvalue == 1
    one = cit_test(*data, null_pool=[stat - 1], alpha=.5)
    assert one.reject and one.pvalue == .5


def test_result_is_immutable(data):
    result = cit_test(*data, null_pool=np.arange(19.))
    with pytest.raises(FrozenInstanceError):
        result.pvalue = 0


@pytest.mark.parametrize("alpha", [0, 1, -1, np.nan, np.inf, [.05]])
def test_invalid_alpha(data, alpha):
    with pytest.raises(ValueError, match="alpha"):
        cit_test(*data, alpha=alpha)


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, [1.2]])
def test_invalid_bandwidth(data, scale):
    with pytest.raises(ValueError, match="h_scale"):
        cit_test(*data, h_scale=scale)


@pytest.mark.parametrize("B", [0, -1, 1.2, True, np.nan, [19]])
def test_invalid_budget(data, B):
    with pytest.raises(ValueError, match="B must"):
        cit_test(*data, B=B)


@pytest.mark.parametrize("seed", [-1, .5, True, np.nan, [1]])
def test_invalid_seed(data, seed):
    with pytest.raises(ValueError, match="seed"):
        cit_test(*data, seed=seed)


@pytest.mark.parametrize("jobs", [0, .5, True, np.nan, [1]])
def test_invalid_jobs(data, jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        cit_test(*data, n_jobs=jobs)


@pytest.mark.parametrize("pool", [[], [np.nan], [np.inf], [[1, 2]], 1.])
def test_invalid_pool(data, pool):
    with pytest.raises(ValueError, match="null_pool"):
        cit_test(*data, null_pool=pool)


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_data(data, index, bad):
    arrays = [a.copy() for a in data]
    arrays[index][0] = bad
    with pytest.raises(ValueError, match="finite"):
        cit_test(*arrays)


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("bad_shape", [(0,), (1,), (12, 0), (12, 1, 1)])
def test_invalid_data_shape(data, index, bad_shape):
    arrays = list(data)
    arrays[index] = np.ones(bad_shape)
    with pytest.raises(ValueError):
        cit_test(*arrays)


@pytest.mark.parametrize("index", range(3))
def test_mismatched_rows(data, index):
    arrays = list(data)
    arrays[index] = arrays[index][:-1]
    with pytest.raises(ValueError, match="same number"):
        cit_test(*arrays)


@pytest.mark.parametrize("index", [0, 1])
def test_multivariate_tested_variables_require_projection(data, index):
    arrays = list(data)
    arrays[index] = np.column_stack((arrays[index], arrays[index]))
    with pytest.raises(ValueError, match="scalar X and Y"):
        cit_test(*arrays)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_statistic_rejected(data, monkeypatch, bad):
    monkeypatch.setattr(api, "rho_hat_multivariate", lambda *a, **kw: bad)
    with pytest.raises(ValueError, match="statistic must be finite"):
        cit_test(*data, null_pool=np.arange(19.))


def test_all_bandwidth_entry_points_default_to_0_8(data):
    from rp_ci.cit.rosenblatt import rosenblatt_XY_given_Z, rosenblatt_Z_only
    for function in (cit_test, band_width, rosenblatt_transform,
                     rosenblatt_XY_given_Z, rosenblatt_Z_only,
                     scalar_projected_statistics):
        assert inspect.signature(function).parameters["h_scale"].default == .8
    assert cit_test(*data, null_pool=np.arange(19.)) == cit_test(
        *data, null_pool=np.arange(19.), h_scale=.8
    )


def test_rp_framework_callback_agrees_with_batch_statistics():
    rng = np.random.default_rng(28)
    x, y, z = rng.normal(size=(12, 3)), rng.normal(size=(12, 4)), rng.normal(size=12)
    pool = simulate_null_pool(12, 1, 1, 1, B=99, seed=610, n_jobs=1)
    def base_test(x, y, z):
        return cit_test(
            x, y, z, null_pool=pool, alpha=.05 / 3, h_scale=.8,
        ).pvalue
    result = rp_test(x, y, z, base_test, k=3, seed=33)
    projection_rng = np.random.default_rng(33)
    xp = batch_unit_random_projection(x, projection_rng, n_proj=3, target_dim=1)
    yp = batch_unit_random_projection(y, projection_rng, n_proj=3, target_dim=1)
    stats = scalar_projected_statistics(xp, yp, z)
    np.testing.assert_array_equal(result.pvalues, plus_one_pvalues(stats, pool))
    assert result.reject == (result.min_pvalue <= .05 / 3)


def test_rp_default_matches_explicit_0_8_and_preserves_explicit_1_2():
    rng = np.random.default_rng(20261014)
    n, k = 16, 5
    z = rng.normal(size=(n, 1))
    x = z + rng.normal(size=(n, 3))
    y = z + rng.normal(size=(n, 4))
    projection_rng = np.random.default_rng(731)
    xp = batch_unit_random_projection(x, projection_rng, n_proj=k, target_dim=1)
    yp = batch_unit_random_projection(y, projection_rng, n_proj=k, target_dim=1)
    pool = simulate_null_pool(n, 1, 1, 1, B=199, seed=884, n_jobs=1)

    default = scalar_projected_statistics(xp, yp, z)
    explicit = scalar_projected_statistics(xp, yp, z, h_scale=.8)
    legacy = scalar_projected_statistics(xp, yp, z, h_scale=1.2)
    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_allclose(
        legacy,
        [3.5489369107890325, 3.445509475045755, 3.570858186208898,
         4.152756289322796, 5.321079689374499],
        rtol=2e-13, atol=2e-13,
    )
    np.testing.assert_array_equal(
        plus_one_pvalues(default, pool), [.39, .445, .795, .455, .125]
    )
    np.testing.assert_array_equal(
        plus_one_pvalues(legacy, pool), [.59, .635, .59, .375, .15]
    )
    assert not np.any(plus_one_pvalues(default, pool) <= .05 / k)
    assert not np.any(plus_one_pvalues(legacy, pool) <= .05 / k)
