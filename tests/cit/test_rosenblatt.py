import numpy as np
import pytest

from rp_ci.cit._utils import _as_2d
from rp_ci.cit.rosenblatt import (
    ROSENBLATT_VERSION,
    band_width,
    conditional_cdf,
    rosenblatt_transform,
    rosenblatt_XY_given_Z,
    rosenblatt_Z_only,
)


def _reference_cdf(target, conditioning, h_scale):
    n, d = conditioning.shape
    sd = np.std(conditioning, axis=0, ddof=1)
    sd = np.where(sd < 1e-8, 1.0, sd)
    h = 1.06 * sd * n ** (-1.0 / (4.0 + d)) * h_scale
    h = np.where(h < 1e-8, 1.0, h)
    result = []
    for i in range(n):
        weights = [np.exp(-0.5 * sum(((conditioning[j] - conditioning[i]) / h) ** 2))
                   for j in range(n)]
        result.append(sum(weights[j] for j in range(n) if target[j] <= target[i]) / sum(weights))
    return np.asarray(result)


def _reference_transform(x, y, z, h_scale):
    w = np.empty_like(z)
    w[:, 0] = [np.count_nonzero(z[:, 0] <= value) / len(z) for value in z[:, 0]]
    for j in range(1, z.shape[1]):
        w[:, j] = _reference_cdf(z[:, j], z[:, :j], h_scale)
    uv = []
    for values in (x, y):
        transformed = np.empty_like(values)
        for j in range(values.shape[1]):
            transformed[:, j] = _reference_cdf(
                values[:, j], np.column_stack((z, values[:, :j])), h_scale
            )
        uv.append(transformed)
    return *uv, w


@pytest.fixture
def data():
    rng = np.random.default_rng(73914)
    return tuple(rng.normal(size=(15, d)) for d in (3, 2, 2))


@pytest.mark.parametrize("dims", [(1, 1, 1), (3, 2, 2)])
@pytest.mark.parametrize("h_scale", [.7, 1.2, 1e-9])
def test_transform_matches_independent_formula(dims, h_scale):
    rng = np.random.default_rng(10413)
    arrays = tuple(rng.normal(size=(19, d)) for d in dims)
    before = [a.copy() for a in arrays]
    expected = _reference_transform(*arrays, h_scale)
    actual = rosenblatt_transform(*arrays, h_scale=h_scale)
    for observed, target, original, unchanged in zip(actual, expected, arrays, before):
        np.testing.assert_allclose(observed, target, rtol=2e-14, atol=2e-14)
        np.testing.assert_array_equal(original, unchanged)
        assert observed.shape == original.shape
        assert np.all((observed >= 0) & (observed <= 1))


def test_ties_use_right_ecdf_and_self_including_cdf():
    x = np.array([0., 1., 2.])
    z = np.zeros_like(x)
    u, v, w = rosenblatt_transform(x, x, z)
    np.testing.assert_array_equal(u[:, 0], [1 / 3, 2 / 3, 1.])
    np.testing.assert_array_equal(v, u)
    np.testing.assert_array_equal(w, np.ones((3, 1)))
    tied = np.array([0., 0., 1., 2., 2.])
    np.testing.assert_array_equal(rosenblatt_Z_only(tied)[:, 0], [.4, .4, .6, 1., 1.])


def test_row_permutation_preserves_tie_results():
    x = np.array([[0., 1.], [0., 1.], [1., 0.], [2., 0.], [2., 3.]])
    y = np.array([[1.], [1.], [0.], [2.], [2.]])
    z = np.array([[0.], [0.], [1.], [1.], [1.]])
    order = np.array([4, 1, 3, 0, 2])
    expected = rosenblatt_transform(x, y, z)
    actual = rosenblatt_transform(x[order], y[order], z[order])
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b[order], atol=2e-14, rtol=2e-14)


def test_isolated_observations_retain_self_weight():
    x = np.arange(3.)[:, None]
    z = x * 1e5
    u, v = rosenblatt_XY_given_Z(x, x, z, hZ=np.array([1e-8]))
    np.testing.assert_array_equal(u, np.ones_like(x))
    np.testing.assert_array_equal(v, u)


def test_explicit_legacy_exclusion_and_weight_fallback():
    t = np.arange(3.)
    c = np.zeros((3, 1))
    assert conditional_cdf(t, c, 0., [0.], [1.]) == 1 / 3
    assert conditional_cdf(t, c, 0., [0.], [1.], exclude=0) == 0.
    assert conditional_cdf(t, c, 1., [1e6], [1.]) == 2 / 3
    assert conditional_cdf(t, c, 1., [1e6], [1.], exclude=0) == .5


def test_precomputed_values_match_fresh_transform(data):
    x, y, z = data
    w = rosenblatt_Z_only(z, h_scale=.8)
    hz = band_width(z, h_scale=.8)
    before = w.copy(), hz.copy()
    cached = rosenblatt_transform(x, y, z, h_scale=.8, W_precomputed=w, hZ_precomputed=hz)
    fresh = rosenblatt_transform(x, y, z, h_scale=.8)
    for a, b in zip(cached, fresh):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(w, before[0])
    np.testing.assert_array_equal(hz, before[1])


def test_bandwidth_policy_and_floors():
    a = np.column_stack((np.arange(5.), np.ones(5), np.arange(5.) * 1e-10))
    sd = a.std(axis=0, ddof=1)
    sd[sd < 1e-8] = 1.
    expected = 1.06 * sd * 5 ** (-1 / 7) * .8
    np.testing.assert_array_equal(band_width(a), expected)
    legacy = 1.06 * sd * 5 ** (-1 / 7) * 1.4
    np.testing.assert_array_equal(band_width(a, h_scale=1.4), legacy)
    np.testing.assert_array_equal(band_width(a, h_scale=1e-12), np.ones(3))


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf, np.array([1.])])
def test_invalid_scale_rejected_even_with_precomputed_values(data, bad):
    x, y, z = data
    calls = (
        lambda: band_width(z, bad),
        lambda: rosenblatt_Z_only(z, bad),
        lambda: rosenblatt_XY_given_Z(x, y, z, bad, hZ=np.ones(2)),
        lambda: rosenblatt_transform(x, y, z, bad, W_precomputed=np.zeros_like(z), hZ_precomputed=np.ones(2)),
    )
    for call in calls:
        with pytest.raises(ValueError, match="h_scale"):
            call()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_nonfinite_raw_data_rejected(data, index, bad):
    arrays = [a.copy() for a in data]
    arrays[index][0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        rosenblatt_transform(*arrays)


@pytest.mark.parametrize("bad", [np.empty((0, 1)), np.empty((15, 0)), np.ones((1, 1)), np.ones((15, 1, 1))])
def test_empty_or_malformed_data_rejected(data, bad):
    for index in range(3):
        arrays = list(data)
        arrays[index] = bad
        with pytest.raises(ValueError):
            rosenblatt_transform(*arrays)


def test_row_mismatch_rejected(data):
    x, y, z = data
    with pytest.raises(ValueError, match="same number of rows"):
        rosenblatt_transform(x[:-1], y, z)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1e-4, 2.])
def test_invalid_precomputed_w_rejected(data, bad):
    with pytest.raises(ValueError, match="W_precomputed"):
        rosenblatt_transform(*data, W_precomputed=np.full_like(data[2], bad))


def test_precomputed_w_shape_rejected(data):
    with pytest.raises(ValueError, match="W_precomputed"):
        rosenblatt_transform(*data, W_precomputed=np.zeros((15, 1)))


@pytest.mark.parametrize("hz", [[0., 1.], [-1., 1.], [np.nan, 1.], [np.inf, 1.], [1.]])
def test_invalid_precomputed_bandwidth_rejected(data, hz):
    with pytest.raises(ValueError, match="bandwidth|Dimension mismatch"):
        rosenblatt_transform(*data, hZ_precomputed=np.asarray(hz))


@pytest.mark.parametrize("exclude", [-1, 3, .5])
def test_invalid_legacy_exclusion_rejected(exclude):
    with pytest.raises(ValueError, match="exclude"):
        conditional_cdf(np.arange(3.), np.zeros((3, 1)), 0., [0.], [1.], exclude=exclude)


def test_local_shape_helper_and_version():
    np.testing.assert_array_equal(_as_2d([1, 2]), np.array([[1.], [2.]]))
    for invalid in (1., np.ones((2, 2, 2))):
        with pytest.raises(ValueError, match="1D or 2D"):
            _as_2d(invalid)
    assert ROSENBLATT_VERSION == "self_including_ecdf_n_v3_guarded"
