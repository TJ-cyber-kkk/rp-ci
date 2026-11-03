import numpy as np
import pytest
from scipy.stats import norm

from rp_ci import rp_test
from rp_ci.gcm import GCMResult, gcm_test, ols_residuals


def _data(seed, n=120, dx=1, dy=1, dz=3, coupling=0.0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, dz))
    B = rng.normal(size=(dz, dx + dy))
    shared = rng.normal(size=(n, 1))
    X = Z @ B[:, :dx] + rng.normal(size=(n, dx)) + coupling * shared
    Y = Z @ B[:, dx:] + rng.normal(size=(n, dy)) + coupling * shared
    return X, Y, Z


@pytest.mark.parametrize("dz", [1, 3, 10])
def test_ols_residuals_match_least_squares_with_intercept(dz):
    X, _, Z = _data(1, dx=4, dz=dz)
    design = np.column_stack([np.ones(len(Z)), Z])
    expected = X - design @ np.linalg.lstsq(design, X, rcond=None)[0]
    np.testing.assert_allclose(ols_residuals(X, Z), expected, rtol=1e-10, atol=1e-12)


def test_ols_residuals_invariant_to_affine_z_and_redundant_columns():
    X, _, Z = _data(2, dx=4)
    shifted = 7.0 + Z * np.array([1e-3, 5.0, 300.0])
    redundant = np.column_stack([Z, np.ones(len(Z)), Z[:, 0]])
    for W in (shifted, redundant):
        np.testing.assert_allclose(ols_residuals(X, W), ols_residuals(X, Z),
                                   rtol=1e-8, atol=1e-10)


def test_ols_residuals_without_z_center():
    X, _, _ = _data(3, dx=4)
    np.testing.assert_allclose(ols_residuals(X, np.empty((len(X), 0))), X - X.mean(0))


def test_ols_residuals_validation():
    with pytest.raises(ValueError, match="same number of rows"):
        ols_residuals(np.ones((5, 2)), np.ones((4, 1)))
    with pytest.raises(ValueError, match="finite"):
        ols_residuals(np.array([[1.0], [np.nan], [2.0]]), np.ones((3, 1)))


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_gcm_matches_official_formula(seed):
    X, Y, Z = _data(seed, coupling=0.2)
    R = ols_residuals(X, Z)[:, 0] * ols_residuals(Y, Z)[:, 0]
    # Formula of GeneralisedCovarianceMeasure::gcm.test, univariate case.
    statistic = np.sqrt(len(R)) * R.mean() / np.sqrt(np.mean(R**2) - R.mean() ** 2)
    result = gcm_test(X, Y, Z, alpha=0.1)
    assert isinstance(result, GCMResult)
    np.testing.assert_allclose(result.statistic, statistic, rtol=1e-10)
    np.testing.assert_allclose(result.pvalue, 2 * norm.sf(abs(statistic)), rtol=1e-8)
    assert result.reject is (result.pvalue <= 0.1)


def test_gcm_is_scale_invariant():
    X, Y, Z = _data(7, coupling=0.2)
    base = gcm_test(X, Y, Z)
    small = gcm_test(1e-8 * X, 1e-8 * Y, Z)
    np.testing.assert_allclose(small.pvalue, base.pvalue, rtol=1e-8)


def test_zero_residuals_give_pvalue_one():
    X, Y, Z = _data(8)
    result = gcm_test(X, Y, Z, residualize=lambda V, W: np.zeros_like(V))
    assert (result.reject, result.pvalue, result.statistic) == (False, 1.0, 0.0)


@pytest.mark.parametrize("scale", [1.0, 1e-8, 1e8])
def test_perfect_fit_gives_pvalue_one_at_any_scale(scale):
    # dz >= n - 1: OLS interpolates and residuals are rounding noise.
    X, Y, Z = _data(10, n=50, dz=60, coupling=0.5)
    result = gcm_test(scale * X, scale * Y, Z)
    assert (result.reject, result.pvalue) == (False, 1.0)


def test_custom_residualizer_is_used():
    X, Y, Z = _data(9)
    calls = []

    def residualize(V, W):
        calls.append(V.shape)
        return ols_residuals(V, W)

    assert gcm_test(X, Y, Z, residualize=residualize) == gcm_test(X, Y, Z)
    assert calls == [X.shape, Y.shape]


@pytest.mark.parametrize(
    "args,kwargs,match",
    [
        ((np.ones((5, 2)), np.ones(5), np.ones(5)), {}, "scalar X and Y"),
        ((np.ones(5), np.ones(4), np.ones(5)), {}, "same number of rows"),
        ((np.ones(2), np.ones(2), np.ones(2)), {}, "n=3"),
        ((np.ones(5), np.ones(5), np.ones(5)), dict(alpha=1.0), "alpha"),
        ((np.ones(5), np.ones(5), np.ones(5)), dict(residualize=lambda V, Z: V[:3]),
         "residualize must return"),
    ],
)
def test_gcm_validation(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        gcm_test(*args, **kwargs)


def _rp_gcm(X, Y, Z, k, seed):
    return rp_test(X, Y, Z, lambda x, y, z: gcm_test(x, y, z).pvalue, k=k, seed=seed)


def test_rp_gcm_null_rejection_rate_is_controlled():
    rejections = [_rp_gcm(*_data(100 + i, n=400, dx=6, dy=6), k=100, seed=i).reject
                  for i in range(100)]
    assert np.mean(rejections) <= 0.1


def test_rp_gcm_detects_residual_covariance():
    rejections = [_rp_gcm(*_data(500 + i, n=300, dx=6, dy=6, coupling=0.6), k=100, seed=i).reject
                  for i in range(10)]
    assert np.mean(rejections) >= 0.9
