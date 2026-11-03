import numpy as np
import pytest

from rp_ci.cit.rosenblatt import rosenblatt_transform
from rp_ci.cit.statistic import (
    RhoHatCachedW, compute_KW, dist_mat_l1, prod_bracket, rho_hat_multivariate,
)


def _reference_statistic(u, v, w):
    # Expanded products and double sum, independent of production helpers.
    e = np.exp(1.)
    c0 = 1 / (13 / e**3 - 40 / e**2 + 13 / e)
    n = len(u)

    def centered_kernel(a, i, j):
        gi = np.prod(2 - np.exp(-a[i]) - np.exp(a[i] - 1))
        gj = np.prod(2 - np.exp(-a[j]) - np.exp(a[j] - 1))
        return np.exp(-np.sum(np.abs(a[i] - a[j]))) + (2 / e)**a.shape[1] - gi - gj

    total = 0.
    for i in range(n):
        for j in range(n):
            total += (centered_kernel(u, i, j) * centered_kernel(v, i, j)
                      * np.exp(-np.sum(np.abs(w[i] - w[j]))))
    return c0 * total / n


@pytest.fixture
def transformed():
    rng = np.random.default_rng(18137)
    return tuple(rng.uniform(size=(8, d)) for d in (2, 3, 1))


@pytest.mark.parametrize("dims", [(2, 1, 1, 1), (9, 2, 3, 1), (12, 5, 4, 2), (8, 200, 200, 1)])
@pytest.mark.parametrize("endpoints", [False, True])
def test_all_paths_match_expanded_formula(dims, endpoints):
    n, du, dv, dw = dims
    rng = np.random.default_rng(99137)
    u, v, w = (rng.uniform(size=(n, d)) for d in (du, dv, dw))
    if endpoints:
        for a in (u, v, w):
            a[0, :] = 0.
            a[-1, :] = 1.
    before = [a.copy() for a in (u, v, w)]
    expected = _reference_statistic(u, v, w)
    kw = compute_KW(w)
    fresh = rho_hat_multivariate(u, v, w)
    np.testing.assert_allclose(fresh, expected, rtol=2e-11, atol=2e-11)
    assert rho_hat_multivariate(u, v, K_W=kw) == fresh
    assert RhoHatCachedW(w)(u, v) == fresh
    for a, b in zip((u, v, w), before):
        np.testing.assert_array_equal(a, b)


def test_statistic_includes_n_and_diagonal_terms(transformed):
    u, v, w = transformed
    original = rho_hat_multivariate(u, v, w)
    repeated = [np.repeat(a, 2, axis=0) for a in (u, v, w)]
    np.testing.assert_allclose(rho_hat_multivariate(*repeated), 2 * original, rtol=2e-14)
    constant = np.full((5, 1), .5)
    np.testing.assert_allclose(
        rho_hat_multivariate(constant, constant, constant),
        _reference_statistic(constant, constant, constant), rtol=2e-11,
    )


def test_helpers_match_their_formulas():
    m = np.array([[0., 1.], [.2, .7], [1., 0.]])
    distances = np.array([[sum(abs(a - b)) for b in m] for a in m])
    np.testing.assert_array_equal(dist_mat_l1(m), distances)
    np.testing.assert_allclose(prod_bracket(m), np.prod(2 - np.exp(-m) - np.exp(m - 1), axis=1))
    np.testing.assert_array_equal(compute_KW(m), np.exp(-distances))
    # Zero off-diagonals from underflow remain valid kernel entries.
    wide = np.array([np.zeros(800), np.ones(800)])
    np.testing.assert_array_equal(compute_KW(wide), np.eye(2))


def test_vectors_match_column_inputs():
    u, v, w = np.linspace(0., 1., 8), np.linspace(.8, .1, 8), np.linspace(.1, .5, 8)
    expected = rho_hat_multivariate(u[:, None], v[:, None], w[:, None])
    assert rho_hat_multivariate(u, v, w) == expected
    assert RhoHatCachedW(w)(u, v) == expected


def test_row_permutation_and_uv_exchange(transformed):
    u, v, w = transformed
    expected = rho_hat_multivariate(u, v, w)
    order = np.random.default_rng(8).permutation(len(u))
    np.testing.assert_allclose(rho_hat_multivariate(u[order], v[order], w[order]), expected, rtol=2e-14)
    assert rho_hat_multivariate(v, u, w) == expected


def test_transform_to_statistic_integration():
    rng = np.random.default_rng(10613)
    arrays = tuple(rng.normal(size=(13, d)) for d in (3, 2, 2))
    u, v, w = rosenblatt_transform(*arrays)
    expected = _reference_statistic(u, v, w)
    np.testing.assert_allclose(rho_hat_multivariate(u, v, w), expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(RhoHatCachedW(w)(u, v), expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -.01, 1.01])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_invalid_transformed_values_rejected(transformed, index, bad):
    arrays = [a.copy() for a in transformed]
    arrays[index][0, 0] = bad
    with pytest.raises(ValueError, match="finite|\\[0, 1\\]"):
        rho_hat_multivariate(*arrays)
    if index == 2:
        with pytest.raises(ValueError):
            compute_KW(arrays[2])
        with pytest.raises(ValueError):
            RhoHatCachedW(arrays[2])
    else:
        with pytest.raises(ValueError):
            RhoHatCachedW(arrays[2])(*arrays[:2])
        with pytest.raises(ValueError):
            rho_hat_multivariate(*arrays[:2], K_W=compute_KW(arrays[2]))


@pytest.mark.parametrize("bad", [np.empty((0, 1)), np.empty((8, 0)), np.ones((1, 1)), np.ones((8, 1, 1))])
def test_invalid_shapes_rejected(transformed, bad):
    u, v, w = transformed
    for index in range(3):
        arrays = [u, v, w]
        arrays[index] = bad
        with pytest.raises(ValueError):
            rho_hat_multivariate(*arrays)
    with pytest.raises(ValueError):
        RhoHatCachedW(bad)
    with pytest.raises(ValueError):
        RhoHatCachedW(w)(bad, v)


def test_missing_w_and_mismatched_rows(transformed):
    u, v, w = transformed
    calls = (
        lambda: rho_hat_multivariate(u, v),
        lambda: rho_hat_multivariate(u[:-1], v, w),
        lambda: rho_hat_multivariate(u, v, w[:-1]),
        lambda: RhoHatCachedW(w)(u[:-1], v),
        lambda: RhoHatCachedW(w)(u[:-1], v[:-1]),
    )
    for call in calls:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("kind", ["negative", "nan", "inf", "above_one", "asymmetric", "diagonal", "shape"])
def test_invalid_external_kernel_rejected_in_both_paths(transformed, kind):
    u, v, w = transformed
    kw = compute_KW(w)
    if kind in {"negative", "nan", "inf", "above_one"}:
        kw[0, 1] = kw[1, 0] = {"negative": -1., "nan": np.nan, "inf": np.inf, "above_one": 1.1}[kind]
    elif kind == "asymmetric":
        kw[0, 1] *= .5
    elif kind == "diagonal":
        kw[0, 0] = .5
    else:
        kw = kw[:-1]
    with pytest.raises(ValueError, match="K_W"):
        rho_hat_multivariate(u, v, K_W=kw)
    cached = RhoHatCachedW(w)
    cached.K_W = kw
    with pytest.raises(ValueError, match="K_W"):
        cached(u, v)


def test_supplied_kernel_precedence_and_no_mutation(transformed):
    u, v, w = transformed
    kw = compute_KW(w)
    expected = rho_hat_multivariate(u, v, w)
    assert rho_hat_multivariate(u, v, W=np.full_like(w, np.nan), K_W=kw) == expected
    kw[0, 1] += 1e-14
    kw[0, 0] -= 1e-14
    before = kw.copy()
    assert np.isfinite(rho_hat_multivariate(u, v, K_W=kw))
    np.testing.assert_array_equal(kw, before)


def test_cache_kernel_is_reused(transformed):
    u, v, w = transformed
    cached = RhoHatCachedW(w)
    kw = cached.K_W
    cached(u, v)
    cached(v, u)
    assert cached.K_W is kw
