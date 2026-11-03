import numpy as np
import pytest

from rp_ci.cit import calibration as cal
from rp_ci.cit.rosenblatt import rosenblatt_transform
from rp_ci.cit.statistic import rho_hat_multivariate


def test_plus_one_counts_ties_and_preserves_shape():
    pool = np.array([2., 0., 2., 1.])
    stats = np.array([[-1., 0., 1.], [2., 3., 4.]])
    before = pool.copy(), stats.copy()
    expected = np.array([[1., 1., .8], [.6, .2, .2]])
    np.testing.assert_array_equal(cal.plus_one_pvalues(stats, pool), expected)
    np.testing.assert_array_equal(
        cal.plus_one_pvalues(stats, np.sort(pool), assume_sorted=True), expected,
    )
    scalar = cal.plus_one_pvalues(3., pool)
    assert scalar.shape == () and scalar == .2
    for actual, original in zip((pool, stats), before):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("B", [1, 2, 9, 20])
@pytest.mark.parametrize("tied", [False, True])
def test_critical_value_matches_pvalues_at_grid_boundaries(B, tied):
    pool = np.zeros(B) if tied else np.arange(B, dtype=float) // 2
    stats = np.unique(np.concatenate([
        [-1., B + 1.], pool, np.nextafter(pool, -np.inf),
        np.nextafter(pool, np.inf),
    ]))
    grid = np.arange(1, B + 1) / (B + 1.)
    levels = np.unique(np.concatenate([
        [.05, .05 / 1000], grid, np.nextafter(grid, 0.),
        np.nextafter(grid, 1.),
    ]))
    pvalues = cal.plus_one_pvalues(stats, pool)
    for level in levels:
        cutoff = cal.plus_one_critical_value(pool, level)
        np.testing.assert_array_equal(stats > cutoff, pvalues <= level)
    assert np.isinf(cal.plus_one_critical_value(pool, np.nextafter(grid[0], 0.)))
    assert cal.plus_one_critical_value(pool, grid[0]) == np.max(pool)


def test_large_budget_rank_boundaries():
    # No simulation is needed to check the experimental budget's rank grid.
    B = 500_000
    pool = np.arange(B, dtype=float)
    for level in [.05 / 1000, 1 / (B + 1), 25 / (B + 1), B / (B + 1)]:
        for alpha in [np.nextafter(level, 0.), level, np.nextafter(level, 1.)]:
            cutoff = cal.plus_one_critical_value(pool, alpha)
            stats = np.array([-1., B + 1.])
            if np.isfinite(cutoff):
                stats = np.append(stats, [np.nextafter(cutoff, -np.inf), cutoff,
                                          np.nextafter(cutoff, np.inf)])
            np.testing.assert_array_equal(
                stats > cutoff, cal.plus_one_pvalues(stats, pool) <= alpha,
            )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_observation_rejected(bad):
    with pytest.raises(ValueError, match="Observed statistics"):
        cal.plus_one_pvalues([0., bad], [1., 2.])


@pytest.mark.parametrize("pool", [[], [np.nan], [np.inf], [-np.inf]])
def test_invalid_plus_one_pool_rejected(pool):
    with pytest.raises(ValueError, match="null_pool"):
        cal.plus_one_pvalues(0., pool)
    with pytest.raises(ValueError, match="null_pool"):
        cal.plus_one_critical_value(pool, .05)


@pytest.mark.parametrize("alpha", [0., 1., -.1, 1.1, np.nan, np.inf])
def test_invalid_levels_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        cal.plus_one_critical_value([0.], alpha)


@pytest.mark.parametrize("B", [1, 2, 9, 12])
@pytest.mark.parametrize("seed", [0, 2**40 + 813])
def test_reference_pool_matches_indexed_uniform_streams(B, seed):
    expected = []
    for index in range(B):
        rng = np.random.default_rng(np.random.SeedSequence([
            seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, index,
        ]))
        u, v, w = (rng.uniform(size=(8, d)) for d in (1, 2, 1))
        expected.append(rho_hat_multivariate(u, v, w))
    actual = cal.simulate_null_pool(8, 1, 2, 1, B=B, seed=seed, n_jobs=1)
    np.testing.assert_array_equal(actual, expected)
    longer = cal.simulate_null_pool(8, 1, 2, 1, B=12, seed=seed, n_jobs=1)
    np.testing.assert_array_equal(actual, longer[:B])


@pytest.mark.parametrize("batch", [2, 5])
def test_reference_pool_independent_of_worker_count_and_batch(batch):
    serial = cal.simulate_null_pool(8, 1, 2, 1, B=9, seed=42, n_jobs=1)
    parallel = cal.simulate_null_pool(
        8, 1, 2, 1, B=9, seed=42, n_jobs=2, batch=batch,
    )
    np.testing.assert_array_equal(parallel, serial)


def test_process_fallback_retains_reference_streams(monkeypatch):
    real_parallel = cal.Parallel
    backends = []

    def restricted_parallel(*args, **kwargs):
        backends.append(kwargs["backend"])
        if kwargs["backend"] == "loky":
            raise PermissionError("test process restriction")
        return real_parallel(*args, **kwargs)

    monkeypatch.setattr(cal, "Parallel", restricted_parallel)
    expected = cal.simulate_null_pool(8, 1, 1, 1, B=7, seed=13, n_jobs=1)
    with pytest.warns(RuntimeWarning, match="falling back to deterministic threads"):
        actual = cal.simulate_null_pool(8, 1, 1, 1, B=7, seed=13, n_jobs=2, batch=3)
    np.testing.assert_array_equal(actual, expected)
    assert backends == ["loky", "threading", "threading", "threading"]


@pytest.mark.parametrize("B", [0, -1, True, np.bool_(False), 2.5, np.nan])
def test_invalid_reference_budget_rejected(B):
    with pytest.raises(ValueError, match="B must be a positive integer"):
        cal.simulate_null_pool(8, 1, 1, 1, B=B, n_jobs=1)


@pytest.mark.parametrize("change", [
    {"n": 1}, {"du": 0}, {"dv": 0}, {"dw": 0}, {"batch": 0}, {"seed": -1},
])
def test_invalid_reference_configuration_rejected(change):
    kwargs = dict(n=8, du=1, dv=1, dw=1, B=1, seed=0, n_jobs=1)
    kwargs.update(change)
    with pytest.raises(ValueError):
        cal.simulate_null_pool(**kwargs)


def test_numpy_integer_budget_and_unseeded_pool():
    pool = cal.simulate_null_pool(8, 1, 1, 1, B=np.int64(2), n_jobs=1)
    assert pool.shape == (2,) and np.isfinite(pool).all()


def test_transform_statistic_calibration_integration():
    rng = np.random.default_rng(20261013)
    x, y, z = (rng.normal(size=(12, 1)) for _ in range(3))
    u, v, w = rosenblatt_transform(x, y, z)
    stat = rho_hat_multivariate(u, v, w)
    pool = cal.simulate_null_pool(12, 1, 1, 1, B=19, seed=714, n_jobs=1)
    pvalue = cal.plus_one_pvalues(stat, pool)
    assert pvalue == (1 + np.count_nonzero(pool >= stat)) / 20
    assert (pvalue <= .2) == (stat > cal.plus_one_critical_value(pool, .2))
