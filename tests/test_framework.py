import numpy as np
import pytest

from rp_ci import RPTestResult, aggregate, rp_test
from rp_ci.aggregation import topr


def _experimental_projections(X, Y, k, seed):
    rng = np.random.default_rng(seed)
    projected = []
    for data in (X, Y):
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        n, d = data.shape
        directions = rng.normal(size=(d, k))
        directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12
        projected.append((data @ directions).reshape(n, k, 1).astype(float))
    return projected


def _stub_pvalue(x, y, z):
    return float(1.0 / (1.0 + np.mean(x**2) + np.mean(y**2) + np.mean(z**2)))


@pytest.mark.parametrize("seed", [0, 42, 20260724])
@pytest.mark.parametrize("k", [1, 3, 10, 1000])
@pytest.mark.parametrize("dims", [(1, 1, 1), (4, 3, 1), (5, 2, 3)])
def test_matches_experimental_projection_and_decision(seed, k, dims):
    rng = np.random.default_rng(12)
    X, Y, Z = [rng.normal(size=(13, d)) for d in dims]
    before = [data.copy() for data in (X, Y, Z)]
    expected_x, expected_y = _experimental_projections(X, Y, k, seed)
    calls = []

    def base_test(x, y, z):
        index = len(calls)
        np.testing.assert_array_equal(x, expected_x[:, index, :])
        np.testing.assert_array_equal(y, expected_y[:, index, :])
        assert z is Z
        calls.append(index)
        return _stub_pvalue(x, y, z)

    result = rp_test(X, Y, Z, base_test, k=k, alpha=0.05, seed=seed)
    expected = np.array([
        _stub_pvalue(expected_x[:, i, :], expected_y[:, i, :], Z) for i in range(k)
    ])
    assert isinstance(result, RPTestResult)
    assert len(calls) == k
    np.testing.assert_array_equal(result.pvalues, expected)
    assert result.min_pvalue == float(np.min(expected))
    assert result.threshold == 0.05 / k
    assert result.reject is bool(np.min(expected) <= 0.05 / k)
    for actual, original in zip((X, Y, Z), before):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("k", [1, 3, 10, 1000])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.5])
@pytest.mark.parametrize("position", ["below", "equal", "above"])
def test_uses_all_base_pvalues_and_original_boundary(k, alpha, position):
    threshold = float(alpha / k)
    minimum = {
        "below": np.nextafter(threshold, 0.0),
        "equal": threshold,
        "above": np.nextafter(threshold, np.inf),
    }[position]
    expected = np.ones(k)
    expected[0] = minimum
    values = iter(expected)
    X = np.arange(12.0).reshape(6, 2)

    result = rp_test(
        X, X, X[:, :1], lambda x, y, z: next(values), k=k, alpha=alpha, seed=42
    )
    assert next(values, None) is None
    np.testing.assert_array_equal(result.pvalues, expected)
    assert result.reject is (position != "above")
    assert result.min_pvalue == minimum
    assert result.threshold == threshold


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 5, 6],
        np.arange(12.0).reshape(6, 2),
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.arange(36.0).reshape(6, 6)[:, ::2],
    ],
)
def test_input_conversion_and_unprojected_z(data):
    before = np.array(data, copy=True)
    expected_z = np.asarray(data, dtype=float)
    if expected_z.ndim == 1:
        expected_z = expected_z[:, None]
    calls = []

    def base_test(x, y, z):
        assert x.shape == y.shape == (6, 1)
        assert x.dtype == y.dtype == z.dtype == np.dtype(float)
        np.testing.assert_array_equal(z, expected_z)
        calls.append(z)
        return 0.5

    result = rp_test(data, data, data, base_test, k=3, seed=1)
    assert len(calls) == 3
    assert all(z is calls[0] for z in calls)
    np.testing.assert_array_equal(result.pvalues, np.full(3, 0.5))
    np.testing.assert_array_equal(data, before)


def test_repeatability_and_default_parameters():
    X = np.arange(12.0).reshape(6, 2)
    Y = X[:, ::-1].copy()
    Z = X[:, :1]
    first = rp_test(X, Y, Z, _stub_pvalue, seed=42)
    second = rp_test(X, Y, Z, _stub_pvalue, seed=42)
    different = rp_test(X, Y, Z, _stub_pvalue, seed=43)
    np.testing.assert_array_equal(first.pvalues, second.pvalues)
    assert not np.array_equal(first.pvalues, different.pvalues)
    assert first.reject == second.reject
    assert first.min_pvalue == second.min_pvalue
    assert first.threshold == second.threshold == 0.05 / 10
    assert first.pvalues.shape == (10,)


def test_base_test_exception_is_not_suppressed():
    def base_test(x, y, z):
        raise RuntimeError("Base test failed.")

    data = np.ones((5, 2))
    with pytest.raises(RuntimeError, match="Base test failed"):
        rp_test(data, data, data, base_test, seed=1)


@pytest.mark.parametrize(
    "pvalue",
    [0, 1, 0.05, np.float32(0.5), np.float64(0.5), np.array(0.5)],
)
def test_valid_scalar_pvalues(pvalue):
    data = np.ones((5, 2))
    result = rp_test(data, data, data, lambda x, y, z: pvalue, k=2, seed=1)
    np.testing.assert_array_equal(result.pvalues, np.full(2, float(pvalue)))


@pytest.mark.parametrize(
    "pvalue",
    [
        np.nan, np.inf, -np.inf, -0.1, 1.1, None, "0.5", True, 0.5 + 0j,
        [0.5], np.array([0.5]), np.array([[0.5]]), (0.5, 0.1),
    ],
)
def test_invalid_base_pvalue_stops_execution(pvalue):
    calls = []

    def base_test(x, y, z):
        calls.append(None)
        return 0.5 if len(calls) == 1 else pvalue

    data = np.ones((5, 2))
    with pytest.raises(ValueError, match="finite scalar p-value"):
        rp_test(data, data, data, base_test, k=3, seed=1)
    assert len(calls) == 2


@pytest.mark.parametrize("data", [[], np.empty((0, 2))])
def test_empty_sample(data):
    with pytest.raises(ValueError, match="at least one sample"):
        rp_test(data, data, data, lambda x, y, z: pytest.fail("Base test was called."))


@pytest.mark.parametrize("index", [0, 1])
def test_missing_tested_features(index):
    data = [np.ones((5, 2)) for _ in range(3)]
    data[index] = np.empty((5, 0))
    with pytest.raises(ValueError, match="at least one feature"):
        rp_test(*data, lambda x, y, z: pytest.fail("Base test was called."))


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, np.nan, np.inf])
def test_invalid_alpha(alpha):
    data = np.ones((5, 2))
    with pytest.raises(ValueError, match="alpha must be in"):
        rp_test(data, data, data, _stub_pvalue, alpha=alpha)


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(k):
    data = np.ones((5, 2))
    with pytest.raises(ValueError, match="k must be >= 1"):
        rp_test(data, data, data, _stub_pvalue, k=k)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_mismatched_rows(index):
    data = [np.ones((5, 2)) for _ in range(3)]
    data[index] = np.ones((4, 2))
    with pytest.raises(ValueError, match="same number of rows"):
        rp_test(*data, _stub_pvalue)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("invalid", [np.array(1.0), np.zeros((2, 3, 4))])
def test_invalid_array_dimensions(index, invalid):
    data = [np.ones((5, 2)) for _ in range(3)]
    data[index] = invalid
    with pytest.raises(ValueError, match="Input must be 1D or 2D array"):
        rp_test(*data, _stub_pvalue)


def test_default_result_reports_max_rule():
    X = np.arange(12.0).reshape(6, 2)
    result = rp_test(X, X, X[:, :1], _stub_pvalue, k=5, seed=3)
    assert (result.rule, result.r, result.rth_pvalue) == ("max", 1, result.min_pvalue)


@pytest.mark.parametrize("rule", ["beta", "markov"])
@pytest.mark.parametrize("r", [1, 2, 4])
def test_topr_uses_the_same_projection_pvalues(rule, r):
    rng = np.random.default_rng(5)
    X, Y, Z = rng.normal(size=(9, 3)), rng.normal(size=(9, 2)), rng.normal(size=(9, 1))
    base = rp_test(X, Y, Z, _stub_pvalue, k=8, seed=11)
    result = rp_test(X, Y, Z, _stub_pvalue, k=8, seed=11, aggregation=rule, r=r)
    np.testing.assert_array_equal(result.pvalues, base.pvalues)
    assert (result.reject, result.rth_pvalue, result.threshold) == topr(
        base.pvalues, alpha=0.05, r=r, rule=rule)
    assert (result.rule, result.r, result.min_pvalue) == (rule, r, base.min_pvalue)


def test_aggregate_max_matches_rp_test():
    rng = np.random.default_rng(6)
    X, Y, Z = rng.normal(size=(9, 3)), rng.normal(size=(9, 2)), rng.normal(size=(9, 1))
    result = rp_test(X, Y, Z, _stub_pvalue, k=8, seed=2)
    again = aggregate(result.pvalues, alpha=0.05)
    assert (again.reject, again.min_pvalue, again.threshold) == (
        result.reject, result.min_pvalue, result.threshold)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(aggregation="max", r=2), "r applies only"),
        (dict(aggregation="beta"), "requires r"),
        (dict(aggregation="markov", r=4), "r must satisfy"),
        (dict(aggregation="holm", r=2), "aggregation must be"),
    ],
)
def test_invalid_aggregation_options(kwargs, match):
    data = np.ones((5, 2))
    with pytest.raises(ValueError, match=match):
        rp_test(data, data, data, lambda x, y, z: pytest.fail("Base test was called."),
                k=3, **kwargs)
