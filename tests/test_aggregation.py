import numpy as np
import pytest
from scipy.stats import beta

from rp_ci.aggregation import bonferroni_max, topr, topr_threshold


def _experimental_decision(pvalues, alpha, k):
    min_p = float(np.min(pvalues))
    threshold = float(alpha / k)
    return bool(min_p <= threshold), min_p, threshold


@pytest.mark.parametrize("seed", [0, 42, 20260724])
@pytest.mark.parametrize("k", [1, 3, 10, 1000])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.5])
def test_matches_experimental_decision(seed, k, alpha):
    pvalues = np.random.default_rng(seed).uniform(size=k)
    before = pvalues.copy()
    result = bonferroni_max(pvalues, alpha=alpha)
    assert result == _experimental_decision(pvalues, alpha, k)
    np.testing.assert_array_equal(pvalues, before)
    assert tuple(type(value) for value in result) == (bool, float, float)


@pytest.mark.parametrize("k", [1, 3, 10, 1000])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.5])
@pytest.mark.parametrize("position", ["below", "equal", "above"])
def test_rejection_boundary(k, alpha, position):
    threshold = float(alpha / k)
    minimum = {
        "below": np.nextafter(threshold, 0.0),
        "equal": threshold,
        "above": np.nextafter(threshold, np.inf),
    }[position]
    pvalues = np.ones(k)
    pvalues[-1] = minimum
    reject, min_pvalue, actual_threshold = bonferroni_max(pvalues, alpha=alpha)
    assert reject is (position != "above")
    assert min_pvalue == minimum
    assert actual_threshold == threshold


@pytest.mark.parametrize(
    "pvalues,alpha,expected",
    [
        ([0.01, 0.4, 0.8], 0.05, True),
        ([0.02, 0.4, 0.8], 0.05, False),
        ([0.05], 0.05, True),
        ([1.0], 0.05, False),
        ([0.0, 1.0], 0.05, True),
        ([0.02, 0.02, 0.02], 0.05, False),
    ],
)
def test_known_decisions(pvalues, alpha, expected):
    reject, minimum, threshold = bonferroni_max(pvalues, alpha=alpha)
    assert reject is expected
    assert minimum == min(pvalues)
    assert threshold == alpha / len(pvalues)


@pytest.mark.parametrize(
    "pvalues",
    [
        np.array([0.01, 0.2, 0.3], dtype=np.float32),
        np.array([0, 1, 1], dtype=int),
        np.array([0.2, 0.7, 0.01, 0.8, 0.3, 0.9])[::2],
    ],
)
def test_array_formats_and_input_preservation(pvalues):
    before = pvalues.copy()
    assert bonferroni_max(pvalues, alpha=0.05) == _experimental_decision(
        pvalues, 0.05, pvalues.size
    )
    np.testing.assert_array_equal(pvalues, before)


def test_projection_order_does_not_change_decision():
    pvalues = np.array([0.3, 0.004, 0.7, 0.2])
    assert bonferroni_max(pvalues, alpha=0.05) == bonferroni_max(
        pvalues[[2, 0, 3, 1]], alpha=0.05
    )


def test_returned_minimum_is_not_bonferroni_adjusted():
    _, minimum, threshold = bonferroni_max([0.01, 0.2, 0.9], alpha=0.05)
    assert minimum == 0.01
    assert threshold == 0.05 / 3


@pytest.mark.parametrize("b,expected", [(19998, False), (19999, True), (500000, True)])
def test_minimum_plus_one_pvalue_at_bonferroni_threshold(b, expected):
    pvalues = np.full(1000, 1.0 / (b + 1))
    reject, _, _ = bonferroni_max(pvalues, alpha=0.05)
    assert reject is expected


def _paper_threshold(rule, r):
    """As in the paper's R7 runner (k=1000, alpha=0.05, lambda=1.5)."""
    return float(beta.ppf(.05, r, 1001 - r) / 1.5) if rule == "beta" else .05 * r / 1000


@pytest.mark.parametrize("rule", ["beta", "markov"])
@pytest.mark.parametrize("r", [2, 5, 10])
def test_topr_threshold_matches_paper(rule, r):
    assert topr_threshold(0.05, 1000, r, rule=rule) == _paper_threshold(rule, r)


def test_topr_thresholds_match_paper_table():
    assert [round(1e3 * topr_threshold(.05, 1000, r), 2) for r in (2, 5, 10)] == [0.24, 1.31, 3.62]
    np.testing.assert_allclose(
        [1e3 * topr_threshold(.05, 1000, r, rule="markov") for r in (2, 5, 10)], [.1, .25, .5]
    )


@pytest.mark.parametrize("k", [1, 10, 1000])
def test_markov_top1_is_bonferroni(k):
    assert topr_threshold(0.05, k, 1, rule="markov") == 0.05 / k


@pytest.mark.parametrize("seed", [0, 7, 20261028])
@pytest.mark.parametrize("rule", ["beta", "markov"])
@pytest.mark.parametrize("r", [2, 5, 10])
def test_topr_matches_paper_decision(seed, rule, r):
    rng = np.random.default_rng(seed)
    pvalues = rng.uniform(size=1000) * rng.choice([1e-3, 1e-2, 1.0])
    before = pvalues.copy()
    reject, rth, threshold = topr(pvalues, alpha=0.05, r=r, rule=rule)
    ordered = np.sort(pvalues)
    assert rth == ordered[r - 1]
    assert threshold == _paper_threshold(rule, r)
    assert reject is bool(ordered[r - 1] <= threshold)
    np.testing.assert_array_equal(pvalues, before)


@pytest.mark.parametrize("rule", ["beta", "markov"])
@pytest.mark.parametrize("position", ["below", "equal", "above"])
def test_topr_rejection_boundary(rule, position):
    threshold = topr_threshold(0.05, 20, 3, rule=rule)
    value = {"below": np.nextafter(threshold, 0.0), "equal": threshold,
             "above": np.nextafter(threshold, 1.0)}[position]
    pvalues = np.ones(20)
    pvalues[[0, 5, 9]] = [0.0, value / 2, value]
    reject, rth, _ = topr(pvalues, alpha=0.05, r=3, rule=rule)
    assert rth == value
    assert reject is (position != "above")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(alpha=0.0, k=10, r=2), "alpha"),
        (dict(alpha=np.nan, k=10, r=2), "alpha"),
        (dict(alpha=0.05, k=0, r=1), "k must be"),
        (dict(alpha=0.05, k=10.0, r=2), "k must be"),
        (dict(alpha=0.05, k=10, r=0), "r must be"),
        (dict(alpha=0.05, k=10, r=11), "r must be"),
        (dict(alpha=0.05, k=10, r=True), "r must be"),
        (dict(alpha=0.05, k=10, r=2, lam=0.5), "lam"),
        (dict(alpha=0.05, k=10, r=2, lam=np.inf), "lam"),
        (dict(alpha=0.05, k=10, r=2, rule="holm"), "rule must be"),
    ],
)
def test_topr_threshold_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        topr_threshold(**kwargs)


@pytest.mark.parametrize("pvalues", [[], [[0.1, 0.2]], [0.1, np.nan], [0.1, -0.1], [0.1, 1.1]])
def test_topr_rejects_invalid_pvalues(pvalues):
    with pytest.raises(ValueError, match="pvalues"):
        topr(pvalues, alpha=0.05, r=1)
