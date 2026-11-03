import numpy as np
import pytest

from rp_ci.projections import batch_unit_random_projection, unit_random_projection


def _experimental_batch_projection(X, rng, n_proj, target_dim=1):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("Input must be 1D or 2D array.")
    if n_proj < 1:
        raise ValueError("n_proj must be >= 1.")
    if target_dim < 1:
        raise ValueError("target_dim must be >= 1.")

    n, d = X.shape
    R = rng.normal(size=(d, n_proj * target_dim))
    R /= (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
    Xp = X @ R
    return Xp.reshape(n, n_proj, target_dim).astype(float)


INPUTS = [
    pytest.param([1, 2, 3, 4, 5], id="integer-list"),
    pytest.param(np.linspace(-2.0, 2.0, 35).reshape(7, 5), id="float64"),
    pytest.param(np.arange(24, dtype=np.float32).reshape(8, 3), id="float32"),
    pytest.param(np.arange(60.0).reshape(10, 6)[::2, ::2], id="noncontiguous"),
]


@pytest.mark.parametrize("X", INPUTS)
@pytest.mark.parametrize("seed", [0, 42, 20260724])
@pytest.mark.parametrize("n_proj,target_dim", [(1, 1), (7, 1), (3, 2)])
def test_batch_matches_experimental_implementation(X, seed, n_proj, target_dim):
    before = np.array(X, copy=True)
    actual_rng = np.random.default_rng(seed)
    reference_rng = np.random.default_rng(seed)

    actual = batch_unit_random_projection(X, actual_rng, n_proj, target_dim)
    expected = _experimental_batch_projection(X, reference_rng, n_proj, target_dim)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (len(X), n_proj, target_dim)
    assert actual.dtype == np.dtype(float)
    assert actual_rng.bit_generator.state == reference_rng.bit_generator.state
    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("X", INPUTS)
@pytest.mark.parametrize("seed", [0, 42, 20260724])
@pytest.mark.parametrize("out_dim", [1, 3])
def test_single_matches_experimental_computation(X, seed, out_dim):
    before = np.array(X, copy=True)
    actual_rng = np.random.default_rng(seed)
    reference_rng = np.random.default_rng(seed)

    actual = unit_random_projection(X, actual_rng, out_dim)
    expected = _experimental_batch_projection(X, reference_rng, 1, out_dim)[:, 0, :]

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (len(X), out_dim)
    assert actual_rng.bit_generator.state == reference_rng.bit_generator.state
    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("seed", [0, 20260815])
@pytest.mark.parametrize("k", [1, 10, 1000])
def test_x_then_y_draw_order_matches_experiments(seed, k):
    X = np.linspace(-1.0, 1.0, 51).reshape(17, 3)
    Y = np.linspace(2.0, -2.0, 85).reshape(17, 5)
    actual_rng = np.random.default_rng(seed)
    reference_rng = np.random.default_rng(seed)

    actual_x = batch_unit_random_projection(X, actual_rng, n_proj=k)
    actual_y = batch_unit_random_projection(Y, actual_rng, n_proj=k)
    expected_x = _experimental_batch_projection(X, reference_rng, n_proj=k)
    expected_y = _experimental_batch_projection(Y, reference_rng, n_proj=k)

    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_array_equal(actual_y, expected_y)
    assert actual_rng.bit_generator.state == reference_rng.bit_generator.state


def test_normalization_retains_experimental_offset():
    seed, d, k = 12, 4, 9
    raw = np.random.default_rng(seed).normal(size=(d, k))
    lengths = np.linalg.norm(raw, axis=0)
    directions = batch_unit_random_projection(
        np.eye(d), np.random.default_rng(seed), n_proj=k
    )[:, :, 0]

    np.testing.assert_allclose(
        np.linalg.norm(directions, axis=0),
        lengths / (lengths + 1e-12),
        rtol=1e-15,
        atol=1e-15,
    )


def test_same_seed_reproduces_output():
    X = np.arange(30.0).reshape(10, 3)
    first = batch_unit_random_projection(X, np.random.default_rng(17), 8)
    second = batch_unit_random_projection(X, np.random.default_rng(17), 8)
    different = batch_unit_random_projection(X, np.random.default_rng(18), 8)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize("X", [np.array(1.0), np.zeros((2, 3, 4))])
@pytest.mark.parametrize("function", [unit_random_projection, batch_unit_random_projection])
def test_invalid_array_dimensions_keep_original_error(X, function):
    kwargs = {"n_proj": 2} if function is batch_unit_random_projection else {}
    with pytest.raises(ValueError, match="Input must be 1D or 2D array"):
        function(X, np.random.default_rng(0), **kwargs)


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_projection_count_keeps_original_error(value):
    with pytest.raises(ValueError, match="n_proj must be >= 1"):
        batch_unit_random_projection(np.ones((4, 2)), np.random.default_rng(0), value)


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_target_dimension_keeps_original_error(value):
    with pytest.raises(ValueError, match="target_dim must be >= 1"):
        batch_unit_random_projection(
            np.ones((4, 2)), np.random.default_rng(0), 2, target_dim=value
        )


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_output_dimension_keeps_original_error(value):
    with pytest.raises(ValueError, match="out_dim must be >= 1"):
        unit_random_projection(np.ones((4, 2)), np.random.default_rng(0), out_dim=value)
