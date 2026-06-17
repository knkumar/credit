import numpy as np
import pytest
from calmmm.transforms.saturation import hill_saturation, ispline_basis


def test_hill_half_saturation_point():
    """At x=K, hill(x) == 0.5 for any alpha."""
    K = 1000.0
    x = np.array([K])
    for alpha in [0.5, 1.0, 2.0, 3.0]:
        result = hill_saturation(x, alpha=alpha, K=K)
        np.testing.assert_allclose(result, [0.5], atol=1e-10)


def test_hill_monotone():
    x = np.linspace(0, 5000, 100)
    result = hill_saturation(x, alpha=2.0, K=1000.0)
    assert np.all(np.diff(result) >= 0)


def test_hill_zero_input():
    result = hill_saturation(np.array([0.0]), alpha=2.0, K=1000.0)
    np.testing.assert_allclose(result, [0.0])


def test_hill_asymptote():
    x = np.array([1e9])
    result = hill_saturation(x, alpha=2.0, K=1000.0)
    assert result[0] > 0.9999


def test_hill_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be > 0"):
        hill_saturation(np.ones(5), alpha=-1.0, K=100.0)


def test_hill_invalid_K_raises():
    with pytest.raises(ValueError, match="K must be > 0"):
        hill_saturation(np.ones(5), alpha=1.0, K=0.0)


def test_ispline_basis_shape():
    x = np.linspace(0, 1, 50)
    B = ispline_basis(x, n_knots=4)
    assert B.shape[0] == 50
    assert B.shape[1] >= 4


def test_ispline_basis_monotone_columns():
    """Every column of the I-spline basis must be monotone non-decreasing."""
    x = np.linspace(0, 1, 200)
    B = ispline_basis(x, n_knots=4)
    for j in range(B.shape[1]):
        diffs = np.diff(B[:, j])
        assert np.all(diffs >= -1e-10), f"Column {j} is not monotone non-decreasing"


def test_ispline_basis_bounded():
    x = np.linspace(0, 1, 100)
    B = ispline_basis(x, n_knots=4)
    assert B.min() >= -1e-10
    assert B.max() <= 1.0 + 1e-10


def test_ispline_basis_nonneg_weights_give_monotone_response():
    """A random non-negative weight vector applied to the basis must yield a monotone response."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 100)
    B = ispline_basis(x, n_knots=4)
    weights = rng.uniform(0, 1, B.shape[1])
    response = B @ weights
    assert np.all(np.diff(response) >= -1e-10)
