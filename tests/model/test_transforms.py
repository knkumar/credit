import numpy as np
import pytensor.tensor as pt
import pytest

from calmmm.model.transforms import geometric_adstock_pt, hill_saturation_pt


# --- geometric_adstock_pt ---

def test_adstock_zero_decay_equals_input():
    T, G, C = 5, 2, 3
    X = np.ones((T, G, C), dtype="float64")
    decay = np.zeros(C, dtype="float64")
    result = geometric_adstock_pt(
        pt.as_tensor_variable(X), pt.as_tensor_variable(decay)
    ).eval()
    np.testing.assert_allclose(result, X)


def test_adstock_shape():
    T, G, C = 10, 2, 3
    X = np.random.rand(T, G, C)
    decay = np.array([0.5, 0.3, 0.7])
    result = geometric_adstock_pt(
        pt.as_tensor_variable(X), pt.as_tensor_variable(decay)
    ).eval()
    assert result.shape == (T, G, C)


def test_adstock_accumulates_over_time():
    # With decay=0.5, constant input=1: values should grow then plateau
    T, G, C = 20, 1, 1
    X = np.ones((T, G, C), dtype="float64")
    decay = np.array([0.5])
    result = geometric_adstock_pt(
        pt.as_tensor_variable(X), pt.as_tensor_variable(decay)
    ).eval()
    # t=0: h=1, t=1: h=1.5, t=2: h=1.75, ... plateau near 2.0
    assert result[0, 0, 0] < result[5, 0, 0]
    assert result[5, 0, 0] < result[15, 0, 0] + 0.01  # nearly converged


def test_adstock_full_decay_doubles_immediately():
    # decay=1 means all prior signal carries over: h[t] = x[t] + h[t-1]
    T, G, C = 5, 1, 1
    X = np.ones((T, G, C), dtype="float64")
    decay = np.array([1.0])
    result = geometric_adstock_pt(
        pt.as_tensor_variable(X), pt.as_tensor_variable(decay)
    ).eval()
    # t=0:1, t=1:2, t=2:3 ...
    np.testing.assert_allclose(result[:, 0, 0], np.arange(1, T + 1, dtype=float))


# --- hill_saturation_pt ---

def test_hill_at_half_saturation_point():
    # At X=k, saturation should be 0.5 (with alpha=1)
    C = 2
    k_vals = np.array([1.0, 2.0])
    X = np.array([[[1.0, 2.0]]])  # [1, 1, 2]
    result = hill_saturation_pt(
        pt.as_tensor_variable(X),
        pt.as_tensor_variable(np.ones(C)),
        pt.as_tensor_variable(k_vals),
    ).eval()
    np.testing.assert_allclose(result[0, 0, :], [0.5, 0.5], atol=1e-3)


def test_hill_range():
    T, G, C = 5, 2, 3
    X = pt.as_tensor_variable(np.abs(np.random.rand(T, G, C)))
    alpha = pt.as_tensor_variable(np.array([0.5, 1.0, 2.0]))
    k = pt.as_tensor_variable(np.array([0.5, 1.0, 0.3]))
    result = hill_saturation_pt(X, alpha, k).eval()
    assert np.all(result >= 0) and np.all(result <= 1)


def test_hill_shape():
    T, G, C = 8, 3, 4
    X = pt.as_tensor_variable(np.random.rand(T, G, C))
    result = hill_saturation_pt(
        X,
        pt.as_tensor_variable(np.ones(C)),
        pt.as_tensor_variable(np.ones(C)),
    ).eval()
    assert result.shape == (T, G, C)


def test_hill_zero_input_is_zero():
    T, G, C = 3, 2, 2
    X = pt.as_tensor_variable(np.zeros((T, G, C)))
    result = hill_saturation_pt(
        X,
        pt.as_tensor_variable(np.array([1.0, 2.0])),
        pt.as_tensor_variable(np.array([0.5, 1.0])),
    ).eval()
    np.testing.assert_allclose(result, 0.0, atol=1e-6)
