import numpy as np
import pytest
from calmmm.transforms.adstock import geometric_adstock, weibull_adstock


def test_geometric_adstock_zero_decay_is_identity():
    x = np.array([1.0, 2.0, 3.0, 0.0])
    result = geometric_adstock(x, decay=0.0)
    np.testing.assert_allclose(result, x)


def test_geometric_adstock_accumulates():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    result = geometric_adstock(x, decay=0.5)
    expected = np.array([1.0, 0.5, 0.25, 0.125])
    np.testing.assert_allclose(result, expected)


def test_geometric_adstock_invalid_decay_raises():
    with pytest.raises(ValueError, match="decay must be in"):
        geometric_adstock(np.ones(5), decay=1.5)


def test_weibull_adstock_output_shape():
    x = np.random.default_rng(0).uniform(0, 1, 52)
    result = weibull_adstock(x, shape=1.5, scale=3.0, n_lags=13)
    assert result.shape == x.shape


def test_weibull_adstock_monotone_decay_when_shape_one():
    """shape=1 is exponential — impulse response should peak at t=0 and decay."""
    x = np.zeros(10)
    x[0] = 1.0
    result = weibull_adstock(x, shape=1.0, scale=2.0, n_lags=10)
    assert result[0] >= result[1] >= result[2]


def test_weibull_adstock_delayed_peak_when_shape_gt_one():
    """shape > 1 — peak effect should be delayed past the first time step."""
    x = np.zeros(20)
    x[0] = 1.0
    result = weibull_adstock(x, shape=3.0, scale=5.0, n_lags=13)
    peak_idx = np.argmax(result)
    assert peak_idx > 0, "Delayed peak expected when shape > 1"


def test_weibull_adstock_zero_spend_stays_zero():
    x = np.zeros(10)
    result = weibull_adstock(x, shape=1.5, scale=3.0, n_lags=5)
    np.testing.assert_allclose(result, np.zeros(10))


def test_weibull_adstock_invalid_shape_raises():
    with pytest.raises(ValueError, match="shape must be > 0"):
        weibull_adstock(np.ones(5), shape=-1.0, scale=2.0)


def test_weibull_adstock_invalid_scale_raises():
    with pytest.raises(ValueError, match="scale must be > 0"):
        weibull_adstock(np.ones(5), shape=1.0, scale=0.0)
