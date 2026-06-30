import numpy as np
import pytest
from calmmm.transforms.scaling import MediaScaler


def test_scaler_output_in_zero_one():
    spend = np.array([100.0, 500.0, 1000.0, 200.0])
    scaler = MediaScaler()
    scaled = scaler.fit_transform(spend)
    assert scaled.max() <= 1.0 + 1e-10
    assert scaled.min() >= 0.0 - 1e-10
    np.testing.assert_allclose(scaled.max(), 1.0)


def test_scaler_roundtrip():
    spend = np.array([100.0, 500.0, 1000.0, 200.0])
    scaler = MediaScaler()
    scaled = scaler.fit_transform(spend)
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered, spend)


def test_scaler_zero_spend_channel():
    spend = np.zeros(10)
    scaler = MediaScaler()
    scaled = scaler.fit_transform(spend)
    np.testing.assert_allclose(scaled, np.zeros(10))


def test_scaler_not_fitted_raises():
    scaler = MediaScaler()
    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.inverse_transform(np.ones(5))


def test_scaler_max_spend_property():
    spend = np.array([0.0, 500.0, 2000.0, 300.0])
    scaler = MediaScaler()
    scaler.fit_transform(spend)
    assert scaler.max_spend == 2000.0


def test_scaler_max_spend_not_fitted_raises():
    scaler = MediaScaler()
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = scaler.max_spend
