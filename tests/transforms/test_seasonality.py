import numpy as np
import pytest
from calmmm.transforms.seasonality import fourier_features


def test_fourier_features_shape():
    t = np.arange(52)
    F = fourier_features(t, n_pairs=2, period=52)
    assert F.shape == (52, 4)  # 2 pairs × (sin + cos)


def test_fourier_features_period():
    """Values at t and t+period must be identical."""
    t = np.arange(104)
    F = fourier_features(t, n_pairs=1, period=52)
    np.testing.assert_allclose(F[0], F[52], atol=1e-10)


def test_fourier_features_bounded():
    t = np.arange(200)
    F = fourier_features(t, n_pairs=3, period=52)
    assert F.max() <= 1.0 + 1e-10
    assert F.min() >= -1.0 - 1e-10


def test_fourier_features_single_pair():
    """n_pairs=1 should return 2 columns: [sin, cos]."""
    t = np.array([0.0, 13.0, 26.0, 39.0])
    F = fourier_features(t, n_pairs=1, period=52)
    assert F.shape == (4, 2)
    # At t=0: sin=0, cos=1
    np.testing.assert_allclose(F[0, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(F[0, 1], 1.0, atol=1e-10)
