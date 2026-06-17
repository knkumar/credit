import numpy as np
import pytest

from calmmm.model.coords import build_coords, build_arrays


def test_build_coords_keys(mmmdata):
    coords = build_coords(mmmdata, n_fourier_pairs=2)
    assert set(coords.keys()) == {"time", "geo", "kpi", "channel", "fourier"}


def test_build_coords_fourier_length(mmmdata):
    coords = build_coords(mmmdata, n_fourier_pairs=3)
    assert len(coords["fourier"]) == 6  # 2 * n_pairs


def test_build_coords_lists_are_sorted(mmmdata):
    coords = build_coords(mmmdata)
    assert coords["geo"] == sorted(coords["geo"])
    assert coords["kpi"] == sorted(coords["kpi"])
    assert coords["channel"] == sorted(coords["channel"])
    assert coords["time"] == sorted(coords["time"])


def test_build_arrays_shapes(mmmdata):
    obs, media, pop = build_arrays(mmmdata)
    T, G, K, C = 52, 2, 4, 2
    assert obs.shape == (T, G, K)
    assert media.shape == (T, G, C)
    assert pop.shape == (T, G, K)


def test_build_arrays_dtype(mmmdata):
    obs, media, pop = build_arrays(mmmdata)
    assert obs.dtype == np.float64
    assert media.dtype == np.float64
    assert pop.dtype == np.float64


def test_build_arrays_obs_no_nan(mmmdata):
    obs, _, _ = build_arrays(mmmdata)
    assert not np.any(np.isnan(obs))


def test_build_arrays_media_nonneg(mmmdata):
    _, media, _ = build_arrays(mmmdata)
    assert np.all(media >= 0)


def test_build_arrays_population_positive(mmmdata):
    _, _, pop = build_arrays(mmmdata)
    # synthetic_panel provides full population coverage — every cell must be positive
    assert np.all(pop > 0)
