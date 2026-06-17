import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest

from calmmm.model.priors import PriorConfig
from calmmm.model.components import _build_baseline, _build_media_hierarchy, _add_likelihood


KPIS = ["applications", "approvals", "revenue", "visits"]  # sorted
GEOS = ["DMA_1", "DMA_2"]
CHANNELS = ["search", "social"]


def _base_coords(T: int = 20, n_fourier: int = 4):
    return {
        "kpi": KPIS,
        "geo": GEOS,
        "channel": CHANNELS,
        "fourier": list(range(n_fourier)),
    }


# ---- Baseline ----

def test_baseline_shape():
    T, G, K, F = 20, 2, 4, 4
    priors = PriorConfig()
    obs_mean_log = np.zeros((K, G))
    fourier_matrix = np.random.rand(T, F)

    with pm.Model(coords=_base_coords(T, F)) as model:
        baseline = _build_baseline(fourier_matrix, obs_mean_log, priors)
        val = pm.draw(baseline, random_seed=0)
    assert val.shape == (T, G, K)


def test_baseline_logp_finite():
    T, G, K, F = 10, 2, 4, 4
    priors = PriorConfig()
    obs_mean_log = np.ones((K, G)) * 5.0
    fourier_matrix = np.random.rand(T, F)

    with pm.Model(coords=_base_coords(T, F)) as model:
        _build_baseline(fourier_matrix, obs_mean_log, priors)
        lp_fn = model.compile_fn(model.logp())
        val = lp_fn(model.initial_point())
    assert np.isfinite(val)


def test_baseline_intercept_shape():
    T, G, K, F = 10, 2, 4, 4
    priors = PriorConfig()
    obs_mean_log = np.zeros((K, G))
    fourier_matrix = np.random.rand(T, F)

    with pm.Model(coords=_base_coords(T, F)) as model:
        _build_baseline(fourier_matrix, obs_mean_log, priors)
        intercept_shape = tuple(model["intercept"].shape.eval())
        fourier_beta_shape = tuple(model["fourier_beta"].shape.eval())
    assert intercept_shape == (K, G)
    assert fourier_beta_shape == (K, F)
