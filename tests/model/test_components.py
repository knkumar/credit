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


# ---- Media hierarchy ----

def test_media_hierarchy_shape():
    T, G, K, C = 10, 2, 4, 2
    priors = PriorConfig()
    X_sat_val = np.random.default_rng(0).random((T, G, C)).astype("float64")

    with pm.Model(coords=_base_coords()) as model:
        X_sat = pt.as_tensor_variable(X_sat_val)
        contrib = _build_media_hierarchy(X_sat, priors)
        val = pm.draw(contrib, random_seed=0)
    assert val.shape == (T, G, K)


def test_media_hierarchy_logp_finite():
    T, G, K, C = 5, 2, 4, 2
    priors = PriorConfig()
    X_sat_val = np.random.default_rng(1).random((T, G, C)).astype("float64")

    with pm.Model(coords=_base_coords()) as model:
        X_sat = pt.as_tensor_variable(X_sat_val)
        _build_media_hierarchy(X_sat, priors)
        lp_fn = model.compile_fn(model.logp())
        val = lp_fn(model.initial_point())
    assert np.isfinite(val)


def test_media_hierarchy_variable_names():
    T, G, K, C = 5, 2, 4, 2
    priors = PriorConfig()
    X_sat_val = np.random.default_rng(2).random((T, G, C)).astype("float64")

    with pm.Model(coords=_base_coords()) as model:
        _build_media_hierarchy(pt.as_tensor_variable(X_sat_val), priors)
        names = {v.name for v in model.free_RVs}
    assert "scale_global" in names
    assert "scale_kpi_raw" in names
    assert "scale_geo_raw" in names


# ---- Likelihoods ----

def _kpi_meta(name, likelihood):
    return pd.DataFrame([{"kpi": name, "likelihood": likelihood, "funnel_stage": None, "family": None}])


def test_likelihood_gaussian():
    T, G, K = 10, 2, 1
    priors = PriorConfig()
    obs = np.abs(np.random.default_rng(0).random((T, G, K))) * 1000 + 1
    pop = np.full((T, G, K), np.nan)
    kpi_meta = _kpi_meta("revenue", "gaussian")

    with pm.Model() as model:
        mu = pt.as_tensor_variable(np.log(obs))
        _add_likelihood(mu, obs, pop, kpi_meta, ["revenue"], priors)
        val = model.compile_fn(model.logp())(model.initial_point())
    assert np.isfinite(val)


def test_likelihood_lognormal():
    T, G, K = 10, 2, 1
    priors = PriorConfig()
    obs = np.abs(np.random.default_rng(1).random((T, G, K))) * 100 + 1
    pop = np.full((T, G, K), np.nan)
    kpi_meta = _kpi_meta("revenue_ln", "lognormal")

    with pm.Model() as model:
        mu = pt.as_tensor_variable(np.log(obs))
        _add_likelihood(mu, obs, pop, kpi_meta, ["revenue_ln"], priors)
        val = model.compile_fn(model.logp())(model.initial_point())
    assert np.isfinite(val)


def test_likelihood_negative_binomial():
    T, G, K = 10, 2, 1
    priors = PriorConfig()
    obs = np.round(np.abs(np.random.default_rng(2).random((T, G, K))) * 100) + 1
    pop = np.full((T, G, K), np.nan)
    kpi_meta = _kpi_meta("visits", "negative_binomial")

    with pm.Model() as model:
        mu = pt.as_tensor_variable(np.log(obs))
        _add_likelihood(mu, obs, pop, kpi_meta, ["visits"], priors)
        val = model.compile_fn(model.logp())(model.initial_point())
    assert np.isfinite(val)


def test_likelihood_binomial():
    T, G, K = 5, 2, 1
    priors = PriorConfig()
    pop = np.full((T, G, K), 1000.0)
    obs = np.round(pop * 0.05)
    kpi_meta = _kpi_meta("rate_kpi", "binomial")

    with pm.Model() as model:
        mu = pt.as_tensor_variable(np.zeros((T, G, K)))
        _add_likelihood(mu, obs, pop, kpi_meta, ["rate_kpi"], priors)
        val = model.compile_fn(model.logp())(model.initial_point())
    assert np.isfinite(val)


def test_likelihood_binomial_requires_population():
    T, G, K = 5, 2, 1
    priors = PriorConfig()
    obs = np.ones((T, G, K))
    pop = np.full((T, G, K), np.nan)
    kpi_meta = _kpi_meta("rate", "binomial")

    with pm.Model():
        mu = pt.as_tensor_variable(np.zeros((T, G, K)))
        with pytest.raises(ValueError, match="population is NaN"):
            _add_likelihood(mu, obs, pop, kpi_meta, ["rate"], priors)


def test_likelihood_unknown_raises():
    T, G, K = 5, 2, 1
    priors = PriorConfig()
    obs = np.ones((T, G, K))
    pop = np.full((T, G, K), np.nan)
    kpi_meta = _kpi_meta("foo", "poisson")

    with pm.Model():
        mu = pt.as_tensor_variable(np.zeros((T, G, K)))
        with pytest.raises(ValueError, match="Unknown likelihood"):
            _add_likelihood(mu, obs, pop, kpi_meta, ["foo"], priors)


def test_likelihood_multi_kpi():
    T, G, K = 10, 2, 2
    priors = PriorConfig()
    obs = np.abs(np.random.default_rng(3).random((T, G, K))) * 100 + 1
    pop = np.full((T, G, K), np.nan)
    kpi_meta = pd.DataFrame([
        {"kpi": "visits", "likelihood": "negative_binomial", "funnel_stage": None, "family": None},
        {"kpi": "revenue", "likelihood": "gaussian", "funnel_stage": None, "family": None},
    ])

    with pm.Model() as model:
        mu = pt.as_tensor_variable(np.log(obs + 1))
        _add_likelihood(mu, obs, pop, kpi_meta, ["visits", "revenue"], priors)
        val = model.compile_fn(model.logp())(model.initial_point())
    assert np.isfinite(val)
