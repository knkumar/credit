import numpy as np
import pymc as pm
import pytest

from calmmm.model.mmm import HierarchicalMMM
from calmmm.model.priors import PriorConfig


def test_build_model_returns_pymc_model(mmmdata):
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    assert isinstance(model, pm.Model)


def test_build_model_has_required_variables(mmmdata):
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    names = {v.name for v in model.free_RVs}
    assert "adstock_decay" in names
    assert "hill_alpha" in names
    assert "hill_k" in names
    assert "intercept" in names
    assert "fourier_beta" in names
    assert "scale_global" in names
    assert "scale_kpi_raw" in names
    assert "scale_geo_raw" in names


def test_build_model_logp_finite(mmmdata):
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    with model:
        ip = model.initial_point()
        lp = model.compile_logp()(ip)
    assert np.isfinite(lp)


def test_build_model_custom_priors(mmmdata):
    priors = PriorConfig(adstock_decay_alpha=5.0, hill_k_sigma=2.0)
    mmm = HierarchicalMMM(priors=priors)
    model = mmm.build_model(mmmdata)
    assert isinstance(model, pm.Model)


def test_build_model_no_holdout(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.0)
    model = mmm.build_model(mmmdata)
    assert mmm._train_mask.all()


def test_build_model_holdout_mask_correct(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    mmm.build_model(mmmdata)
    T = 52
    n_holdout = int(T * 0.2)
    assert mmm._train_mask.sum() == T - n_holdout
    assert not mmm._train_mask[-1]
    assert mmm._train_mask[0]


def test_build_model_deterministic_mu(mmmdata):
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    det_names = {v.name for v in model.deterministics}
    assert "mu" in det_names


def test_build_model_obs_nodes_per_kpi(mmmdata):
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    obs_names = {v.name for v in model.observed_RVs}
    for kpi in mmmdata.kpis:
        assert f"obs_{kpi}" in obs_names
