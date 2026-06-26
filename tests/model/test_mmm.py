import numpy as np
import pymc as pm
import pytest

from calmmm.model.mmm import HierarchicalMMM
from calmmm.model.fit import MMMFit
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


import arviz as az


@pytest.mark.slow
def test_fit_sample_returns_mmmfit(mmmdata):
    mmm = HierarchicalMMM()
    fit = mmm.fit(mmmdata, mode="sample", draws=50, tune=50, chains=1, progressbar=False, random_seed=42)
    from calmmm.model.fit import MMMFit
    assert isinstance(fit, MMMFit)
    assert fit.trace is not None
    assert isinstance(fit.trace, az.InferenceData)
    assert fit.model is not None
    assert fit.data is mmmdata


@pytest.mark.slow
def test_fit_vi_returns_mmmfit(mmmdata):
    mmm = HierarchicalMMM()
    fit = mmm.fit(mmmdata, mode="vi", n=100, progressbar=False)
    from calmmm.model.fit import MMMFit
    assert isinstance(fit, MMMFit)
    assert fit.trace is not None


@pytest.mark.slow
def test_fit_map_returns_mmmfit(mmmdata):
    mmm = HierarchicalMMM()
    fit = mmm.fit(mmmdata, mode="map")
    from calmmm.model.fit import MMMFit
    assert isinstance(fit, MMMFit)
    assert fit.map_params is not None
    assert isinstance(fit.map_params, dict)


def test_fit_invalid_mode_raises(mmmdata):
    mmm = HierarchicalMMM()
    with pytest.raises(ValueError, match="mode"):
        mmm.fit(mmmdata, mode="invalid")


@pytest.mark.slow
def test_fit_reuses_built_model(mmmdata):
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    fit = mmm.fit(mmmdata, mode="map")
    assert fit.model is model


@pytest.mark.slow
def test_fit_with_experiments_completes(mmmdata, lift_tests):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    fit = mmm.fit(mmmdata, experiments=lift_tests, mode="map")
    assert isinstance(fit, MMMFit)


@pytest.mark.slow
def test_fit_with_experiments_has_calibration_targets(mmmdata, lift_tests):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    fit = mmm.fit(mmmdata, experiments=lift_tests, mode="map")
    assert len(fit.calibration_targets) == 1
    assert fit.calibration_targets[0].test_id == "search_holdout_q1"


@pytest.mark.slow
def test_fit_with_experiments_has_lift_obs_node(mmmdata, lift_tests):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    fit = mmm.fit(mmmdata, experiments=lift_tests, mode="map")
    obs_names = {v.name for v in fit.model.observed_RVs}
    assert "lift_obs_search_holdout_q1" in obs_names


@pytest.mark.slow
def test_fit_without_experiments_no_lift_nodes(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    fit = mmm.fit(mmmdata, mode="map")
    obs_names = {v.name for v in fit.model.observed_RVs}
    assert not any("lift_obs" in n for n in obs_names)
    assert fit.calibration_targets == []


def test_public_import_calibration_target():
    from calmmm import CalibrationTarget
    assert CalibrationTarget is not None


def test_rebuild_guard_switches_from_calibrated_to_uncalibrated(mmmdata):
    """
    Verify that calling build_model(data, experiments=None) after a calibrated build
    triggers a rebuild and resets calibration targets.

    Regression test for: when experiments=None is passed on a second call after a
    calibrated first call, the model should be rebuilt to remove calibration likelihood nodes.
    """
    mmm = HierarchicalMMM()
    # Initial uncalibrated build
    mmm.build_model(mmmdata, experiments=None)
    assert mmm._calibration_targets == []

    # Simulate a prior calibrated build by manually injecting targets
    mmm._calibration_targets = ["sentinel_target"]
    initial_model_id = id(mmm._model)

    # Second call with experiments=None should trigger rebuild
    # because experiments is None AND _calibration_targets is non-empty
    mmm.build_model(mmmdata, experiments=None)

    # After rebuild, targets should be reset to empty
    assert mmm._calibration_targets == []
    # Model object should be replaced (new build_model call)
    assert id(mmm._model) != initial_model_id
