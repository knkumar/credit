import numpy as np
import pymc as pm
import pytest

from calmmm.calibration.targets import build_calibration_targets
from calmmm.calibration.likelihood import add_calibration_likelihood
from calmmm.model.mmm import HierarchicalMMM


def test_add_calibration_likelihood_adds_observed_node(lift_tests, mmmdata):
    """After calling add_calibration_likelihood, the model has a lift_obs_* observed RV."""
    T = len(mmmdata.times)
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    model = mmm.build_model(mmmdata)
    targets = build_calibration_targets(lift_tests, mmmdata, mmm._train_mask)

    with model:
        add_calibration_likelihood(model, targets)
        obs_names = {v.name for v in model.observed_RVs}

    assert "lift_obs_search_holdout_q1" in obs_names


def test_add_calibration_likelihood_logp_finite(lift_tests, mmmdata):
    """logp must remain finite at initial point after adding calibration terms."""
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    model = mmm.build_model(mmmdata)
    targets = build_calibration_targets(lift_tests, mmmdata, mmm._train_mask)

    with model:
        add_calibration_likelihood(model, targets)
        lp = model.compile_logp()(model.initial_point())

    assert np.isfinite(lp)


def test_add_calibration_likelihood_empty_targets(mmmdata):
    """Calling with an empty target list is a no-op."""
    mmm = HierarchicalMMM(holdout_fraction=0.0)
    model = mmm.build_model(mmmdata)
    n_obs_before = len(model.observed_RVs)

    with model:
        add_calibration_likelihood(model, [])

    assert len(model.observed_RVs) == n_obs_before


def test_add_calibration_likelihood_unsupported_estimand_raises(lift_tests, mmmdata):
    """CalibrationTargets with estimand != 'total' must raise NotImplementedError."""
    from calmmm.calibration.targets import CalibrationTarget
    target = CalibrationTarget(
        test_id="exp_immediate",
        t_indices=np.array([0, 1]),
        g_indices=np.array([0]),
        c_indices=np.array([0]),
        k_index=0,
        lift_obs=1000.0,
        se=200.0,
        calibration_likelihood="normal",
        estimand="immediate",  # not supported in MVP
    )
    mmm = HierarchicalMMM(holdout_fraction=0.0)
    model = mmm.build_model(mmmdata)

    with model:
        with pytest.raises(NotImplementedError, match="immediate"):
            add_calibration_likelihood(model, [target])
