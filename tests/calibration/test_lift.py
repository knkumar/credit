import numpy as np
import pandas as pd
import pytest

from calmmm.calibration.targets import build_calibration_targets
from calmmm.calibration.lift import compute_model_lift
from calmmm.model.mmm import HierarchicalMMM


@pytest.fixture
def map_fit_with_targets(lift_tests, mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    fit = mmm.fit(mmmdata, mode="map")
    targets = build_calibration_targets(lift_tests, mmmdata, mmm._train_mask)
    return fit, targets


@pytest.mark.slow
def test_compute_model_lift_returns_dataframe(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.slow
def test_compute_model_lift_columns(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert set(result.columns) == {"test_id", "lift_model", "lift_obs", "se", "z_score"}


@pytest.mark.slow
def test_compute_model_lift_one_row_per_target(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert len(result) == len(targets)


@pytest.mark.slow
def test_compute_model_lift_test_id(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert result.iloc[0]["test_id"] == "search_holdout_q1"


@pytest.mark.slow
def test_compute_model_lift_lift_obs_correct(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert result.iloc[0]["lift_obs"] == 12_000.0


@pytest.mark.slow
def test_compute_model_lift_model_lift_is_finite(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert np.isfinite(result.iloc[0]["lift_model"])


@pytest.mark.slow
def test_compute_model_lift_model_lift_is_positive(map_fit_with_targets):
    """Model-implied lift should be positive since media contrib > 0."""
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert result.iloc[0]["lift_model"] > 0


@pytest.mark.slow
def test_compute_model_lift_z_score_formula(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    row = result.iloc[0]
    expected_z = (row["lift_model"] - row["lift_obs"]) / row["se"]
    assert abs(row["z_score"] - expected_z) < 1e-9


@pytest.mark.slow
def test_compute_model_lift_empty_targets(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.0)
    fit = mmm.fit(mmmdata, mode="map")
    result = compute_model_lift(fit, [])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
