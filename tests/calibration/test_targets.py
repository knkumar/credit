import numpy as np
import pandas as pd
import pytest

from calmmm.calibration.targets import CalibrationTarget, build_calibration_targets


# ---- CalibrationTarget ----

def test_calibration_target_fields():
    target = CalibrationTarget(
        test_id="exp_1",
        t_indices=np.array([2, 3, 4]),
        g_indices=np.array([0]),
        c_indices=np.array([0]),
        k_index=1,
        lift_obs=1000.0,
        se=200.0,
        calibration_likelihood="normal",
        estimand="total",
    )
    assert target.test_id == "exp_1"
    assert list(target.t_indices) == [2, 3, 4]
    assert target.k_index == 1
    assert target.lift_obs == 1000.0
    assert target.se == 200.0


# ---- build_calibration_targets ----

def test_build_calibration_targets_returns_list(lift_tests, mmmdata):
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    assert isinstance(targets, list)
    assert len(targets) == len(lift_tests)


def test_build_calibration_targets_test_id(lift_tests, mmmdata):
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    assert targets[0].test_id == "search_holdout_q1"


def test_build_calibration_targets_kpi_index(lift_tests, mmmdata):
    # experiment KPI is "visits"; sorted kpis are ["applications", "approvals", "revenue", "visits"]
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    kpis = mmmdata.kpis  # sorted
    expected_k = kpis.index("visits")
    assert targets[0].k_index == expected_k


def test_build_calibration_targets_channel_index(lift_tests, mmmdata):
    # experiment channel is "search"; sorted channels are ["search", "social"]
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    channels = mmmdata.channels  # sorted
    expected_c = channels.index("search")
    assert list(targets[0].c_indices) == [expected_c]


def test_build_calibration_targets_geo_index(lift_tests, mmmdata):
    # experiment geo_scope is "DMA_1"; sorted geos are ["DMA_1", "DMA_2"]
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    geos = mmmdata.geos  # sorted
    expected_g = geos.index("DMA_1")
    assert list(targets[0].g_indices) == [expected_g]


def test_build_calibration_targets_t_indices_in_window(lift_tests, mmmdata):
    # experiment window: 2024-03-04 to 2024-03-25; full training
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    t_idx = targets[0].t_indices
    # All t_indices must correspond to times within the experiment window
    times = mmmdata.times
    exp_start = pd.Timestamp("2024-03-04")
    exp_end = pd.Timestamp("2024-03-25")
    for i in t_idx:
        assert exp_start <= times[i] <= exp_end


def test_build_calibration_targets_lift_se(lift_tests, mmmdata):
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    assert targets[0].lift_obs == 12_000.0
    assert targets[0].se == 2_500.0


def test_build_calibration_targets_no_training_times_raises(lift_tests, mmmdata):
    """If the experiment window falls entirely in holdout, raise ValueError."""
    T = len(mmmdata.times)
    # holdout = last 20 weeks; experiment window 2024-03-04..2024-03-25 is weeks 9-12
    # To put window in holdout: set train_mask to only first 5 weeks
    train_mask = np.zeros(T, dtype=bool)
    train_mask[:5] = True  # weeks 0-4 only; experiment starts week 9
    with pytest.raises(ValueError, match="no training time steps"):
        build_calibration_targets(lift_tests, mmmdata, train_mask)
