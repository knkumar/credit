import numpy as np
import pytest
from calmmm.attribution.contributions import channel_contributions, marginal_contributions


def test_attribution_importable():
    from calmmm import channel_contributions, compute_roi, saturation_curve
    assert callable(channel_contributions)
    assert callable(compute_roi)
    assert callable(saturation_curve)


@pytest.mark.slow
def test_channel_contributions_columns(attr_map_fit):
    df = channel_contributions(attr_map_fit)
    assert set(df.columns) >= {"time", "geo", "kpi", "channel", "contribution"}


@pytest.mark.slow
def test_channel_contributions_channels_present(attr_map_fit):
    df = channel_contributions(attr_map_fit)
    fit = attr_map_fit
    expected = set(fit.data.channels) | {"baseline"}
    assert set(df["channel"].unique()) == expected


@pytest.mark.slow
def test_channel_contributions_nrows(attr_map_fit):
    df = channel_contributions(attr_map_fit)
    fit = attr_map_fit
    T_train = int(fit._mmm._train_mask.sum())
    G = len(fit.data.geos)
    K = len(fit.data.kpis)
    C = len(fit.data.channels)
    expected_rows = T_train * G * K * (C + 1)  # C channels + 1 baseline
    assert len(df) == expected_rows


@pytest.mark.slow
def test_channel_contributions_additive(attr_map_fit):
    """
    Proportional decomposition: baseline + all channel contributions = exp(mu)
    for every (time, geo, kpi) cell.
    """
    df = channel_contributions(attr_map_fit)
    total = df.groupby(["time", "geo", "kpi"])["contribution"].sum().reset_index()
    fit = attr_map_fit
    mu_val = np.array(fit.map_params["mu"])  # [T_train, G, K]
    exp_mu_flat = np.exp(mu_val).ravel()
    assert np.allclose(total["contribution"].values, exp_mu_flat, rtol=1e-4)


@pytest.mark.slow
def test_baseline_equals_no_media_counterfactual(attr_map_fit):
    """
    Baseline contribution must equal exp(mu - sum(cc)) — the outcome without any media.
    """
    df = channel_contributions(attr_map_fit)
    baseline_df = df[df["channel"] == "baseline"].copy()

    fit = attr_map_fit
    mu_val = np.array(fit.map_params["mu"])        # [T, G, K]
    cc_val = np.array(fit.map_params["channel_contrib"])  # [T, G, K, C]
    cc_sum = cc_val.sum(axis=-1)
    expected_baseline = np.exp(mu_val - cc_sum).ravel()

    assert np.allclose(baseline_df["contribution"].values, expected_baseline, rtol=1e-4)


@pytest.mark.slow
def test_marginal_contributions_columns(attr_map_fit):
    df = marginal_contributions(attr_map_fit)
    assert set(df.columns) >= {"time", "geo", "kpi", "channel", "contribution"}


@pytest.mark.slow
def test_marginal_contributions_no_baseline_row(attr_map_fit):
    df = marginal_contributions(attr_map_fit)
    assert "baseline" not in df["channel"].values


@pytest.mark.slow
def test_marginal_contributions_not_additive(attr_map_fit):
    """
    Marginal contributions intentionally do NOT sum to exp(mu) — this is correct.
    Verify the sum differs from exp(mu) by more than floating-point noise when
    there are multiple channels (the non-additivity is O(channel interactions)).
    """
    fit = attr_map_fit
    if len(fit.data.channels) < 2:
        pytest.skip("non-additivity only observable with ≥2 channels")

    df = marginal_contributions(fit)
    total = df.groupby(["time", "geo", "kpi"])["contribution"].sum().reset_index()
    mu_val = np.array(fit.map_params["mu"])
    exp_mu_flat = np.exp(mu_val).ravel()
    # They will generally differ; assert they are not all exactly equal
    assert not np.allclose(total["contribution"].values, exp_mu_flat, rtol=1e-6)
