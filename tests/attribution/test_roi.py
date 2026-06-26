import numpy as np
import pytest
from calmmm.attribution.roi import compute_roi


@pytest.mark.slow
def test_compute_roi_columns(attr_map_fit):
    df = compute_roi(attr_map_fit)
    assert set(df.columns) >= {"kpi", "channel", "total_contribution", "total_spend", "roi"}


@pytest.mark.slow
def test_compute_roi_no_baseline(attr_map_fit):
    df = compute_roi(attr_map_fit)
    assert "baseline" not in df["channel"].values


@pytest.mark.slow
def test_compute_roi_nrows(attr_map_fit):
    fit = attr_map_fit
    df = compute_roi(fit)
    expected = len(fit.data.kpis) * len(fit.data.channels)
    assert len(df) == expected


@pytest.mark.slow
def test_compute_roi_total_spend_matches_data(attr_map_fit):
    fit = attr_map_fit
    df = compute_roi(fit)
    train_times = set(
        t for t, m in zip(fit.data.times, fit._mmm._train_mask) if m
    )
    media_train = fit.data.media[fit.data.media["time"].isin(train_times)]
    for ch in fit.data.channels:
        expected_spend = media_train[media_train["channel"] == ch]["spend"].sum()
        row = df[df["channel"] == ch]
        assert np.isclose(row["total_spend"].values[0], expected_spend, rtol=1e-6)


@pytest.mark.slow
def test_compute_roi_finite(attr_map_fit):
    df = compute_roi(attr_map_fit)
    assert df["total_contribution"].notna().all()
    assert (df["total_spend"] > 0).all()
    assert df["roi"].notna().all()
