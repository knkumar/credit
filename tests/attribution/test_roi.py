import numpy as np
import pytest
import pymc as pm
from calmmm.attribution.roi import compute_roi
from calmmm.model.fit import MMMFit


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


def test_compute_roi_raises_without_mmm():
    fit = MMMFit(trace=None, map_params={}, model=pm.Model(), data=None, _mmm=None)
    with pytest.raises(ValueError, match="_mmm is None"):
        compute_roi(fit)


def test_compute_roi_zero_spend_returns_nan():
    """When a channel has zero spend, ROI should be NaN, not inf."""
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from unittest.mock import patch

    # Fake marginal_contributions returning a channel with contribution but zero spend
    fake_contribs = pd.DataFrame({
        "time": [pd.Timestamp("2024-01-01")] * 2,
        "geo": ["A", "A"],
        "kpi": ["visits", "visits"],
        "channel": ["search", "zero_ch"],
        "contribution": [100.0, 50.0],
    })
    fake_media = pd.DataFrame({
        "time": [pd.Timestamp("2024-01-01")] * 2,
        "geo": ["A", "A"],
        "channel": ["search", "zero_ch"],
        "spend": [1000.0, 0.0],
    })
    mock_mmm = SimpleNamespace(
        _train_mask=np.array([True]),
    )
    mock_data = SimpleNamespace(
        times=[pd.Timestamp("2024-01-01")],
        media=fake_media,
    )
    fit = SimpleNamespace(
        _mmm=mock_mmm,
        data=mock_data,
    )

    with patch("calmmm.attribution.roi.marginal_contributions", return_value=fake_contribs):
        from calmmm.attribution.roi import compute_roi
        result = compute_roi(fit)

    zero_row = result[result["channel"] == "zero_ch"]
    assert zero_row["total_spend"].values[0] == 0.0
    assert np.isnan(zero_row["roi"].values[0]), "ROI should be NaN when spend is zero"
    # Non-zero channel should have a finite ROI
    search_row = result[result["channel"] == "search"]
    assert np.isfinite(search_row["roi"].values[0])
