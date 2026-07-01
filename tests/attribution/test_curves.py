import numpy as np
import pytest
import pandas as pd

from calmmm.attribution.curves import saturation_curve, spend_response_report


class _FakeData:
    channels = ["search", "social"]


class _FakeFit:
    data = _FakeData()


def test_spend_response_report_uses_saturation_curves(monkeypatch):
    def fake_saturation_curve(_fit, channel, n_points=100):
        curves = {
            "search": pd.DataFrame(
                {
                    "channel": ["search", "search"],
                    "spend": [0.0, 400.0],
                    "saturation": [0.0, 0.8],
                }
            ),
            "social": pd.DataFrame(
                {
                    "channel": ["social", "social"],
                    "spend": [0.0, 200.0],
                    "saturation": [0.0, 0.5],
                }
            ),
        }
        return curves[channel]

    monkeypatch.setattr(
        "calmmm.attribution.curves.saturation_curve",
        fake_saturation_curve,
    )
    panel = pd.DataFrame(
        {
            "search_spend": [100.0, 300.0],
            "social_spend": [50.0, 150.0],
        }
    )

    report = spend_response_report(
        _FakeFit(),
        panel,
        spend_columns={"search": "search_spend", "social": "social_spend"},
        spend_multiplier=1.10,
    )

    search = report.set_index("channel").loc["search"]
    assert search["current_spend"] == 200.0
    assert round(search["increased_spend"], 6) == 220.0
    assert search["current_response"] == 0.4
    assert round(search["increased_response"], 6) == 0.44
    assert round(search["response_lift"], 6) == 0.04
    assert round(search["response_lift_pct"], 6) == 0.10


@pytest.mark.slow
def test_saturation_curve_columns(attr_map_fit):
    df = saturation_curve(attr_map_fit, "tv")
    assert set(df.columns) >= {"spend", "saturation", "channel"}


@pytest.mark.slow
def test_saturation_curve_n_points(attr_map_fit):
    df = saturation_curve(attr_map_fit, "tv", n_points=20)
    assert len(df) == 20


@pytest.mark.slow
def test_saturation_curve_default_n_points(attr_map_fit):
    df = saturation_curve(attr_map_fit, "tv")
    assert len(df) == 50


@pytest.mark.slow
def test_saturation_curve_spend_range(attr_map_fit):
    df = saturation_curve(attr_map_fit, "tv")
    assert df["spend"].iloc[0] == pytest.approx(0.0)
    expected_max = 2 * attr_map_fit._mmm._media_max[attr_map_fit.data.channels.index("tv")]
    assert df["spend"].iloc[-1] == pytest.approx(expected_max, rel=1e-5)


@pytest.mark.slow
def test_saturation_curve_saturation_in_zero_one(attr_map_fit):
    df = saturation_curve(attr_map_fit, "tv")
    assert (df["saturation"] >= 0).all()
    assert (df["saturation"] <= 1).all()


@pytest.mark.slow
def test_saturation_curve_monotone(attr_map_fit):
    df = saturation_curve(attr_map_fit, "digital")
    diffs = np.diff(df["saturation"].values)
    assert (diffs >= -1e-9).all()


@pytest.mark.slow
def test_saturation_curve_unknown_channel_raises(attr_map_fit):
    with pytest.raises(ValueError, match="unknown channel"):
        saturation_curve(attr_map_fit, "unknown_channel")
