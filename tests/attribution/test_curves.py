import numpy as np
import pytest
from calmmm.attribution.curves import saturation_curve


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
