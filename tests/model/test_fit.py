import numpy as np
import pytest

from calmmm.model.mmm import HierarchicalMMM
from calmmm.model.fit import MMMFit


# ---------------------------------------------------------------------------
# Fast serialization tests — no PyMC inference required
# ---------------------------------------------------------------------------

def test_to_netcdf_map_roundtrip(tmp_path):
    """MAP params survive a to_netcdf / from_netcdf round-trip."""
    map_params = {"mu": np.array([1.0, 2.0, 3.0]), "sigma": np.array([0.5])}
    fit = MMMFit(trace=None, map_params=map_params, model=None, data=None, _mmm=None)

    path = tmp_path / "map_fit.nc"
    fit.to_netcdf(path)
    assert path.exists()

    class _FakeMMM:
        _model = None
        _calibration_targets = []

        def build_model(self, data, experiments=None):
            pass

    loaded = MMMFit.from_netcdf(path, data=None, mmm=_FakeMMM())
    assert loaded.trace is None
    assert loaded.map_params is not None
    assert np.allclose(loaded.map_params["mu"], map_params["mu"])
    assert np.allclose(loaded.map_params["sigma"], map_params["sigma"])


def test_to_netcdf_raises_when_nothing_to_save():
    """to_netcdf raises ValueError when both trace and map_params are None."""
    fit = MMMFit(trace=None, map_params=None, model=None, data=None, _mmm=None)
    with pytest.raises(ValueError, match="nothing to save"):
        fit.to_netcdf("/tmp/should_not_exist.nc")


@pytest.fixture(scope="session")
def map_fit(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    return mmm.fit(mmmdata, mode="map")


@pytest.fixture(scope="session")
def sample_fit(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    return mmm.fit(mmmdata, mode="sample", draws=50, tune=50, chains=1, progressbar=False, random_seed=42)


@pytest.mark.slow
def test_holdout_metrics_returns_dict(map_fit):
    metrics = map_fit.holdout_metrics()
    assert isinstance(metrics, dict)


@pytest.mark.slow
def test_holdout_metrics_has_rmse_per_kpi(map_fit, mmmdata):
    metrics = map_fit.holdout_metrics()
    for kpi in mmmdata.kpis:
        assert f"rmse_{kpi}" in metrics


@pytest.mark.slow
def test_holdout_metrics_finite(map_fit, mmmdata):
    metrics = map_fit.holdout_metrics()
    for kpi in mmmdata.kpis:
        assert np.isfinite(metrics[f"rmse_{kpi}"])


@pytest.mark.slow
def test_holdout_metrics_no_holdout_raises(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.0)
    fit = mmm.fit(mmmdata, mode="map")
    with pytest.raises(ValueError, match="holdout"):
        fit.holdout_metrics()


@pytest.mark.slow
def test_posterior_predictive_returns_dict(sample_fit, mmmdata):
    ppc = sample_fit.posterior_predictive()
    assert isinstance(ppc, dict)
    for kpi in mmmdata.kpis:
        assert f"obs_{kpi}" in ppc


@pytest.mark.slow
def test_posterior_predictive_shape(sample_fit, mmmdata):
    ppc = sample_fit.posterior_predictive()
    T = len(mmmdata.times)
    T_train = T - int(T * 0.2)  # holdout_fraction=0.2
    G = len(mmmdata.geos)
    for kpi in mmmdata.kpis:
        arr = ppc[f"obs_{kpi}"]
        assert arr.ndim >= 2
        assert arr.shape[-2] == T_train
        assert arr.shape[-1] == G


@pytest.mark.slow
def test_posterior_predictive_not_available_for_map(map_fit):
    with pytest.raises(ValueError, match="trace"):
        map_fit.posterior_predictive()
