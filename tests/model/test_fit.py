import numpy as np
import pandas as pd
import pytest
import arviz as az

from calmmm.model.mmm import HierarchicalMMM
from calmmm.model.fit import MMMFit, _regression_metrics


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


def test_regression_metrics_include_r2_per_kpi():
    observed = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ]
    )
    predicted = np.array(
        [
            [[1.0, 12.0], [2.0, 18.0]],
            [[3.0, 32.0], [5.0, 38.0]],
        ]
    )

    metrics = _regression_metrics(observed, predicted, ["applications", "revenue"])

    assert set(metrics) == {
        "rmse_applications",
        "r2_applications",
        "rmse_revenue",
        "r2_revenue",
    }
    assert metrics["rmse_applications"] == 0.5
    assert round(metrics["r2_applications"], 6) == 0.8
    assert 0.9 < metrics["r2_revenue"] < 1.0


def test_mcmc_diagnostics_returns_empty_table_for_map_fit():
    fit = MMMFit(trace=None, map_params={}, model=None, data=None, _mmm=None)

    diagnostics = fit.mcmc_diagnostics()

    assert list(diagnostics.columns) == ["parameter", "r_hat", "ess_bulk", "ess_tail"]
    assert diagnostics.empty


def test_mcmc_diagnostics_returns_arviz_summary_columns():
    trace = az.from_dict(
        posterior={
            "adstock_decay": np.array([[[0.2, 0.3], [0.25, 0.35]], [[0.22, 0.32], [0.24, 0.34]]]),
            "hill_alpha": np.array([[[1.0, 1.2], [1.1, 1.3]], [[0.9, 1.1], [1.0, 1.2]]]),
        }
    )
    fit = MMMFit(trace=trace, map_params=None, model=None, data=None, _mmm=None)

    diagnostics = fit.mcmc_diagnostics(var_names=["adstock_decay"])

    assert isinstance(diagnostics, pd.DataFrame)
    assert {"parameter", "r_hat", "ess_bulk", "ess_tail"}.issubset(diagnostics.columns)
    assert diagnostics["parameter"].str.contains("adstock_decay").any()


def test_fit_metrics_uses_training_predictions():
    class _FakeData:
        kpis = ["applications"]

    class _FakeMMM:
        _obs_array = np.array([[[10.0]], [[20.0]], [[30.0]]])
        _train_mask = np.array([True, True, False])

    fit = MMMFit(
        trace=None,
        map_params={"mu": np.log(np.array([[[12.0]], [[18.0]]])), "channel_contrib": np.zeros((2, 1, 1, 1))},
        model=None,
        data=_FakeData(),
        _mmm=_FakeMMM(),
    )

    metrics = fit.fit_metrics()

    assert metrics["rmse_applications"] == pytest.approx(2.0)
    assert metrics["r2_applications"] == pytest.approx(0.84)


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
