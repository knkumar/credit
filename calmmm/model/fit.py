from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
import pymc as pm

from calmmm.data.containers import MMMData

if TYPE_CHECKING:
    from calmmm.model.mmm import HierarchicalMMM
    from calmmm.calibration.targets import IncrementalityTests


def eval_mu_and_channel_contrib(fit: "MMMFit"):
    """Return (mu, channel_contrib) as numpy arrays. 
    Shape [S, T, G, K] and [S, T, G, K, C] for traces, or [T, G, K] and [T, G, K, C] for MAP."""
    if fit.map_params is not None:
        return (
            np.array(fit.map_params["mu"]),
            np.array(fit.map_params["channel_contrib"]),
        )
    if fit.trace is not None:
        mu = fit.trace.posterior["mu"].values
        cc = fit.trace.posterior["channel_contrib"].values
        return (
            mu.reshape(-1, *mu.shape[2:]),
            cc.reshape(-1, *cc.shape[2:]),
        )
    raise ValueError("MMMFit has neither map_params nor trace.")


def _regression_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    kpis: list[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k, kpi in enumerate(kpis):
        y_true = observed[:, :, k].ravel()
        y_pred = predicted[:, :, k].ravel()
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        
        if len(y_true) == 0:
            metrics[f"rmse_{kpi}"] = np.nan
            metrics[f"r2_{kpi}"] = np.nan
            continue
            
        residual = y_true - y_pred
        sse = float(np.sum(residual**2))
        centered = y_true - float(np.mean(y_true))
        sst = float(np.sum(centered**2))
        metrics[f"rmse_{kpi}"] = float(np.sqrt(np.mean(residual**2)))
        metrics[f"r2_{kpi}"] = float(1.0 - sse / sst) if sst > 0 else np.nan
    return metrics


@dataclass
class MMMFit:
    """
    Result of HierarchicalMMM.fit().

    Attributes
    ----------
    trace : arviz InferenceData (MCMC/VI) or None (MAP)
    map_params : dict of param_name → value (MAP) or None
    model : the underlying PyMC model
    data : the MMMData used to build the model
    _mmm : the HierarchicalMMM instance that produced this fit
    """
    trace: Optional[Any]
    map_params: Optional[dict]
    model: pm.Model
    data: MMMData
    _mmm: Optional["HierarchicalMMM"] = field(default=None, repr=False)
    calibration_targets: list = field(default_factory=list)

    def to_netcdf(self, path) -> None:
        """
        Serialize the fit to a netCDF file.

        If this is an MCMC/VI fit (trace is not None), saves the arviz
        InferenceData.  If this is a MAP fit, saves map_params as a plain
        xarray Dataset.  The PyMC model, MMMData, and HierarchicalMMM
        instance are intentionally NOT saved — they must be reconstructed
        by the caller via ``from_netcdf``.

        Calibration targets are not saved; pass the original experiments to
        ``from_netcdf`` (or ``HierarchicalMMM.fit()``) separately if needed.

        Parameters
        ----------
        path : str or Path
            Destination file path.

        Raises
        ------
        ValueError
            If both ``trace`` and ``map_params`` are None (nothing to save).
        """
        from pathlib import Path as _Path
        import numpy as _np
        import arviz as _az
        import xarray as _xr

        path = str(_Path(path))

        if self.trace is not None:
            _az.to_netcdf(self.trace, path)
        elif self.map_params is not None:
            # Use per-variable dimension names to avoid xarray alignment errors
            # when variables have different sizes.
            data_vars = {}
            for k, v in self.map_params.items():
                arr = _np.asarray(v)
                dims = [f"{k}_dim_{i}" for i in range(arr.ndim)]
                data_vars[k] = _xr.DataArray(arr, dims=dims)
            ds = _xr.Dataset(data_vars)
            ds.to_netcdf(path)
        else:
            raise ValueError(
                "MMMFit has nothing to save: both trace and map_params are None."
            )

    @classmethod
    def from_netcdf(cls, path, data, mmm, experiments: Optional["IncrementalityTests"] = None) -> "MMMFit":
        """
        Reconstruct an ``MMMFit`` from a netCDF file written by ``to_netcdf``.

        Calls ``mmm.build_model(data)`` to re-instantiate the PyMC model
        before returning.

        Parameters
        ----------
        path : str or Path
            File produced by ``to_netcdf``.
        data : MMMData or None
            The original training data.
        mmm : HierarchicalMMM
            A fresh ``HierarchicalMMM`` instance with the same configuration
            used to produce the original fit.

        Returns
        -------
        MMMFit
        """
        from pathlib import Path as _Path
        import arviz as _az
        import xarray as _xr

        path = str(_Path(path))

        # Reconstruct the PyMC model (required even for MAP fits so downstream
        # methods that need self.model work correctly).
        mmm.build_model(data, experiments=experiments)
        model = getattr(mmm, "_model", None)

        # Try loading as arviz InferenceData first.
        try:
            trace = _az.from_netcdf(path)
            if hasattr(trace, "posterior"):
                return cls(
                    trace=trace,
                    map_params=None,
                    model=model,
                    data=data,
                    _mmm=mmm,
                    calibration_targets=list(getattr(mmm, "_calibration_targets", [])),
                )
        except (ValueError, KeyError):
            pass

        # Fall back to MAP params stored as a plain xarray Dataset.
        with _xr.open_dataset(path) as ds:
            map_params = {k: ds[k].values for k in ds.data_vars}
        return cls(
            trace=None,
            map_params=map_params,
            model=model,
            data=data,
            _mmm=mmm,
            calibration_targets=list(getattr(mmm, "_calibration_targets", [])),
        )

    def holdout_metrics(self) -> dict[str, float]:
        """
        Compute RMSE on the holdout time window (last holdout_fraction of T).

        For MAP fits, rebuilds a full-T model and re-runs find_MAP to evaluate mu
        over all time steps, then slices the holdout window.
        For MCMC/VI fits, runs sample_posterior_predictive on a full-T model.

        Returns
        -------
        dict with keys rmse_{kpi} for each KPI.

        Raises
        ------
        ValueError if no holdout time steps exist (holdout_fraction=0.0).
        """
        mmm = self._mmm
        if mmm is None or mmm._train_mask is None:
            raise ValueError(
                "holdout_metrics() requires a model built via HierarchicalMMM.fit()"
            )

        holdout_mask = ~mmm._train_mask
        if not holdout_mask.any():
            raise ValueError(
                "No holdout time steps — set holdout_fraction > 0 when creating HierarchicalMMM."
            )

        obs_holdout = mmm._obs_array[holdout_mask]  # [T_holdout, G, K]

        # Update data in the existing model to evaluate mu over all time steps
        full_model = self.model
        with full_model:
            pm.set_data({
                "X_media": mmm._media_scaled,
                "fourier_features": mmm._fourier_matrix,
                "obs_array": mmm._obs_array,
                "pop_array": mmm._pop_array,
            })
            if getattr(mmm, "_ctrl_array", None) is not None:
                pm.set_data({"ctrl_array": mmm._ctrl_array})

        try:
            if self.map_params is not None:
                # Evaluate mu on the full-T model using the *trained* parameter values.
                latent_init = full_model.initial_point()
                latent_params = {k: self.map_params[k] for k in latent_init if k in self.map_params}
                with full_model:
                    fn = full_model.compile_fn(full_model["mu"])
                    mu_val = fn(latent_params)
                mu_holdout = np.array(mu_val)[holdout_mask]
    
            elif self.trace is not None:
                with full_model:
                    ppc = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=["mu"],
                        progressbar=False,
                        predictions=True,
                    )
                if hasattr(ppc, "predictions"):
                    mu_samples = ppc.predictions["mu"].values
                else:
                    mu_samples = ppc.posterior_predictive["mu"].values
                mu_samples = mu_samples.reshape(-1, *mu_samples.shape[2:])
                mu_holdout = mu_samples[:, holdout_mask, ...]  # [S, T_holdout, G, K]
    
            else:
                raise ValueError("No params or trace available for prediction.")
        finally:
            with full_model:
                pm.set_data({
                    "X_media": mmm._media_scaled[mmm._train_mask],
                    "fourier_features": mmm._fourier_matrix[mmm._train_mask],
                    "obs_array": mmm._obs_array[mmm._train_mask],
                    "pop_array": mmm._pop_array[mmm._train_mask],
                })
                if getattr(mmm, "_ctrl_array", None) is not None:
                    pm.set_data({"ctrl_array": mmm._ctrl_array[mmm._train_mask]})

        # Apply inverse link to get predicted mean
        pred_mean = np.zeros_like(mu_holdout)  # [S?, T_holdout, G, K]
        for k, kpi in enumerate(mmm._data.kpis):
            likelihood = mmm._data.kpi_metadata.loc[mmm._data.kpi_metadata["kpi"] == kpi, "likelihood"].values[0]
            if likelihood == "binomial":
                from scipy.special import expit
                pop_k = mmm._pop_array[holdout_mask][:, :, k]
                pred_mean[..., :, k] = expit(mu_holdout[..., :, k]) * pop_k
            else:
                pred_mean[..., :, k] = np.exp(mu_holdout[..., :, k])

        if pred_mean.ndim > 3:
            pred_mean = pred_mean.mean(axis=0)

        return _regression_metrics(obs_holdout, pred_mean, mmm._data.kpis)

    def fit_metrics(self) -> dict[str, float]:
        """
        Compute training-window RMSE and R2 from fitted mean predictions.

        This evaluates the model on the same time window used by the likelihood.
        It is intended as a lightweight goodness-of-fit summary for reporting.
        Use ``holdout_metrics()`` separately when a holdout-specific score is
        required.
        """
        mmm = self._mmm
        if mmm is None or mmm._train_mask is None:
            raise ValueError("fit_metrics() requires a model built via HierarchicalMMM.fit()")

        mu, _channel_contrib = eval_mu_and_channel_contrib(self)
        observed = mmm._obs_array[mmm._train_mask]
        
        predicted = np.zeros_like(mu)
        for k, kpi in enumerate(self.data.kpis):
            likelihood = self.data.kpi_metadata.loc[self.data.kpi_metadata["kpi"] == kpi, "likelihood"].values[0]
            if likelihood == "binomial":
                from scipy.special import expit
                pop_k = mmm._pop_array[mmm._train_mask][:, :, k]
                predicted[..., :, :, k] = expit(mu[..., :, :, k]) * pop_k
            else:
                predicted[..., :, :, k] = np.exp(mu[..., :, :, k])
                
        if predicted.ndim > 3:
            predicted = predicted.mean(axis=0)
            
        return _regression_metrics(observed, predicted, self.data.kpis)

    def mcmc_diagnostics(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """
        Return compact MCMC diagnostics for posterior parameters.

        MAP fits do not have posterior samples, so this returns an empty table
        with stable columns. For MCMC/VI fits, the table includes ArviZ R-hat
        and effective sample size diagnostics where available.
        """
        columns = ["parameter", "r_hat", "ess_bulk", "ess_tail"]
        if self.trace is None:
            return pd.DataFrame(columns=columns)

        import arviz as az

        default_vars = ["adstock_decay", "hill_alpha", "hill_k"]
        requested = var_names or [
            name for name in default_vars if name in self.trace.posterior
        ]
        if not requested:
            return pd.DataFrame(columns=columns)

        summary = az.summary(self.trace, var_names=requested, kind="diagnostics")
        diagnostics = summary.reset_index(names="parameter")
        for column in columns:
            if column not in diagnostics.columns:
                diagnostics[column] = np.nan
        return diagnostics[columns]

    def posterior_predictive(self) -> dict[str, np.ndarray]:
        """
        Generate posterior predictive samples for all training time steps.

        Returns
        -------
        dict mapping obs_{kpi} → ndarray of shape [samples, T_train, G]

        Raises
        ------
        ValueError if no trace is available (MAP fit).
        """
        if self.trace is None:
            raise ValueError(
                "posterior_predictive() requires a trace (MCMC or VI). "
                "MAP fits do not have posterior samples."
            )

        mmm = self._mmm
        with self.model:
            ppc = pm.sample_posterior_predictive(self.trace, progressbar=False)

        result = {}
        for kpi in mmm._data.kpis:
            key = f"obs_{kpi}"
            arr = ppc.posterior_predictive[key].values  # [chains, draws, T_train, G]
            S = arr.shape[0] * arr.shape[1]
            result[key] = arr.reshape(S, *arr.shape[2:])  # [S, T_train, G]

        return result
