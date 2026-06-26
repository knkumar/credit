from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pymc as pm

from calmmm.data.containers import MMMData

if TYPE_CHECKING:
    from calmmm.model.mmm import HierarchicalMMM


def eval_mu_and_channel_contrib(fit: "MMMFit"):
    """Return (mu [T,G,K], channel_contrib [T,G,K,C]) as numpy arrays."""
    if fit.map_params is not None:
        return (
            np.array(fit.map_params["mu"]),
            np.array(fit.map_params["channel_contrib"]),
        )
    if fit.trace is not None:
        return (
            fit.trace.posterior["mu"].values.mean(axis=(0, 1)),
            fit.trace.posterior["channel_contrib"].values.mean(axis=(0, 1)),
        )
    raise ValueError("MMMFit has neither map_params nor trace.")


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
                arr = _np.atleast_1d(v)
                dims = [f"{k}_dim_{i}" for i in range(arr.ndim)]
                data_vars[k] = _xr.DataArray(arr, dims=dims)
            ds = _xr.Dataset(data_vars)
            ds.to_netcdf(path)
        else:
            raise ValueError(
                "MMMFit has nothing to save: both trace and map_params are None."
            )

    @classmethod
    def from_netcdf(cls, path, data, mmm) -> "MMMFit":
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
        mmm.build_model(data)
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

        # Build a full-T model (no holdout) to evaluate mu over all time steps
        full_mmm = mmm.__class__(
            priors=mmm.priors,
            n_fourier_pairs=mmm.n_fourier_pairs,
            holdout_fraction=0.0,
        )
        full_model = full_mmm.build_model(mmm._data)

        if self.map_params is not None:
            # Evaluate mu on the full-T model using the *trained* parameter values.
            # model.initial_point() gives the key set for all free (latent) RVs in the
            # transformed space — the same format find_MAP() returns.  Filtering
            # map_params to these keys excludes deterministics (mu, channel_contrib,
            # scale_kpi, …) which would cause "too many parameters" in compile_fn.
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
                )
            # ppc.posterior_predictive["mu"]: [chains, draws, T, G, K]
            mu_samples = ppc.posterior_predictive["mu"].values
            mu_holdout = mu_samples.mean(axis=(0, 1))[holdout_mask]  # [T_holdout, G, K]

        else:
            raise ValueError("No params or trace available for prediction.")

        # mu is on log scale → exp to get predicted mean
        pred_mean = np.exp(mu_holdout)  # [T_holdout, G, K]

        kpis = mmm._data.kpis
        return {
            f"rmse_{kpi}": float(np.sqrt(np.mean(
                (obs_holdout[:, :, k].ravel() - pred_mean[:, :, k].ravel()) ** 2
            )))
            for k, kpi in enumerate(kpis)
        }

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
