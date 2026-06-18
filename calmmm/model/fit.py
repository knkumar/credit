from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pymc as pm

from calmmm.data.containers import MMMData


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
    _mmm: Optional[Any] = field(default=None, repr=False)
    calibration_targets: list = field(default_factory=list)

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
            with full_model:
                map_full = pm.find_MAP(progressbar=False)
                fn = full_model.compile_fn(full_model["mu"])
                mu_val = fn(map_full)
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
