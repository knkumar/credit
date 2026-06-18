from __future__ import annotations

from typing import Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from calmmm.data.containers import MMMData
from calmmm.model.coords import build_coords, build_arrays
from calmmm.model.priors import PriorConfig
from calmmm.model.transforms import geometric_adstock_pt, hill_saturation_pt
from calmmm.model.components import _build_baseline, _build_media_hierarchy, _add_likelihood
from calmmm.transforms.seasonality import fourier_features


class HierarchicalMMM:
    """
    Hierarchical Bayesian MMM with geo×KPI pooling.

    Parameters
    ----------
    priors : PriorConfig or None — use PriorConfig() defaults if None
    n_fourier_pairs : int — number of sin/cos pairs for seasonal baseline
    holdout_fraction : float — fraction of time steps (last) excluded from likelihood
    """

    def __init__(
        self,
        *,
        priors: Optional[PriorConfig] = None,
        n_fourier_pairs: int = 2,
        holdout_fraction: float = 0.2,
    ) -> None:
        self.priors = priors or PriorConfig()
        self.n_fourier_pairs = n_fourier_pairs
        self.holdout_fraction = holdout_fraction
        # Set by build_model()
        self._model: Optional[pm.Model] = None
        self._data: Optional[MMMData] = None
        self._train_mask: Optional[np.ndarray] = None
        self._obs_array: Optional[np.ndarray] = None
        self._media_scaled: Optional[np.ndarray] = None
        self._fourier_matrix: Optional[np.ndarray] = None
        self._pop_array: Optional[np.ndarray] = None

    def build_model(self, data: MMMData) -> pm.Model:
        """
        Construct the PyMC model for the given dataset.

        The model uses a log-link for all KPIs:
            log(E[y]) = baseline[t,g,k] + media_contrib[t,g,k]

        Media pipeline:
            raw_spend → scale (÷ panel max) → geometric adstock → Hill saturation
            → geo×KPI hierarchy → additive log contribution

        Baseline:
            intercept[K, G] (informed by log mean outcome) + Fourier seasonality[K, F]

        Returns
        -------
        pm.Model
        """
        self._data = data
        coords = build_coords(data, n_fourier_pairs=self.n_fourier_pairs)
        obs_array, media_array, pop_array = build_arrays(data)

        T = len(data.times)

        # Scale media per-channel by panel max
        media_max = media_array.max(axis=(0, 1), keepdims=True)  # [1, 1, C]
        media_scaled = media_array / np.maximum(media_max, 1e-8)

        # Fourier features: t = 0-based week index
        fourier_matrix = fourier_features(
            t=np.arange(T, dtype=float),
            n_pairs=self.n_fourier_pairs,
            period=52.0,
        ).astype(np.float64)

        # Holdout mask
        n_holdout = int(T * self.holdout_fraction)
        train_mask = np.ones(T, dtype=bool)
        if n_holdout > 0:
            train_mask[-n_holdout:] = False
        self._train_mask = train_mask

        # Baseline intercept initialization: log(mean_outcome) per KPI×geo
        obs_mean = np.nanmean(obs_array, axis=0)  # [G, K]
        obs_mean_log = np.log(np.maximum(obs_mean.T, 1.0))  # [K, G]

        # Store for use in fit()
        self._obs_array = obs_array
        self._media_scaled = media_scaled
        self._fourier_matrix = fourier_matrix
        self._pop_array = pop_array

        # Train slices
        X_media_train = media_scaled[train_mask]       # [T_train, G, C]
        fourier_train = fourier_matrix[train_mask]     # [T_train, F]
        obs_train = obs_array[train_mask]              # [T_train, G, K]
        pop_train = pop_array[train_mask]              # [T_train, G, K]

        with pm.Model(coords=coords) as model:
            # Adstock params
            decay = pm.Beta(
                "adstock_decay",
                alpha=self.priors.adstock_decay_alpha,
                beta=self.priors.adstock_decay_beta,
                dims="channel",
            )
            # Adstock transform
            X_adstocked = geometric_adstock_pt(
                pt.as_tensor_variable(X_media_train), decay
            )  # [T_train, G, C]

            # Saturation params
            hill_alpha = pm.HalfNormal(
                "hill_alpha", sigma=self.priors.hill_alpha_sigma, dims="channel"
            )
            hill_k = pm.HalfNormal(
                "hill_k", sigma=self.priors.hill_k_sigma, dims="channel"
            )
            # Saturation transform
            X_sat = hill_saturation_pt(X_adstocked, hill_alpha, hill_k)  # [T_train, G, C]

            # Baseline
            baseline = _build_baseline(fourier_train, obs_mean_log, self.priors)

            # Media hierarchy
            media_contrib = _build_media_hierarchy(X_sat, self.priors)

            # Linear predictor (log scale)
            mu = pm.Deterministic("mu", baseline + media_contrib)

            # Observation likelihoods (train only)
            _add_likelihood(
                mu, obs_train, pop_train,
                data.kpi_metadata, data.kpis, self.priors
            )

        self._model = model
        return model

    def fit(
        self,
        data: MMMData,
        *,
        mode: str = "sample",
        **kwargs,
    ) -> "MMMFit":
        """
        Build (if needed) and run inference on the model.

        Parameters
        ----------
        data : MMMData
        mode : "sample" | "vi" | "map"
        **kwargs : passed to pm.sample / pm.fit / pm.find_MAP

        Returns
        -------
        MMMFit
        """
        from calmmm.model.fit import MMMFit

        if self._model is None or self._data is not data:
            self.build_model(data)

        model = self._model

        if mode == "sample":
            kwargs.setdefault("progressbar", False)
            with model:
                trace = pm.sample(**kwargs)
            return MMMFit(trace=trace, map_params=None, model=model, data=data)

        elif mode == "vi":
            kwargs.setdefault("progressbar", False)
            # Extract n before passing to pm.fit; don't forward it to approx.sample
            n = kwargs.pop("n", 10000)
            with model:
                approx = pm.fit(n=n, **kwargs)
                trace = approx.sample(draws=200)
            return MMMFit(trace=trace, map_params=None, model=model, data=data)

        elif mode == "map":
            with model:
                map_params = pm.find_MAP(**kwargs)
            return MMMFit(trace=None, map_params=map_params, model=model, data=data)

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Expected: 'sample', 'vi', 'map'."
            )
