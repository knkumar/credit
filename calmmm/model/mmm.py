from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from calmmm.data.containers import IncrementalityTests
    from calmmm.model.fit import MMMFit

import numpy as np
import pymc as pm
import pytensor.tensor as pt

logger = logging.getLogger(__name__)

from scipy.special import logit

from calmmm.data.containers import MMMData
from calmmm.data.validation import validate_mmmdata
from calmmm.model.coords import build_coords, build_arrays, build_controls_array
from calmmm.model.priors import PriorConfig
from calmmm.model.transforms import geometric_adstock_pt, hill_saturation_pt
from calmmm.model.components import _build_baseline, _build_media_hierarchy, _add_likelihood
from calmmm.model.interactions import InteractionGraph, build_interaction_step
from calmmm.transforms.seasonality import fourier_features
from calmmm.calibration.targets import build_calibration_targets
from calmmm.calibration.likelihood import add_calibration_likelihood


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
        interaction_graph: Optional[InteractionGraph] = None,
    ) -> None:
        if not (0.0 <= holdout_fraction < 1.0):
            raise ValueError("holdout_fraction must be in [0.0, 1.0)")
        self.priors = priors or PriorConfig()
        self.n_fourier_pairs = n_fourier_pairs
        self.holdout_fraction = holdout_fraction
        self.interaction_graph = interaction_graph
        # Set by build_model()
        self._model: Optional[pm.Model] = None
        self._data: Optional[MMMData] = None
        self._train_mask: Optional[np.ndarray] = None
        self._obs_array: Optional[np.ndarray] = None
        self._media_scaled: Optional[np.ndarray] = None
        self._media_max: Optional[np.ndarray] = None
        self._fourier_matrix: Optional[np.ndarray] = None
        self._pop_array: Optional[np.ndarray] = None
        self._calibration_targets: list = []
        self._last_experiments = None

    @property
    def model(self) -> "Optional[pm.Model]":
        return self._model

    def build_model(self, data: MMMData, experiments=None) -> pm.Model:
        """
        Construct the PyMC model for the given dataset.

        The model uses a log-link for all KPIs:
            log(E[y]) = baseline[t,g,k] + media_contrib[t,g,k]
            (For LogNormal likelihood, mu = baseline + media_contrib models the log-median,
             and the true log-mean includes the variance term: mu + sigma^2 / 2)

        Media pipeline:
            raw_spend → scale (÷ panel max) → geometric adstock → Hill saturation
            → geo×KPI hierarchy → additive log contribution

        Baseline:
            intercept[K, G] (informed by log mean outcome) + Fourier seasonality[K, F]

        Returns
        -------
        pm.Model
        """
        validate_mmmdata(data).raise_if_errors()

        self._data = data
        coords = build_coords(data, n_fourier_pairs=self.n_fourier_pairs)
        obs_array, media_array, pop_array = build_arrays(data)
        ctrl_array, _ctrl_names = build_controls_array(data)

        T = len(data.times)

        # Holdout mask
        n_holdout = int(T * self.holdout_fraction)
        train_mask = np.ones(T, dtype=bool)
        if n_holdout > 0:
            train_mask[-n_holdout:] = False
        self._train_mask = train_mask

        # Scale media per-channel by panel max (from train set)
        media_max = media_array[train_mask].max(axis=(0, 1), keepdims=True)  # [1, 1, C]
        global_max = media_array.max(axis=(0, 1), keepdims=True)
        if (global_max == 0.0).any():
            logger.warning("The dataset has zero media spend for one or more channels, indicating a likely data issue.")
        media_max = np.where(media_max == 0.0, global_max, media_max)
        self._media_max = media_max[0, 0, :]  # [C] — per-channel panel max spend
        media_scaled = media_array / np.maximum(media_max, 1e-8)

        # Determine period from data times
        if len(data.times) < 2:
            raise ValueError("Insufficient data: at least 2 time steps are required.")

        import pandas as pd
        diffs = pd.Series(data.times).diff().dropna()
        median_days = diffs.dt.total_seconds().median() / 86400.0
        period = 365.25 / median_days
        if 11.5 < period < 12.5:
            period = 12.0
        
        standard_periods = [1.0, 4.0, 12.0, 26.0, 26.08, 52.17, 365.25]
        if not any(abs(period - sp) < 0.15 * sp for sp in standard_periods):
            logger.warning(
                "Inferred seasonality period (%.2f) deviates significantly from standard cyclic patterns (e.g. 12, 52, 365).",
                period
            )

        # Fourier features: t = 0-based index
        fourier_matrix = fourier_features(
            t=np.arange(T, dtype=float),
            period=period,
            n_pairs=self.n_fourier_pairs,
        ).astype(np.float64)

        logger.info(
            "build_model: T=%d, G=%d, K=%d, C=%d, holdout=%d",
            T, len(data.geos), len(data.kpis), len(data.channels), n_holdout,
        )

        # Baseline intercept initialization: log(mean_outcome) per KPI×geo (logit for binomial)
        obs_mean_log = np.zeros((len(data.kpis), len(data.geos)))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for k, kpi in enumerate(data.kpis):
                likelihood = data.kpi_metadata.loc[data.kpi_metadata["kpi"] == kpi, "likelihood"].values[0]
                if likelihood == "binomial":
                    p = np.nan_to_num(np.nanmean(obs_array[train_mask, :, k] / np.maximum(pop_array[train_mask, :, k], 1.0), axis=0), nan=0.0)
                    p = np.clip(p, 1e-4, 1.0 - 1e-4)
                    obs_mean_log[k, :] = logit(p)
                elif likelihood == "lognormal":
                    obs_mean_log[k, :] = np.nan_to_num(np.nanmean(np.log(np.maximum(obs_array[train_mask, :, k], 1e-8)), axis=0))
                else:
                    obs_mean = np.nan_to_num(np.nanmean(obs_array[train_mask, :, k], axis=0), nan=0.0)
                    obs_mean_log[k, :] = np.log(np.maximum(obs_mean, 1e-8))

        # Store for use in fit()
        self._obs_array = obs_array
        self._media_scaled = media_scaled
        self._fourier_matrix = fourier_matrix
        self._pop_array = pop_array
        if ctrl_array is not None:
            ctrl_std = np.where(ctrl_array.std(axis=0) == 0, 1.0, ctrl_array.std(axis=0))
            self._ctrl_array = (ctrl_array - ctrl_array.mean(axis=0)) / ctrl_std
        else:
            self._ctrl_array = None

        # Train slices
        X_media_train = media_scaled[train_mask]       # [T_train, G, C]
        fourier_train = fourier_matrix[train_mask]     # [T_train, F]
        obs_train = obs_array[train_mask]              # [T_train, G, K]
        pop_train = pop_array[train_mask]              # [T_train, G, K]
        ctrl_train = ctrl_array[train_mask] if ctrl_array is not None else None  # [T_train, G, N] or None

        coords["time"] = [t for i, t in enumerate(data.times) if train_mask[i]]
        with pm.Model(coords=coords) as model:
            # Wrap inputs in Data to avoid recompilation
            X_media_train_data = pm.Data("X_media", X_media_train, dims=("time", "geo", "channel"))
            fourier_train_data = pm.Data("fourier_features", fourier_train, dims=("time", "fourier"))
            obs_train_data = np.ma.masked_invalid(obs_train)
            pop_train_data = pm.Data("pop_array", np.nan_to_num(pop_train, nan=1.0), dims=("time", "geo", "kpi"))
            
            ctrl_train_data = None
            if ctrl_train is not None:
                ctrl_train_data = pm.Data("ctrl_array", ctrl_train, dims=("time", "geo", "control"))
            # Adstock params
            decay = pm.Beta(
                "adstock_decay",
                alpha=self.priors.adstock_decay_alpha,
                beta=self.priors.adstock_decay_beta,
                dims="channel",
            )
            # Adstock transform
            X_adstocked = geometric_adstock_pt(
                X_media_train_data, decay
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
            baseline = _build_baseline(fourier_train_data, obs_mean_log, self.priors, ctrl_train_data)

            # Media hierarchy, with optional channel-to-channel interactions
            apply_interactions = None
            if self.interaction_graph is not None:
                apply_interactions = build_interaction_step(
                    self.interaction_graph,
                    channels=data.channels,
                    X_adstock=X_adstocked,
                )
            media_contrib = _build_media_hierarchy(X_sat, self.priors, apply_interactions=apply_interactions)

            # Linear predictor (log scale)
            mu = pm.Deterministic("mu", baseline + media_contrib)

            # Observation likelihoods (train only)
            _add_likelihood(
                mu, obs_train_data, pop_train_data,
                data.kpi_metadata, data.kpis, self.priors
            )

        self._model = model

        if experiments is not None:
            targets = build_calibration_targets(experiments, data, self._train_mask)
            with model:
                add_calibration_likelihood(
                    model, targets,
                    kpi_metadata=data.kpi_metadata,
                    kpis=data.kpis,
                    pop_array=pop_train_data,
                )
            self._calibration_targets = targets
        else:
            self._calibration_targets = []
            
        self._last_experiments = experiments

        return model

    def fit(
        self,
        data: MMMData,
        *,
        experiments: Optional["IncrementalityTests"] = None,
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

        has_experiments = experiments is not None and len(experiments) > 0
        has_targets = bool(self._calibration_targets)
        curr_exps = experiments if experiments is not None else []
        last_exps = getattr(self, "_last_experiments", None)
        last_exps = last_exps if last_exps is not None else []

        if (
            self._model is None
            or self._data is not data
            or (has_experiments and not has_targets)
            or (not has_experiments and has_targets)
            or curr_exps != last_exps
        ):
            self.build_model(data, experiments=experiments)

        model = self._model

        if mode == "sample":
            kwargs.setdefault("progressbar", False)
            with model:
                trace = pm.sample(**kwargs)
            return MMMFit(trace=trace, map_params=None, model=model, data=data, _mmm=self, calibration_targets=self._calibration_targets)

        elif mode == "vi":
            kwargs.setdefault("progressbar", False)
            # Extract n before passing to pm.fit; don't forward it to approx.sample
            n = kwargs.pop("n", 10000)
            draws = kwargs.pop("draws", 200)
            with model:
                approx = pm.fit(n=n, **kwargs)
                trace = approx.sample(draws=draws)
            return MMMFit(trace=trace, map_params=None, model=model, data=data, _mmm=self, calibration_targets=self._calibration_targets)

        elif mode == "map":
            with model:
                map_params = pm.find_MAP(**kwargs)
            return MMMFit(trace=None, map_params=map_params, model=model, data=data, _mmm=self, calibration_targets=self._calibration_targets)

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Expected: 'sample', 'vi', 'map'."
            )
