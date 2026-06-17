from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from calmmm.model.priors import PriorConfig


def _build_baseline(
    fourier_matrix: np.ndarray,
    obs_mean_log: np.ndarray,
    priors: PriorConfig,
) -> pt.TensorVariable:
    """
    Baseline = per-(KPI, geo) intercept + Fourier seasonality.

    Parameters
    ----------
    fourier_matrix : [T, F] numpy array — Fourier features (deterministic)
    obs_mean_log : [K, G] numpy array — log(mean_outcome) per KPI×geo,
                   used as intercept prior mean (log scale)
    priors : PriorConfig

    Returns
    -------
    pytensor tensor [T, G, K] — baseline on log scale

    Must be called inside a pm.Model context with coords
    {"kpi": [...], "geo": [...], "fourier": [...]}.
    """
    intercept = pm.Normal(
        "intercept",
        mu=obs_mean_log,
        sigma=priors.baseline_sigma,
        dims=("kpi", "geo"),
    )
    fourier_beta = pm.Normal(
        "fourier_beta",
        mu=0.0,
        sigma=priors.seasonality_sigma,
        dims=("kpi", "fourier"),
    )
    # intercept [K, G] → [1, G, K]
    intercept_tgk = intercept.T[None, :, :]
    # fourier_matrix [T, F] @ fourier_beta.T [F, K] → [T, K] → [T, 1, K]
    fourier_contrib = pt.dot(fourier_matrix, fourier_beta.T)[:, None, :]
    return intercept_tgk + fourier_contrib  # [T, G, K]


def _build_media_hierarchy(X_sat, priors: PriorConfig):
    """Stub — implemented in Task 5."""
    raise NotImplementedError("_build_media_hierarchy implemented in Task 5")


def _add_likelihood(mu, obs_array, pop_array, kpi_metadata, kpis, priors):
    """Stub — implemented in Task 6."""
    raise NotImplementedError("_add_likelihood implemented in Task 6")
