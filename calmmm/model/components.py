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


def _build_media_hierarchy(
    X_sat: pt.TensorVariable,
    priors: PriorConfig,
) -> pt.TensorVariable:
    """
    Three-level non-centered geo×KPI hierarchy for media contributions.

    Parameters
    ----------
    X_sat : [T, G, C] pytensor tensor — saturation-transformed spend
    priors : PriorConfig

    Returns
    -------
    pytensor tensor [T, G, K] — media contribution on log scale

    Must be called inside pm.Model context with coords
    {"channel": [...], "kpi": [...], "geo": [...]}.

    Hierarchy:
        scale_global[C] ~ HalfNormal
        scale_kpi[C, K] = scale_global + sigma_kpi * Normal(0,1)  (non-centered)
        scale_geo[C, K, G] = scale_kpi + sigma_geo * Normal(0,1)  (non-centered)
        contrib[t,g,k] = sum_c( X_sat[t,g,c] * scale_geo[c,k,g] )
    """
    # Global channel scale
    scale_global = pm.HalfNormal(
        "scale_global",
        sigma=priors.channel_scale_global_sigma,
        dims="channel",
    )
    # KPI level — non-centered
    scale_kpi_raw = pm.Normal("scale_kpi_raw", 0.0, 1.0, dims=("channel", "kpi"))
    scale_kpi_sigma = pm.HalfNormal(
        "scale_kpi_sigma", sigma=priors.channel_scale_kpi_sigma, dims="channel"
    )
    scale_kpi = pm.Deterministic(
        "scale_kpi",
        scale_global[:, None] + scale_kpi_sigma[:, None] * scale_kpi_raw,
        dims=("channel", "kpi"),
    )
    # Geo level — non-centered
    scale_geo_raw = pm.Normal(
        "scale_geo_raw", 0.0, 1.0, dims=("channel", "kpi", "geo")
    )
    scale_geo_sigma = pm.HalfNormal(
        "scale_geo_sigma", sigma=priors.channel_scale_geo_sigma, dims="channel"
    )
    scale_geo = pm.Deterministic(
        "scale_geo",
        scale_kpi[:, :, None] + scale_geo_sigma[:, None, None] * scale_geo_raw,
        dims=("channel", "kpi", "geo"),
    )
    # Contribution: einsum-style sum over channels
    # X_sat [T,G,C] → [T,G,1,C]; scale_geo [C,K,G] → [G,K,C] → [1,G,K,C]
    scale_geo_gkc = scale_geo.dimshuffle(2, 1, 0)  # [G, K, C]
    channel_contrib_tgkc = X_sat[:, :, None, :] * scale_geo_gkc[None, :, :, :]  # [T, G, K, C]
    pm.Deterministic("channel_contrib", channel_contrib_tgkc)
    return channel_contrib_tgkc.sum(axis=-1)  # [T, G, K]


def _add_likelihood(
    mu: pt.TensorVariable,
    obs_array: np.ndarray,
    pop_array: np.ndarray,
    kpi_metadata,
    kpis: list[str],
    priors: PriorConfig,
) -> None:
    """
    Add per-KPI observed likelihood nodes to the current pm.Model context.

    Parameters
    ----------
    mu : pytensor tensor [T, G, K] — log-scale linear predictor
    obs_array : [T, G, K] float64 — observed outcomes
    pop_array : [T, G, K] float64 — population (NaN where not used)
    kpi_metadata : pd.DataFrame with columns "kpi", "likelihood"
    kpis : list[str] — KPI names in axis-K order
    priors : PriorConfig

    Link function: log-link (exp(mu_k) is the mean for all non-binomial likelihoods).
    Binomial uses sigmoid(mu_k) as probability.
    """
    for k, kpi in enumerate(kpis):
        row = kpi_metadata.loc[kpi_metadata["kpi"] == kpi]
        likelihood = row["likelihood"].values[0]
        y_obs = obs_array[:, :, k]
        mu_k = mu[:, :, k]

        if likelihood == "gaussian":
            sigma_k = pm.HalfNormal(f"sigma_{kpi}", sigma=priors.sigma_sigma)
            pm.Normal(f"obs_{kpi}", mu=pm.math.exp(mu_k), sigma=sigma_k, observed=y_obs)

        elif likelihood == "lognormal":
            sigma_k = pm.HalfNormal(f"sigma_{kpi}", sigma=priors.sigma_sigma)
            pm.LogNormal(f"obs_{kpi}", mu=mu_k, sigma=sigma_k, observed=y_obs)

        elif likelihood == "negative_binomial":
            alpha_k = pm.HalfNormal(f"nb_alpha_{kpi}", sigma=priors.nb_alpha_sigma)
            pm.NegativeBinomial(
                f"obs_{kpi}",
                mu=pm.math.exp(mu_k),
                alpha=alpha_k,
                observed=y_obs,
            )

        elif likelihood == "binomial":
            n_pop = pop_array[:, :, k]
            if np.any(np.isnan(n_pop)):
                raise ValueError(
                    f"KPI '{kpi}' has likelihood='binomial' but population is NaN. "
                    "Provide a population column in MMMData."
                )
            pm.Binomial(
                f"obs_{kpi}",
                n=n_pop.astype(int),
                p=pm.math.sigmoid(mu_k),
                observed=y_obs,
            )

        else:
            raise ValueError(
                f"Unknown likelihood '{likelihood}' for KPI '{kpi}'. "
                "Expected: gaussian, lognormal, negative_binomial, binomial."
            )
