from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PriorConfig:
    # Geometric adstock: decay ~ Beta(alpha, beta)
    adstock_decay_alpha: float = 3.0
    adstock_decay_beta: float = 3.0

    # Hill saturation
    hill_alpha_sigma: float = 0.5   # HalfNormal sigma for shape exponent
    hill_k_sigma: float = 1.0       # HalfNormal sigma for half-saturation point

    # Baseline (log scale)
    baseline_sigma: float = 2.0     # intercept prior spread around log(mean)
    seasonality_sigma: float = 0.5  # Fourier coefficient scale

    # Media hierarchy (non-centered)
    channel_scale_global_sigma: float = 1.0
    channel_scale_kpi_sigma: float = 0.5
    channel_scale_geo_sigma: float = 0.25

    # KPI dispersion
    sigma_sigma: float = 0.5        # Gaussian/LogNormal sigma
    nb_alpha_sigma: float = 1.0     # NegBin dispersion
