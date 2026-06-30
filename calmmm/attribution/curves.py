from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from calmmm.model.fit import MMMFit


def saturation_curve(fit: "MMMFit", channel: str, n_points: int = 50) -> pd.DataFrame:
    """
    Evaluate the Hill saturation curve for one channel.

    Parameters
    ----------
    fit : MMMFit
    channel : str — must be in fit.data.channels
    n_points : int — number of spend grid points

    Returns
    -------
    DataFrame with columns: spend, saturation, channel
        spend is in original (unscaled) spend units, grid from 0 to 2×panel_max
        saturation is Hill(spend/panel_max, alpha, k), values in [0, 1]
    """
    channels = fit.data.channels
    if channel not in channels:
        raise ValueError(f"unknown channel '{channel}'. Available: {channels}")

    c_idx = channels.index(channel)
    hill_alpha, hill_k = _eval_hill_params(fit)
    alpha_c = float(hill_alpha[c_idx])
    k_c = float(hill_k[c_idx])

    media_max = fit._mmm._media_max  # [C]
    max_spend = float(media_max[c_idx])

    x = np.linspace(0.0, 2.0 * max_spend, n_points)
    x_scaled = x / max(max_spend, 1e-8)
    x_pow = np.clip(x_scaled, 0.0, None) ** alpha_c
    k_pow = k_c ** alpha_c
    saturation = x_pow / (x_pow + k_pow + 1e-9)

    return pd.DataFrame({"spend": x, "saturation": saturation, "channel": channel})


def _eval_hill_params(fit):
    """Return (hill_alpha [C], hill_k [C]) as numpy arrays."""
    if fit.map_params is not None:
        return (
            np.array(fit.map_params["hill_alpha"]),
            np.array(fit.map_params["hill_k"]),
        )
    if fit.trace is not None:
        return (
            fit.trace.posterior["hill_alpha"].values.mean(axis=(0, 1)),
            fit.trace.posterior["hill_k"].values.mean(axis=(0, 1)),
        )
    raise ValueError("MMMFit has neither map_params nor trace.")
