from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from calmmm.model.fit import MMMFit


def channel_contributions(fit: "MMMFit") -> pd.DataFrame:
    """
    Additive channel attribution via hybrid proportional decomposition.

    The log-linear model has no unique additive decomposition in outcome space.
    This function uses the following convention:

        baseline_contribution[t,g,k]  = exp(mu - Σcc)
            — the outcome that would remain if all media were removed.
        total_media_increment[t,g,k]  = exp(mu) - baseline_contribution
            — the total incremental outcome from all media combined.
        contribution_c[t,g,k]         = total_media_increment * cc_c / Σcc
            — each channel's share proportional to its log-scale coefficient.
            Zero when Σcc == 0 (no media spend in that cell).

    By construction: baseline + Σ(contribution_c) = exp(mu) for every (t,g,k).

    Note: individual channel contributions can be negative when a channel's
    log-scale coefficient is negative (e.g. geo-level cannibalization effects).
    Use marginal_contributions() when you need the counterfactual removal
    interpretation (e.g. iROAS calculation).

    Returns
    -------
    DataFrame: time, geo, kpi, channel, contribution
        channel is one of the model's channel names or "baseline".
        Only training-time steps are included.
    """
    data = fit.data
    mmm = fit._mmm

    mu_val, cc_val = _eval_params(fit)
    # mu_val: [T_train, G, K]
    # cc_val: [T_train, G, K, C]

    train_mask = mmm._train_mask
    train_times = [t for t, m in zip(data.times, train_mask) if m]
    geos = data.geos
    kpis = data.kpis
    channels = data.channels

    T, G, K, C = cc_val.shape

    exp_mu = np.exp(mu_val)                  # [T, G, K]
    cc_sum = cc_val.sum(axis=-1)             # [T, G, K]
    baseline_contrib = np.exp(mu_val - cc_sum)  # [T, G, K] — counterfactual no-media outcome
    total_media = exp_mu - baseline_contrib  # [T, G, K]

    # Guard against Σcc == 0 (no media spend → channel shares are undefined)
    safe_cc_sum = np.where(cc_sum == 0, 1.0, cc_sum)

    rows = []

    for ti, t in enumerate(train_times):
        for gi, g in enumerate(geos):
            for ki, k in enumerate(kpis):
                rows.append({
                    "time": t, "geo": g, "kpi": k,
                    "channel": "baseline",
                    "contribution": float(baseline_contrib[ti, gi, ki]),
                })

    for ci, ch in enumerate(channels):
        cc_c = cc_val[:, :, :, ci]  # [T, G, K]
        contrib_c = np.where(
            cc_sum == 0, 0.0,
            total_media * cc_c / safe_cc_sum,
        )  # [T, G, K]
        for ti, t in enumerate(train_times):
            for gi, g in enumerate(geos):
                for ki, k in enumerate(kpis):
                    rows.append({
                        "time": t, "geo": g, "kpi": k,
                        "channel": ch,
                        "contribution": float(contrib_c[ti, gi, ki]),
                    })

    return pd.DataFrame(rows, columns=["time", "geo", "kpi", "channel", "contribution"])


def marginal_contributions(fit: "MMMFit") -> pd.DataFrame:
    """
    Counterfactual (marginal removal) channel attribution.

    For each channel c:
        contribution_c[t,g,k] = exp(mu[t,g,k]) - exp(mu[t,g,k] - cc_c[t,g,k])
        — the outcome lost if channel c were removed entirely, all else equal.

    These values do NOT sum to exp(mu); they measure economic impact per channel
    and are the correct input for iROAS / budget optimisation calculations.

    Returns
    -------
    DataFrame: time, geo, kpi, channel, contribution
        Does not include a "baseline" row.
        Only training-time steps are included.
    """
    data = fit.data
    mmm = fit._mmm

    mu_val, cc_val = _eval_params(fit)

    train_mask = mmm._train_mask
    train_times = [t for t, m in zip(data.times, train_mask) if m]
    geos = data.geos
    kpis = data.kpis
    channels = data.channels

    exp_mu = np.exp(mu_val)  # [T, G, K]

    rows = []
    for ci, ch in enumerate(channels):
        cc_c = cc_val[:, :, :, ci]
        contrib_c = exp_mu - np.exp(mu_val - cc_c)  # [T, G, K]
        for ti, t in enumerate(train_times):
            for gi, g in enumerate(geos):
                for ki, k in enumerate(kpis):
                    rows.append({
                        "time": t, "geo": g, "kpi": k,
                        "channel": ch,
                        "contribution": float(contrib_c[ti, gi, ki]),
                    })

    return pd.DataFrame(rows, columns=["time", "geo", "kpi", "channel", "contribution"])


def _eval_params(fit):
    """Return (mu_val [T,G,K], cc_val [T,G,K,C]) as numpy arrays."""
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
