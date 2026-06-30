from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from calmmm.model.fit import eval_mu_and_channel_contrib as _eval_params

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

    exp_mu = np.exp(mu_val)                   # [T, G, K]
    cc_sum = cc_val.sum(axis=-1)              # [T, G, K]
    baseline_contrib = np.exp(mu_val - cc_sum)  # [T, G, K]
    total_media = exp_mu - baseline_contrib   # [T, G, K]

    # Guard against Σcc == 0 (no media spend → channel shares are undefined)
    safe_cc_sum = np.where(cc_sum == 0, 1.0, cc_sum)

    n_cells = T * G * K

    # Index arrays of length n_cells (row-major, matching array ravel order)
    t_idx = np.repeat(np.arange(T), G * K)
    g_idx = np.tile(np.repeat(np.arange(G), K), T)
    k_idx = np.tile(np.tile(np.arange(K), G), T)

    times_arr = np.array(train_times)
    geos_arr = np.array(geos)
    kpis_arr = np.array(kpis)

    all_times = np.tile(times_arr[t_idx], C + 1)
    all_geos = np.tile(geos_arr[g_idx], C + 1)
    all_kpis = np.tile(kpis_arr[k_idx], C + 1)

    # Channel labels: "baseline" + one label per channel, each repeated n_cells times
    channel_labels = np.repeat(np.array(["baseline"] + list(channels)), n_cells)

    # Contribution values: baseline block then C channel blocks
    baseline_flat = baseline_contrib.ravel()
    channel_contribs = []
    for ci in range(C):
        cc_c = cc_val[:, :, :, ci]
        contrib_c = np.where(cc_sum == 0, 0.0, total_media * cc_c / safe_cc_sum)
        channel_contribs.append(contrib_c.ravel())

    all_contributions = np.concatenate([baseline_flat] + channel_contribs)

    return pd.DataFrame({
        "time": all_times,
        "geo": all_geos,
        "kpi": all_kpis,
        "channel": channel_labels,
        "contribution": all_contributions,
    })


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

    T, G, K, C = cc_val.shape

    exp_mu = np.exp(mu_val)  # [T, G, K]

    n_cells = T * G * K

    # Index arrays (row-major, matching array ravel order)
    t_idx = np.repeat(np.arange(T), G * K)
    g_idx = np.tile(np.repeat(np.arange(G), K), T)
    k_idx = np.tile(np.tile(np.arange(K), G), T)

    times_arr = np.array(train_times)
    geos_arr = np.array(geos)
    kpis_arr = np.array(kpis)

    all_times = np.tile(times_arr[t_idx], C)
    all_geos = np.tile(geos_arr[g_idx], C)
    all_kpis = np.tile(kpis_arr[k_idx], C)

    channel_labels = np.repeat(np.array(list(channels)), n_cells)

    channel_contribs = []
    for ci in range(C):
        cc_c = cc_val[:, :, :, ci]
        contrib_c = exp_mu - np.exp(mu_val - cc_c)
        channel_contribs.append(contrib_c.ravel())

    all_contributions = np.concatenate(channel_contribs)

    return pd.DataFrame({
        "time": all_times,
        "geo": all_geos,
        "kpi": all_kpis,
        "channel": channel_labels,
        "contribution": all_contributions,
    })
