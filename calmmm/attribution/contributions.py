from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scipy.special import expit

from calmmm.model.fit import eval_mu_and_channel_contrib as _eval_params
from calmmm.model.fit import get_sigma_val

if TYPE_CHECKING:
    from calmmm.model.fit import MMMFit


def channel_contributions(fit: "MMMFit", chunk_size: int = 100) -> pd.DataFrame:
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
    is_mcmc = mu_val.ndim > 3
    # mu_val: [S, T_train, G, K] or [T_train, G, K]
    # cc_val: [S, T_train, G, K, C] or [T_train, G, K, C]

    train_mask = mmm._train_mask
    train_times = [t for t, m in zip(data.times, train_mask) if m]
    geos = data.geos
    kpis = data.kpis
    channels = data.channels

    if is_mcmc:
        S, T, G, K, C = cc_val.shape
    else:
        T, G, K, C = cc_val.shape
        S = 1
        mu_val = mu_val[np.newaxis, ...]
        cc_val = cc_val[np.newaxis, ...]

    likelihoods = {
        kpi: data.kpi_metadata.loc[data.kpi_metadata["kpi"] == kpi, "likelihood"].values[0]
        for kpi in kpis
    }
    precomputed_sigma = {
        kpi: get_sigma_val(fit, kpi, mu_val.ndim)
        for kpi in kpis if likelihoods[kpi] == "lognormal"
    }

    baseline_contrib_mean = np.zeros((T, G, K))
    channel_contribs_mean = [np.zeros((T, G, K)) for _ in range(C)]

    for start_idx in range(0, S, chunk_size):
        end_idx = min(start_idx + chunk_size, S)
        mu_chunk = mu_val[start_idx:end_idx]
        cc_chunk = cc_val[start_idx:end_idx]
        
        cc_sum_chunk = cc_chunk.sum(axis=-1)
        exp_mu_chunk = np.zeros_like(mu_chunk)
        baseline_contrib_chunk = np.zeros_like(mu_chunk)
        total_media_chunk = np.zeros_like(mu_chunk)
        
        for k, kpi in enumerate(kpis):
            likelihood = likelihoods[kpi]
            if likelihood == "binomial":
                pop_k = mmm._pop_array[mmm._train_mask][:, :, k]
                exp_mu_chunk[..., :, k] = expit(mu_chunk[..., :, k]) * pop_k
                baseline_contrib_chunk[..., :, k] = expit(mu_chunk[..., :, k] - cc_sum_chunk[..., :, k]) * pop_k
                total_media_chunk[..., :, k] = exp_mu_chunk[..., :, k] - baseline_contrib_chunk[..., :, k]
            elif likelihood == "lognormal":
                sigma_chunk = precomputed_sigma[kpi][start_idx:end_idx] if is_mcmc else precomputed_sigma[kpi]
                exp_mu_chunk[..., :, k] = np.exp(mu_chunk[..., :, k] + sigma_chunk**2 / 2.0)
                baseline_contrib_chunk[..., :, k] = np.exp(mu_chunk[..., :, k] - cc_sum_chunk[..., :, k] + sigma_chunk**2 / 2.0)
                total_media_chunk[..., :, k] = baseline_contrib_chunk[..., :, k] * np.expm1(cc_sum_chunk[..., :, k])
            else:
                exp_mu_chunk[..., :, k] = np.exp(mu_chunk[..., :, k])
                baseline_contrib_chunk[..., :, k] = np.exp(mu_chunk[..., :, k] - cc_sum_chunk[..., :, k])
                total_media_chunk[..., :, k] = baseline_contrib_chunk[..., :, k] * np.expm1(cc_sum_chunk[..., :, k])

        safe_cc_sum_chunk = np.where(cc_sum_chunk == 0, 1.0, cc_sum_chunk)
        media_ratio_chunk = np.where(cc_sum_chunk == 0, 0.0, total_media_chunk / safe_cc_sum_chunk)

        baseline_contrib_mean += baseline_contrib_chunk.sum(axis=0) / S
        for ci in range(C):
            channel_contribs_mean[ci] += (media_ratio_chunk * cc_chunk[..., ci]).sum(axis=0) / S

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
    baseline_flat = baseline_contrib_mean.flatten(order='C')
    channel_contribs = []
    for ci in range(C):
        channel_contribs.append(channel_contribs_mean[ci].flatten(order='C'))

    all_contributions = np.concatenate([baseline_flat] + channel_contribs)

    return pd.DataFrame({
        "time": all_times,
        "geo": all_geos,
        "kpi": all_kpis,
        "channel": channel_labels,
        "contribution": all_contributions,
    })


def marginal_contributions(fit: "MMMFit", chunk_size: int = 100) -> pd.DataFrame:
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
    is_mcmc = mu_val.ndim > 3

    train_mask = mmm._train_mask
    train_times = [t for t, m in zip(data.times, train_mask) if m]
    geos = data.geos
    kpis = data.kpis
    channels = data.channels

    if is_mcmc:
        S, T, G, K, C = cc_val.shape
    else:
        T, G, K, C = cc_val.shape
        S = 1
        mu_val = mu_val[np.newaxis, ...]
        cc_val = cc_val[np.newaxis, ...]

    likelihoods = {
        kpi: data.kpi_metadata.loc[data.kpi_metadata["kpi"] == kpi, "likelihood"].values[0]
        for kpi in kpis
    }
    precomputed_sigma = {
        kpi: get_sigma_val(fit, kpi, mu_val.ndim)
        for kpi in kpis if likelihoods[kpi] == "lognormal"
    }

    channel_contribs_mean = [np.zeros((T, G, K)) for _ in range(C)]

    for start_idx in range(0, S, chunk_size):
        end_idx = min(start_idx + chunk_size, S)
        mu_chunk = mu_val[start_idx:end_idx]
        cc_chunk = cc_val[start_idx:end_idx]
        
        exp_mu_chunk = np.zeros_like(mu_chunk)
        for k, kpi in enumerate(kpis):
            likelihood = likelihoods[kpi]
            if likelihood == "binomial":
                pop_k = mmm._pop_array[mmm._train_mask][:, :, k]
                exp_mu_chunk[..., :, k] = expit(mu_chunk[..., :, k]) * pop_k
            elif likelihood == "lognormal":
                sigma_chunk = precomputed_sigma[kpi][start_idx:end_idx] if is_mcmc else precomputed_sigma[kpi]
                exp_mu_chunk[..., :, k] = np.exp(mu_chunk[..., :, k] + sigma_chunk**2 / 2.0)
            else:
                exp_mu_chunk[..., :, k] = np.exp(mu_chunk[..., :, k])
                
        for ci in range(C):
            cc_c_chunk = cc_chunk[..., ci]
            val_chunk = np.empty((end_idx - start_idx, T, G, K), dtype=mu_chunk.dtype)
            
            for k, kpi in enumerate(kpis):
                likelihood = likelihoods[kpi]
                if likelihood == "binomial":
                    pop_k = mmm._pop_array[mmm._train_mask][:, :, k]
                    val_chunk[..., k] = exp_mu_chunk[..., :, k] - (expit(mu_chunk[..., :, k] - cc_c_chunk[..., :, k]) * pop_k)
                elif likelihood == "lognormal":
                    sigma_chunk = precomputed_sigma[kpi][start_idx:end_idx] if is_mcmc else precomputed_sigma[kpi]
                    val_chunk[..., k] = exp_mu_chunk[..., :, k] - np.exp(mu_chunk[..., :, k] - cc_c_chunk[..., :, k] + sigma_chunk**2 / 2.0)
                else:
                    val_chunk[..., k] = exp_mu_chunk[..., :, k] - np.exp(mu_chunk[..., :, k] - cc_c_chunk[..., :, k])
                    
            channel_contribs_mean[ci] += val_chunk.sum(axis=0) / S

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
        channel_contribs.append(channel_contribs_mean[ci].flatten(order='C'))

    all_contributions = np.concatenate(channel_contribs)

    return pd.DataFrame({
        "time": all_times,
        "geo": all_geos,
        "kpi": all_kpis,
        "channel": channel_labels,
        "contribution": all_contributions,
    })
