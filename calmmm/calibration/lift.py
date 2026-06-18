from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from calmmm.calibration.targets import CalibrationTarget
    from calmmm.model.fit import MMMFit


def compute_model_lift(
    fit: "MMMFit",
    targets: list["CalibrationTarget"],
) -> pd.DataFrame:
    """
    Compute model-implied lift for each CalibrationTarget.

    For MAP fits, evaluates mu and channel_contrib at the MAP point.
    For MCMC/VI fits, uses the posterior mean of mu and channel_contrib.

    Lift formula (numpy):
        mu_exp[T_exp, G_exp] = mu[t_indices, :, k_index][:, g_indices]
        cc_exp[T_exp, G_exp] = channel_contrib[t_indices, :, k_index, :][:, :, c_indices].sum(-1)
        lift_model = sum(exp(mu_exp) - exp(mu_exp - cc_exp))

    Parameters
    ----------
    fit : MMMFit
    targets : list[CalibrationTarget]

    Returns
    -------
    pd.DataFrame with columns: test_id, lift_model, lift_obs, se, z_score
    One row per target, empty DataFrame if targets is empty.
    """
    if not targets:
        return pd.DataFrame(columns=["test_id", "lift_model", "lift_obs", "se", "z_score"])

    mu_val, cc_val = _eval_mu_and_channel_contrib(fit)

    rows = []
    for target in targets:
        t = target.t_indices
        g = target.g_indices
        k = target.k_index
        c = target.c_indices

        mu_exp = mu_val[t][:, g, k]                              # [T_exp, G_exp]
        cc_total = cc_val[t][:, g, k, :][:, :, c].sum(axis=-1)  # [T_exp, G_exp]

        lift_model = float((np.exp(mu_exp) - np.exp(mu_exp - cc_total)).sum())
        z_score = (lift_model - target.lift_obs) / target.se

        rows.append({
            "test_id": target.test_id,
            "lift_model": lift_model,
            "lift_obs": target.lift_obs,
            "se": target.se,
            "z_score": z_score,
        })

    return pd.DataFrame(rows)


def _eval_mu_and_channel_contrib(
    fit: "MMMFit",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mu, channel_contrib) as numpy arrays [T_train, G, K] and [T_train, G, K, C].

    For MAP: evaluates at map_params.
    For MCMC/VI: returns posterior mean over chains and draws.
    """
    if fit.map_params is not None:
        # pm.find_MAP() returns a dict that includes deterministic values
        # alongside latent RVs — extract mu and channel_contrib directly.
        mu_val = np.array(fit.map_params["mu"])
        cc_val = np.array(fit.map_params["channel_contrib"])
        return mu_val, cc_val

    if fit.trace is not None:
        # posterior["mu"]: [chains, draws, T_train, G, K]
        mu_val = fit.trace.posterior["mu"].values.mean(axis=(0, 1))
        # posterior["channel_contrib"]: [chains, draws, T_train, G, K, C]
        cc_val = fit.trace.posterior["channel_contrib"].values.mean(axis=(0, 1))
        return mu_val, cc_val

    raise ValueError("MMMFit has neither map_params nor trace.")
