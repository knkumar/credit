from __future__ import annotations

from typing import TYPE_CHECKING

import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    from calmmm.calibration.targets import CalibrationTarget


def add_calibration_likelihood(
    model: pm.Model,
    targets: list["CalibrationTarget"],
) -> None:
    """
    Add one pm.Normal calibration likelihood node per target to the current model.

    Must be called inside a `with model:` context (or the model context is entered
    internally). For each target:

        lift_model_e = sum_{t in t_indices, g in g_indices} [
            exp(mu[t,g,k]) - exp(mu[t,g,k] - sum_{c in c_indices} channel_contrib[t,g,k,c])
        ]
        pm.Normal("lift_obs_{test_id}", mu=lift_model_e, sigma=se, observed=lift_obs)

    Parameters
    ----------
    model : pm.Model — must contain "mu" [T_train, G, K] and "channel_contrib" [T_train, G, K, C]
    targets : list[CalibrationTarget]

    Raises
    ------
    NotImplementedError for estimands other than "total".
    """
    if not targets:
        return

    mu = model["mu"]                         # [T_train, G, K]
    channel_contrib = model["channel_contrib"]  # [T_train, G, K, C]

    for target in targets:
        if target.estimand != "total":
            raise NotImplementedError(
                f"Estimand '{target.estimand}' is not supported in MVP. "
                "Only 'total' estimand is implemented."
            )

        t = target.t_indices   # [T_exp]
        g = target.g_indices   # [G_exp]
        k = target.k_index     # int
        c = target.c_indices   # [C_exp]

        # Slice mu for experiment window and geos: [T_exp, G_exp]
        mu_exp = mu[t][:, g, k]  # [T_exp, G_exp]

        # Sum channel contributions over experiment channels: [T_exp, G_exp]
        cc_exp = channel_contrib[t][:, g, k, :][:, :, c].sum(axis=-1)

        # Counterfactual: remove experiment channel contributions
        mu_cf = mu_exp - cc_exp  # [T_exp, G_exp]

        # Lift = sum of (factual outcome - counterfactual outcome) over window
        lift_model = (pt.exp(mu_exp) - pt.exp(mu_cf)).sum()

        cal_lik = target.calibration_likelihood
        if cal_lik == "normal":
            pm.Normal(
                f"lift_obs_{target.test_id}",
                mu=lift_model,
                sigma=target.se,
                observed=target.lift_obs,
            )
        elif cal_lik == "student_t":
            pm.StudentT(
                f"lift_obs_{target.test_id}",
                nu=target.student_t_nu,
                mu=lift_model,
                sigma=target.se,
                observed=target.lift_obs,
            )
        else:
            raise NotImplementedError(
                f"Calibration likelihood '{cal_lik}' is not yet implemented. "
                "Supported: 'normal', 'student_t'."
            )
