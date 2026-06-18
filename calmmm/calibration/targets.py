from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from calmmm.data.containers import IncrementalityTests, MMMData


@dataclass
class CalibrationTarget:
    """
    An incrementality experiment expressed as integer indices into the model arrays.

    Attributes
    ----------
    test_id : str — unique experiment identifier
    t_indices : int array — time indices into the TRAINING-FILTERED axis
                (i.e. indices into channel_contrib[T_train, ...])
    g_indices : int array — geo indices; use all geos if experiment is national
    c_indices : int array — channel indices for the tested channel bundle
    k_index : int — KPI axis index
    lift_obs : float — observed cumulative lift from the experiment
    se : float — standard error of the lift estimate
    calibration_likelihood : str — "normal" (others deferred)
    estimand : str — "total" (others deferred)
    """

    test_id: str
    t_indices: np.ndarray
    g_indices: np.ndarray
    c_indices: np.ndarray
    k_index: int
    lift_obs: float
    se: float
    calibration_likelihood: str
    estimand: str


def build_calibration_targets(
    experiments: "IncrementalityTests",
    data: "MMMData",
    train_mask: np.ndarray,
) -> list[CalibrationTarget]:
    """
    Convert IncrementalityTests to CalibrationTargets with integer model indices.

    Parameters
    ----------
    experiments : IncrementalityTests
    data : MMMData — provides times, geos, channels, kpis for index lookup
    train_mask : bool array [T] — True for time steps included in training

    Returns
    -------
    list[CalibrationTarget] — one per experiment, with validated index arrays

    Raises
    ------
    ValueError if any experiment's window contains no training time steps.
    """
    times = data.times  # sorted list of pd.Timestamp, length T
    geos = data.geos    # sorted list of str
    channels = data.channels  # sorted list of str
    kpis = data.kpis    # sorted list of str

    g_idx = {g: i for i, g in enumerate(geos)}
    c_idx = {c: i for i, c in enumerate(channels)}
    k_idx = {k: i for i, k in enumerate(kpis)}

    # Map absolute time index → training-filtered index
    train_abs_indices = np.where(train_mask)[0]  # absolute positions of training steps
    abs_to_filtered = {int(abs_i): filt_i for filt_i, abs_i in enumerate(train_abs_indices)}

    targets = []
    for exp in experiments:
        # Collect time indices in experiment window that fall in training
        window_filtered = []
        for abs_i, t in enumerate(times):
            if exp.start_date <= t <= exp.end_date and abs_i in abs_to_filtered:
                window_filtered.append(abs_to_filtered[abs_i])

        if not window_filtered:
            raise ValueError(
                f"Experiment '{exp.test_id}' window "
                f"[{exp.start_date.date()}, {exp.end_date.date()}] "
                f"has no training time steps (all fall in holdout or outside panel)."
            )

        t_indices = np.array(window_filtered, dtype=int)
        unknown_geos = [g for g in exp.geo_scope if g not in g_idx]
        if unknown_geos:
            raise ValueError(
                f"Experiment '{exp.test_id}' references unknown geos: {unknown_geos}. "
                f"Known geos: {sorted(g_idx.keys())}"
            )
        g_indices = np.array([g_idx[g] for g in exp.geo_scope], dtype=int)
        if len(g_indices) == 0:
            # If no specific geos matched, use all (national scope)
            g_indices = np.arange(len(geos), dtype=int)
        c_indices = np.array([c_idx[c] for c in exp.channel_bundle], dtype=int)
        k_index = k_idx[exp.kpi]

        targets.append(
            CalibrationTarget(
                test_id=exp.test_id,
                t_indices=t_indices,
                g_indices=g_indices,
                c_indices=c_indices,
                k_index=k_index,
                lift_obs=exp.lift,
                se=exp.se,
                calibration_likelihood=exp.calibration_likelihood.value,
                estimand=exp.estimand.value,
            )
        )

    return targets
