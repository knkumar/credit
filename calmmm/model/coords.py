from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from calmmm.data.containers import MMMData

logger = logging.getLogger(__name__)


def build_coords(data: MMMData, n_fourier_pairs: int = 2) -> dict[str, list]:
    """Return PyMC coords dict for use in pm.Model(coords=...)."""
    coords: dict[str, list] = {
        "time": list(data.times),
        "geo": data.geos,
        "kpi": data.kpis,
        "channel": data.channels,
        "fourier": list(range(2 * n_fourier_pairs)),
    }
    ctrl_names = sorted(data.controls["control"].unique().tolist()) if not data.controls.empty else []
    if ctrl_names:
        coords["control"] = ctrl_names
    return coords


def build_controls_array(data: MMMData) -> tuple[np.ndarray | None, list[str]]:
    """
    Pivot controls long frame into a dense array.

    Returns
    -------
    ctrl_array : float64 [T, G, N_ctrl] or None if no controls
    ctrl_names : list of control names (sorted)
    """
    if data.controls.empty:
        return None, []

    ctrl_names = sorted(data.controls["control"].unique().tolist())
    times = data.times
    geos = data.geos
    T, G, N = len(times), len(geos), len(ctrl_names)

    t_idx = {t: i for i, t in enumerate(times)}
    g_idx = {g: i for i, g in enumerate(geos)}
    n_idx = {n: i for i, n in enumerate(ctrl_names)}

    ctrl_array = np.zeros((T, G, N), dtype=np.float64)

    df = data.controls
    ti = df["time"].map(t_idx).values
    gi = df["geo"].map(g_idx).values
    ni = df["control"].map(n_idx).values
    # Filter out rows with unmapped keys (map returns NaN for missing)
    valid = ~(np.isnan(ti) | np.isnan(gi) | np.isnan(ni))
    if not valid.all():
        raise ValueError("Extraneous or unmapped rows found in controls data. Please fix your unmapped string keys.")
    ti = ti[valid].astype(int)
    gi = gi[valid].astype(int)
    ni = ni[valid].astype(int)
    if len(np.unique(np.column_stack((ti, gi, ni)), axis=0)) != len(ti):
        raise ValueError("data.controls contains duplicate (time, geo, control) rows")
    ctrl_array[ti, gi, ni] = df["value"].values[valid]

    return ctrl_array, ctrl_names


def build_arrays(
    data: MMMData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pivot MMMData long frames into dense numpy arrays.

    Returns
    -------
    obs_array : float64 [T, G, K] — observed outcomes
    media_array : float64 [T, G, C] — spend (raw, unscaled)
    pop_array : float64 [T, G, K] — population; NaN where unavailable
    """
    times = data.times
    geos = data.geos
    kpis = data.kpis
    channels = data.channels
    T = len(times)
    G = len(geos)
    K = len(kpis)
    C = len(channels)

    t_idx = {t: i for i, t in enumerate(times)}
    g_idx = {g: i for i, g in enumerate(geos)}
    k_idx = {k: i for i, k in enumerate(kpis)}
    c_idx = {c: i for i, c in enumerate(channels)}

    # Observations → [T, G, K]
    obs_array = np.full((T, G, K), np.nan)
    df = data.observations
    ti = df["time"].map(t_idx).values
    gi = df["geo"].map(g_idx).values
    ki = df["kpi"].map(k_idx).values
    
    # Filter valid rows and cast to int
    valid = ~(pd.isna(ti) | pd.isna(gi) | pd.isna(ki))
    if not valid.all():
        raise ValueError("Extraneous or unmapped rows found in observations data. Please fix your unmapped string keys.")
    ti_valid = ti[valid].astype(int)
    gi_valid = gi[valid].astype(int)
    ki_valid = ki[valid].astype(int)
    
    if len(np.unique(np.column_stack((ti_valid, gi_valid, ki_valid)), axis=0)) != len(ti_valid):
        raise ValueError("data.observations contains duplicate (time, geo, kpi) rows")
    obs_array[ti_valid, gi_valid, ki_valid] = df["outcome"].values[valid]

    # Media → [T, G, C]
    media_array = np.zeros((T, G, C))
    mdf = data.media
    mti = mdf["time"].map(t_idx).values
    mgi = mdf["geo"].map(g_idx).values
    mci = mdf["channel"].map(c_idx).values
    
    valid_m = ~(pd.isna(mti) | pd.isna(mgi) | pd.isna(mci))
    if not valid_m.all():
        raise ValueError("Extraneous or unmapped rows found in media data. Please fix your unmapped string keys.")
    mti_valid = mti[valid_m].astype(int)
    mgi_valid = mgi[valid_m].astype(int)
    mci_valid = mci[valid_m].astype(int)

    if len(np.unique(np.column_stack((mti_valid, mgi_valid, mci_valid)), axis=0)) != len(mti_valid):
        raise ValueError("data.media contains duplicate (time, geo, channel) rows")
    media_array[mti_valid, mgi_valid, mci_valid] = mdf["spend"].values[valid_m]

    # Population → [T, G, K]  (reuses ti/gi/ki from observations pivot)
    pop_array = np.full((T, G, K), np.nan)
    if "population" in df.columns:
        valid_pop = valid & df["population"].notna().values
        ti_pop = ti[valid_pop].astype(int)
        gi_pop = gi[valid_pop].astype(int)
        ki_pop = ki[valid_pop].astype(int)
        pop_array[ti_pop, gi_pop, ki_pop] = df["population"].to_numpy()[valid_pop]

    return (
        obs_array.astype(np.float64),
        media_array.astype(np.float64),
        pop_array.astype(np.float64),
    )
