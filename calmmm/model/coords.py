from __future__ import annotations

import numpy as np

from calmmm.data.containers import MMMData


def build_coords(data: MMMData, n_fourier_pairs: int = 2) -> dict[str, list]:
    """Return PyMC coords dict for use in pm.Model(coords=...)."""
    return {
        "time": list(data.times),
        "geo": data.geos,
        "kpi": data.kpis,
        "channel": data.channels,
        "fourier": list(range(2 * n_fourier_pairs)),
    }


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
    if len(set(zip(ti, gi, ki))) != len(ti):
        raise ValueError("data.observations contains duplicate (time, geo, kpi) rows")
    obs_array[ti, gi, ki] = df["outcome"].values

    # Media → [T, G, C]
    media_array = np.zeros((T, G, C))
    mdf = data.media
    mti = mdf["time"].map(t_idx).values
    mgi = mdf["geo"].map(g_idx).values
    mci = mdf["channel"].map(c_idx).values
    if len(set(zip(mti, mgi, mci))) != len(mti):
        raise ValueError("data.media contains duplicate (time, geo, channel) rows")
    media_array[mti, mgi, mci] = mdf["spend"].values

    # Population → [T, G, K]  (reuses ti/gi/ki from observations pivot)
    pop_array = np.full((T, G, K), np.nan)
    if "population" in df.columns:
        valid = df["population"].notna().values
        pop_array[ti[valid], gi[valid], ki[valid]] = df["population"].to_numpy()[valid]

    return (
        obs_array.astype(np.float64),
        media_array.astype(np.float64),
        pop_array.astype(np.float64),
    )
