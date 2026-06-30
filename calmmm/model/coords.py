from __future__ import annotations

import numpy as np

from calmmm.data.containers import MMMData


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
    for _, row in data.controls.iterrows():
        ti = t_idx.get(row["time"])
        gi = g_idx.get(row["geo"])
        ni = n_idx.get(row["control"])
        if ti is not None and gi is not None and ni is not None:
            ctrl_array[ti, gi, ni] = row["value"]

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
