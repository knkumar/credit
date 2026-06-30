from __future__ import annotations

import numpy as np


def fourier_features(
    t: np.ndarray,
    n_pairs: int = 2,
    period: float = 52.0,
) -> np.ndarray:
    """
    Fourier seasonal features.

    For each harmonic n = 1..n_pairs, produces sin(2π n t / period) and
    cos(2π n t / period). Returns array of shape (len(t), 2 * n_pairs).

    Default period=52 assumes weekly data with annual seasonality.
    """
    t = np.asarray(t, dtype=float)
    cols = []
    for n in range(1, n_pairs + 1):
        angle = 2.0 * np.pi * n * t / period
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.column_stack(cols)
