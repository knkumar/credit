from __future__ import annotations

from typing import Optional

import numpy as np


class MediaScaler:
    """
    Scales media spend to [0, 1] by dividing by the per-channel panel maximum.

    Stores the max so that model-space predictions can be converted back to
    spend units via inverse_transform.
    """

    def __init__(self) -> None:
        self._max: Optional[float] = None

    def fit_transform(self, spend: np.ndarray) -> np.ndarray:
        spend = np.asarray(spend, dtype=float)
        self._max = float(spend.max())
        if self._max == 0.0:
            return np.zeros_like(spend)
        return spend / self._max

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        if self._max is None:
            raise RuntimeError("MediaScaler is not fitted; call fit_transform first")
        return np.asarray(scaled, dtype=float) * self._max

    @property
    def max_spend(self) -> float:
        if self._max is None:
            raise RuntimeError("MediaScaler is not fitted")
        return self._max
