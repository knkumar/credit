from __future__ import annotations

import numpy as np


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Recursive geometric adstock: x_out[t] = x[t] + decay * x_out[t-1].

    decay must be in [0, 1).

    NOTE: This is a pure-NumPy reference implementation. When wiring into a
    PyMC model, a pytensor.scan-compatible version will be required so the
    recurrence is traceable and differentiable by PyTensor's autodiff.
    """
    if not (0.0 <= decay < 1.0):
        raise ValueError(f"decay must be in [0, 1), got {decay}")
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + decay * out[t - 1]
    return out


def weibull_adstock(
    x: np.ndarray,
    shape: float,
    scale: float,
    n_lags: int = 13,
) -> np.ndarray:
    """
    Weibull lag-kernel adstock via convolution.

    Lag weights are the Weibull PDF evaluated at lags 1..n_lags, normalized to
    sum to 1. shape > 1 gives a delayed peak; shape = 1 is exponential decay;
    shape < 1 is front-loaded with a heavy tail.

    shape: Weibull shape parameter (k), must be > 0
    scale: Weibull scale parameter (lambda), must be > 0
    n_lags: number of lag periods (default 13 = one quarter at weekly frequency)
    """
    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    from scipy.stats import weibull_min  # lazy import — scipy is optional at module load time

    x = np.asarray(x, dtype=float)
    lags = np.arange(1, n_lags + 1, dtype=float)
    weights = weibull_min.pdf(lags, c=shape, scale=scale)
    total = weights.sum()
    weights = weights / total if total > 0 else np.ones(n_lags) / n_lags

    # Causal convolution: out[t] = sum_{l=0}^{n_lags-1} weights[l] * x[t - l]
    return np.convolve(x, weights, mode="full")[: len(x)]
