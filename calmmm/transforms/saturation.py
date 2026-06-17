from __future__ import annotations

import numpy as np


def hill_saturation(x: np.ndarray, alpha: float, K: float) -> np.ndarray:
    """
    Hill (power) saturation curve.

    f(x) = x^alpha / (x^alpha + K^alpha)

    At x=K, f(K) = 0.5 for any alpha > 0.
    alpha: shape (curvature), must be > 0
    K: half-saturation point, must be > 0
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    x = np.asarray(x, dtype=float)
    x_pow = np.power(np.clip(x, 0.0, None), alpha)
    K_pow = K ** alpha
    return x_pow / (x_pow + K_pow)


def ispline_basis(x: np.ndarray, n_knots: int = 4, degree: int = 3) -> np.ndarray:
    """
    I-spline basis matrix guaranteeing monotone non-decreasing response.

    Each column is the numerical integral of a cubic B-spline basis function,
    normalized to [0, 1]. A linear combination with non-negative weights is
    guaranteed to be monotone non-decreasing.

    x: 1-D evaluation points (need not be sorted)
    n_knots: number of interior knots placed at equally-spaced quantiles of x
    degree: B-spline degree (3 = cubic)

    Returns array of shape (len(x), n_basis) where n_basis = n_knots + degree - 1.
    """
    from scipy.interpolate import BSpline

    x = np.asarray(x, dtype=float)
    x_min, x_max = x.min(), x.max()

    if x_max <= x_min:
        raise ValueError("x must have at least two distinct values")

    quantiles = np.linspace(0, 100, n_knots + 2)[1:-1]
    interior_knots = np.percentile(x, quantiles)

    t = np.concatenate([
        np.full(degree, x_min),
        interior_knots,
        np.full(degree, x_max),
    ])

    n_basis = len(t) - degree - 1
    B = np.zeros((len(x), n_basis))

    # Dense grid for numerical integration
    x_dense = np.linspace(x_min, x_max, max(500, len(x) * 5))
    dx = x_dense[1] - x_dense[0]

    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spl = BSpline(t, c, degree, extrapolate=False)
        b_dense = np.nan_to_num(spl(x_dense), nan=0.0)
        cumint = np.cumsum(b_dense) * dx
        max_val = cumint[-1]
        if max_val > 0:
            cumint = cumint / max_val
        B[:, i] = np.interp(x, x_dense, cumint)

    return np.clip(B, 0.0, 1.0)
