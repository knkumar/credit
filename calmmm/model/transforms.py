from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt


# PyTensor (differentiable) counterparts of the NumPy reference implementations
# in calmmm/transforms/adstock.py and calmmm/transforms/saturation.py.
# The NumPy versions validate inputs (e.g. decay ∈ [0,1)); the PyTensor
# versions rely on priors for range enforcement.


def geometric_adstock_pt(X, decay):
    """
    Geometric adstock via pytensor.scan.

    Parameters
    ----------
    X : tensor [T, G, C]
    decay : tensor [C], values in [0, 1]

    Returns
    -------
    tensor [T, G, C]
        Adstocked spend. h[t] = X[t] + decay * h[t-1], h[0] = X[0].
    """
    def _step(x_t, h_prev, decay_):
        # x_t: [G, C], h_prev: [G, C], decay_: [C]
        return x_t + h_prev * decay_[None, :]

    h0 = pt.zeros_like(X[0])  # [G, C]
    h_seq, _ = pytensor.scan(
        _step,
        sequences=[X],
        outputs_info=[h0],
        non_sequences=[decay],
    )
    return h_seq  # [T, G, C]


def hill_saturation_pt(X, alpha, k):
    """
    Hill saturation curve (vectorized over channels).

    Parameters
    ----------
    X : tensor [T, G, C] — input values (should be >= 0)
    alpha : tensor [C] — exponent / steepness (> 0)
    k : tensor [C] — half-saturation point (> 0)

    Returns
    -------
    tensor same shape as X, values in [0, 1]
    """
    # Broadcast alpha and k over leading [T, G] dims for X shape [T, G, C]
    a = alpha[None, None, :]
    kk = k[None, None, :]
    # Match the NumPy reference implementation, including exact zero response
    # for zero or negative input when alpha is small but positive.
    X_safe = pt.clip(X, 0.0, np.inf)
    x_pow = X_safe ** a
    k_pow = kk ** a
    response = x_pow / (x_pow + k_pow)
    return pt.where(pt.eq(X_safe, 0.0), 0.0, response)
