from __future__ import annotations

import pytensor
import pytensor.tensor as pt


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
    h_seq = pytensor.scan(
        _step,
        sequences=[X],
        outputs_info=[h0],
        non_sequences=[decay],
        return_updates=False,
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
    x_pow = X ** a
    k_pow = kk ** a
    return x_pow / (x_pow + k_pow + 1e-9)
