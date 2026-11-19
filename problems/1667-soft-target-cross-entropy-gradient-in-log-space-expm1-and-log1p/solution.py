import numpy as np


def soft_target_grad_fixed(z_S, z_T, remove_drift=True):
    """Equation 9 of the paper: the soft-target CE gradient computed from delta = z_S - z_T.

    Returns g (or u when remove_drift is False) as an array of the inputs' dtype.
    """
    def softmax(x):
        x = x - x.max(axis=-1,keepdims=True)
        return np.exp(x) / np.exp(x).sum(axis=-1,keepdims=True)

    p = softmax(z_T)
    q = softmax(z_S)
    delta = z_S - z_T
    expml = lambda u: np.expm1(u)
    log1p = lambda u: np.log1p(u)
    S = (p * delta).sum()
    d = delta - S
    s = d - log1p((p * expml(d)).sum())
    u = p * expml(s)
    g = u - q * u.sum()
    return g if remove_drift else u