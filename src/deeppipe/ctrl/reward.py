"""Winkler interval score — inner reward for DeepPIPE-Ctrl."""

import numpy as np


def interval_score(y, lo, hi, p=0.90):
    """Winkler interval score S(L, U, y). Lower is better."""
    alpha = 1.0 - p
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    width = hi - lo
    under = (2.0 / alpha) * (lo - y) * (y < lo)
    over = (2.0 / alpha) * (y - hi) * (y > hi)
    s = width + under + over
    return float(s) if np.ndim(s) == 0 else s


def interval_score_norm(y, lo, hi, q, p=0.90):
    if q is None or q == 0:
        raise ValueError("normalising half-width q must be non-zero")
    return interval_score(y, lo, hi, p) / float(q)
