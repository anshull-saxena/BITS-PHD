"""Winkler interval score — the inner (train-time) reward for DeepPIPE-Ctrl.

Design doc §4.4. The interval score is a strictly proper scoring rule
(Gneiting & Raftery 2007): its expectation is uniquely minimised by the
correct p-level interval with minimal width. The RL controller *maximises*
``-interval_score``, which targets conditional calibration (the right
interval at each state) — strictly stronger than the paper's unconditional
PICP, and exactly the property gap G3 identifies.

    S(L, U, y) = (U - L)
               + (2/alpha) * (L - y) * 1[y < L]
               + (2/alpha) * (y - U) * 1[y > U]

with alpha = 1 - p (p is the target coverage, ``config.CONF`` by default).
"""
import numpy as np

import config


def interval_score(y, lo, hi, p=None):
    """Winkler interval score S(L, U, y). Lower is better.

    Scalars or NumPy arrays are both accepted; with arrays the per-element
    score is returned (same shape as broadcast inputs). ``p`` defaults to the
    live ``config.CONF`` so E6-style level sweeps are respected.
    """
    if p is None:
        p = config.CONF
    alpha = 1.0 - p
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    width = hi - lo
    under = (2.0 / alpha) * (lo - y) * (y < lo)
    over = (2.0 / alpha) * (y - hi) * (y > hi)
    s = width + under + over
    return float(s) if np.ndim(s) == 0 else s


def interval_score_norm(y, lo, hi, q, p=None):
    """Interval score normalised by the base half-width ``q`` (design doc §4.4).

    Widths are in index points (thousands), so raw scores are large. Dividing
    by ``q_hat_alpha`` reports the score in "units of base half-width" for
    numerical stability. Monotone scaling leaves the argmax unchanged.
    """
    if q is None or q == 0:
        raise ValueError("normalising half-width q must be non-zero")
    return interval_score(y, lo, hi, p) / float(q)
