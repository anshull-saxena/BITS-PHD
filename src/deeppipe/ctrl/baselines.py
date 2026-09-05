"""Non-learned controller baselines."""

import numpy as np


def static_conformal(point, q, actions=None):
    m = np.ones(len(point), dtype=float)
    return dict(name="static_conformal", m=m, lo=point - q, hi=point + q)


def conformal_pid(
    point,
    y,
    q,
    p=0.90,
    k_p=0.6,
    k_i=0.05,
    k_d=0.1,
    m_lo=0.5,
    m_hi=1.5,
    ema_beta=0.1,
):
    n = len(point)
    m = np.ones(n, dtype=float)
    lo = np.empty(n)
    hi = np.empty(n)
    ema_cov = p
    integ = 0.0
    prev_e = 0.0
    m_t = 1.0
    for t in range(n):
        m[t] = m_t
        lo[t] = point[t] - m_t * q
        hi[t] = point[t] + m_t * q
        hit = 1.0 if (lo[t] <= y[t] <= hi[t]) else 0.0
        ema_cov = (1 - ema_beta) * ema_cov + ema_beta * hit
        e = p - ema_cov
        integ += e
        de = e - prev_e
        prev_e = e
        m_t = float(np.clip(m_t + k_p * e + k_i * integ + k_d * de, m_lo, m_hi))
    return dict(name="conformal_pid", m=m, lo=lo, hi=hi)
