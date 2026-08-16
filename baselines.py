"""Non-learned baselines the controller must beat (design doc §6).

All baselines share the same yhat and q_hat_alpha as the learned tiers.

    static_conformal  -- the paper's E9: m_t = 1 always (the thing being upgraded)
    conformal_pid     -- Angelopoulos et al. 2023 PID controller on coverage error;
                         the hand-tuned *adaptive* bar T2 has to clear
"""
import numpy as np

import config


def static_conformal(point, q, actions=None):
    """m_t = 1 for every step (paper's static split-conformal)."""
    m = np.ones(len(point), dtype=float)
    return dict(name="static_conformal", m=m,
                lo=point - q, hi=point + q)


def conformal_pid(point, y, q, p=None, k_p=0.6, k_i=0.05, k_d=0.1,
                  m_lo=0.5, m_hi=1.5, ema_beta=0.1):
    """Conformal-PID width controller (design doc §6.3).

        e_t     = p_target - EMA_coverage_t
        m_{t+1} = clip( m_t + k_p e_t + k_i sum e + k_d de_t, 0.5, 1.5 )

    Causal: the multiplier for step t is formed from coverage feedback observed
    strictly before t. Returns the realised per-step multipliers and bounds.
    """
    if p is None:
        p = config.CONF
    n = len(point)
    m = np.ones(n, dtype=float)
    lo = np.empty(n)
    hi = np.empty(n)
    ema_cov = p          # seed at target so early steps start neutral
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
