"""Statistical tests and aggregation helpers."""

import math

import numpy as np
from scipy import stats


def diebold_mariano(y, p1, p2, h=1, power=2):
    e1 = np.abs(y - p1) ** power
    e2 = np.abs(y - p2) ** power
    d = e1 - e2
    n = len(d)
    db = d.mean()
    var = np.var(d, ddof=0) / n
    if var <= 0:
        return 0.0, 1.0
    dm = db / math.sqrt(var)
    dm *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))


def bootstrap_mae_ci(y, p, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    ae = np.abs(y - p)
    st = [ae[rng.integers(0, n, n)].mean() for _ in range(B)]
    lo, hi = np.percentile(st, [2.5, 97.5])
    return float(ae.mean()), float(lo), float(hi)


def bootstrap_diff_ci(y, p_dp, p_base, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.abs(y - p_dp) - np.abs(y - p_base)
    n = len(d)
    st = [d[rng.integers(0, n, n)].mean() for _ in range(B)]
    lo, hi = np.percentile(st, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def kupiec_pof(cap, conf=0.90):
    n = len(cap)
    x = int((~cap).sum())
    pe = 1 - conf
    pi = x / n if n else 0.0
    if pi in (0.0, 1.0):
        lr = -2 * (x * math.log(pe + 1e-12) + (n - x) * math.log(1 - pe + 1e-12))
    else:
        lr = -2 * (
            (x * math.log(pe) + (n - x) * math.log(1 - pe))
            - (x * math.log(pi) + (n - x) * math.log(1 - pi))
        )
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1)), x, n


def christoffersen_ind(cap):
    hit = (~cap).astype(int)
    n = len(hit)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        a, b = hit[i - 1], hit[i]
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    t0, t1 = n00 + n01, n10 + n11
    if t0 == 0 or t1 == 0:
        return 0.0, 1.0
    p01 = n01 / t0 if t0 else 0
    p11 = n11 / t1 if t1 else 0
    p = (n01 + n11) / (t0 + t1)
    lln = (n00 + n10) * math.log(1 - p + 1e-12) + (n01 + n11) * math.log(p + 1e-12)
    lla = (
        n00 * math.log(1 - p01 + 1e-12)
        + n01 * math.log(p01 + 1e-12)
        + n10 * math.log(1 - p11 + 1e-12)
        + n11 * math.log(p11 + 1e-12)
    )
    lr = -2 * (lln - lla)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def agg(rows, keys=("MAE", "PICP", "MPIW", "RMSE", "R2")):
    return {
        k: dict(
            mean=float(np.mean([r[k] for r in rows])),
            std=float(np.std([r[k] for r in rows])),
        )
        for k in keys
    }
