"""Walk-forward rollout and inner calibration metrics."""

import numpy as np

from deeppipe.core.model import interval_metrics
from deeppipe.core.stats import christoffersen_ind, kupiec_pof
from deeppipe.ctrl.policies import act
from deeppipe.ctrl.reward import interval_score


def _inner_metrics(y, lo, hi, q, p, vix=None, m=None):
    cap = (y >= lo) & (y <= hi)
    klr, kp, kx, kn = kupiec_pof(cap, p)
    clr, cpv = christoffersen_ind(cap)
    scores = interval_score(y, lo, hi, p) / q
    out = dict(
        **interval_metrics(y, lo, hi),
        mean_interval_score=float(np.mean(scores)),
        kupiec=dict(LR=klr, p=kp, failures=kx, n=kn),
        christoffersen=dict(LR=clr, p=cpv),
    )
    width = hi - lo
    if vix is not None and len(vix) >= len(width):
        out["corr_width_vix"] = float(np.corrcoef(vix[: len(width)], width)[0, 1])
    if m is not None:
        out["mean_multiplier"] = float(np.mean(m))
        out["multiplier_hist"] = {
            str(a): int(np.sum(np.isclose(m, a))) for a in sorted(set(np.round(m, 4)))
        }
    return out


def roll_controller(D, policy, p=0.90, vix=None):
    point, y, q = D["point"], D["y"], D["q"]
    S = D["S"]
    n = len(y)
    m = np.array([act(policy, S[t]) for t in range(n)], dtype=float)
    lo = point - m * q
    hi = point + m * q
    return dict(m=m, lo=lo, hi=hi, metrics=_inner_metrics(y, lo, hi, q, p, vix=vix, m=m))


def eval_intervals(y, lo, hi, q, m, p=0.90, vix=None):
    return _inner_metrics(
        np.asarray(y), np.asarray(lo), np.asarray(hi), q, p, vix=vix, m=np.asarray(m)
    )
