"""Bandit dataset construction for DeepPIPE-Ctrl.

Design doc §4.1 & §8.3. Because yhat_t, q_hat_alpha and the realized y_{t+1}
are all known after the fact, the reward of *every* action is fully observable
offline — no off-policy correction needed for v1. The bandit dataset is a
complete reward table R in R^{N x |A|}, which reduces policy learning to
offline supervised optimisation against a proper scoring rule.

    make_bandit_dataset(sp, model, q, ...) -> {
        'S': (N, d) state matrix,
        'A': list of multipliers (the action grid),
        'R': (N, |A|) reward table  (reward = -interval_score),
        'point','y','lo0','hi0','target_rows': bookkeeping arrays,
    }
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from config import CTRL_ACTIONS, SEQ_LEN, BATCH, device
from data import TSData, inv_target
from reward import interval_score
from state import compute_train_stats, build_state_matrix


def predict_split(model, sp, kx, ky):
    """Return (y_true, point_pred) in price units for split keys kx/ky."""
    model.eval()
    yt, pr = [], []
    dl = DataLoader(TSData(sp[kx], sp[ky]), batch_size=BATCH, shuffle=False)
    with torch.no_grad():
        for xb, yb in dl:
            o = model(xb.to(device))
            yt.extend(yb.numpy().flatten())
            pr.extend(o["point"].cpu().numpy().flatten())
    s, f, ti = sp["scaler"], sp["features"], sp["tgt_idx"]
    return inv_target(yt, s, f, ti), inv_target(pr, s, f, ti)


def conformal_q(model, sp, tau=None):
    """Split-conformal base half-width q_hat from validation residuals (E9 seam).

    Fold-local, no leakage: uses the fold's own validation split exactly like
    the notebook's E9 cell.
    """
    if tau is None:
        tau = config.CONF
    yv, pv = predict_split(model, sp, "Xva", "yva")
    cal = np.abs(yv - pv)
    nc = len(cal)
    kk = min(nc, int(np.ceil((nc + 1) * tau)))
    return float(np.sort(cal)[kk - 1]), nc


def make_bandit_dataset(raw, sp, model, q, tr_end, split="Xte", ysplit="yte",
                        row_offset=None, actions=None, p=None):
    """Build the offline bandit dataset (state + full reward table).

    ``row_offset`` is the raw-row index of the first *predicted* close in the
    chosen split. For the standard test split this is ``va_end + SEQ_LEN``
    (each sequence of length SEQ_LEN predicts the next day). Callers that build
    custom splits (e.g. the rolling harness) pass their own offset.
    """
    if actions is None:
        actions = list(CTRL_ACTIONS)
    if p is None:
        p = config.CONF
    y, point = predict_split(model, sp, split, ysplit)
    N = len(y)
    if row_offset is None:
        # Standard split-conformal test seam: va_end is the split start.
        row_offset = sp.get("_va_end", 0) + SEQ_LEN
    target_rows = np.arange(row_offset, row_offset + N)

    train_stats = compute_train_stats(raw, tr_end)
    S = build_state_matrix(raw, train_stats, point, y, target_rows, q)

    A = list(actions)
    R = np.zeros((N, len(A)), dtype=float)
    for a_i, m in enumerate(A):
        lo = point - m * q
        hi = point + m * q
        R[:, a_i] = -interval_score(y, lo, hi, p)  # reward = -score (normalised below)
    R = R / q  # report in units of base half-width; argmax unchanged (§4.4)

    return dict(S=S, A=A, R=R, point=point, y=y,
                lo0=point - q, hi0=point + q, q=q,
                target_rows=target_rows, train_stats=train_stats)
