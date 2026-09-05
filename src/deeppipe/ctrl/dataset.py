"""Bandit dataset construction for DeepPIPE-Ctrl."""

import numpy as np

from deeppipe.config.schema import CtrlConfig, TrainConfig
from deeppipe.core.conformal import conformal_q, predict_split
from deeppipe.ctrl.reward import interval_score
from deeppipe.ctrl.state import build_state_matrix, compute_train_stats

__all__ = ["make_bandit_dataset", "conformal_q", "predict_split"]


def make_bandit_dataset(
    raw,
    sp,
    model,
    q,
    tr_end,
    split="Xte",
    ysplit="yte",
    row_offset=None,
    actions=None,
    p=None,
    train_cfg: TrainConfig | None = None,
    ctrl_cfg: CtrlConfig | None = None,
):
    cfg = train_cfg or TrainConfig()
    ctrl = ctrl_cfg or CtrlConfig()
    if actions is None:
        actions = list(ctrl.actions)
    if p is None:
        p = cfg.conf
    q = float(q)
    if q <= 0:
        raise ValueError(f"conformal half-width q must be positive, got {q}")
    y, point = predict_split(model, sp, split, ysplit, cfg)
    N = len(y)
    seq_len = sp.get("_seq_len", cfg.seq_len)
    if row_offset is None:
        row_offset = sp.get("_va_end", 0) + seq_len
    target_rows = np.arange(row_offset, row_offset + N)

    train_stats = compute_train_stats(raw, tr_end, ctrl)
    S = build_state_matrix(raw, train_stats, point, y, target_rows, q, ctrl)

    A = list(actions)
    R = np.zeros((N, len(A)), dtype=float)
    for a_i, m in enumerate(A):
        lo = point - m * q
        hi = point + m * q
        R[:, a_i] = -interval_score(y, lo, hi, p)
    R = R / q

    return dict(
        S=S,
        A=A,
        R=R,
        point=point,
        y=y,
        lo0=point - q,
        hi0=point + q,
        q=q,
        target_rows=target_rows,
        train_stats=train_stats,
    )
