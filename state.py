"""Controller state features — the context s_t for DeepPIPE-Ctrl.

Design doc §4.2. Eight exogenous market-state features (independent of the
controller's action). All z-scores / statistics are computed **fold-locally
from training rows only** (no leakage), matching the notebook's
``MinMaxScaler`` discipline.

    #  Feature              Formula (fold-local)                    Targets
    1  VIX level            (VIX - mu)/sigma                         G1
    2  VIX daily change     (dVIX - mu)/sigma                        G1
    3  Realized vol 5d      z( std(returns_{t-4:t}) )                G1
    4  Realized vol 20d     z( std(returns_{t-19:t}) )              G1
    5  Momentum 5d          z( sum(returns_{t-4:t}) )                regime
    6  Momentum 20d         z( sum(returns_{t-19:t}) )               regime
    7  OOD distance         (Close - max_train_Close)/max_train_Close G2  <-- centrepiece
    8  Rolling |residual|   trailing mean |y - yhat| / q_hat          error scale

Feature 7 is a normalised live flag for "are we beyond the training price
range?" — the exact regime where the paper showed coverage collapses (G2).

Causality: for a next-step forecast of the close on raw row ``rr`` (made at the
close of ``rr-1``), every feature is evaluated on the last *observed* day
``rr-1``; feature 8 uses only residuals from strictly earlier steps.
"""
import numpy as np
import pandas as pd

from config import (CTRL_RV_SHORT, CTRL_RV_LONG, CTRL_RESID_WIN, TARGET)

FEATURE_NAMES = [
    "vix_level", "vix_change", "rvol_5", "rvol_20",
    "mom_5", "mom_20", "ood_distance", "roll_abs_resid",
]
STATE_DIM = len(FEATURE_NAMES)

_Z_COLS = ["vix", "dvix", "rv5", "rv20", "mom5", "mom20"]


def _feature_frame(raw):
    """Causal rolling market-state series, indexed by raw row (price units)."""
    close = raw[TARGET].astype(float)
    ret = close.pct_change()
    vix = raw["INDIA_VIX_Close"].astype(float) if "INDIA_VIX_Close" in raw else pd.Series(
        np.zeros(len(raw)), index=raw.index
    )
    return pd.DataFrame({
        "vix": vix,
        "dvix": vix.diff(),
        "rv5": ret.rolling(CTRL_RV_SHORT).std(),
        "rv20": ret.rolling(CTRL_RV_LONG).std(),
        "mom5": ret.rolling(CTRL_RV_SHORT).sum(),
        "mom20": ret.rolling(CTRL_RV_LONG).sum(),
    })


def compute_train_stats(raw, tr_end):
    """Fold-local means/stds + train price ceiling, computed on rows [:tr_end]."""
    ff = _feature_frame(raw).iloc[:tr_end]
    st = {}
    for c in _Z_COLS:
        mu = float(ff[c].mean())
        sd = float(ff[c].std())
        st[c + "_mu"] = 0.0 if np.isnan(mu) else mu
        st[c + "_sd"] = sd if (sd and not np.isnan(sd)) else 1.0
    st["max_train_close"] = float(raw[TARGET].iloc[:tr_end].max())
    return st


def build_state_matrix(raw, train_stats, point_preds, y_true, target_rows,
                       q, resid_win=CTRL_RESID_WIN):
    """Full (N, STATE_DIM) state matrix for a rollout/bandit period.

    Parameters
    ----------
    raw : DataFrame from ``load_raw``.
    train_stats : dict from ``compute_train_stats`` (fold-local).
    point_preds, y_true : arrays (price units) aligned to ``target_rows``.
    target_rows : int raw-row index of the *predicted* close for each step.
    q : base conformal half-width q_hat_alpha (normalises feature 8).
    """
    ff = _feature_frame(raw)
    close = raw[TARGET].astype(float).values
    max_tc = train_stats["max_train_close"]
    target_rows = np.asarray(target_rows, dtype=int)
    n_rows = len(close)
    resid = np.abs(np.asarray(y_true, dtype=float) - np.asarray(point_preds, dtype=float))
    q = float(q) if q else 1.0

    def z(col, row):
        return (ff[col].iloc[row] - train_stats[col + "_mu"]) / train_stats[col + "_sd"]

    N = len(target_rows)
    S = np.zeros((N, STATE_DIM), dtype=float)
    for j, rr in enumerate(target_rows):
        d = min(max(rr - 1, 0), n_rows - 1)  # last observed day (clamped)
        lo = max(0, j - resid_win)
        f8 = float(resid[lo:j].mean()) / q if j > 0 else 0.0
        row = [
            z("vix", d), z("dvix", d), z("rv5", d), z("rv20", d),
            z("mom5", d), z("mom20", d),
            (close[d] - max_tc) / max_tc,   # OOD distance (feature 7)
            f8,                              # rolling |residual| (feature 8)
        ]
        S[j] = np.nan_to_num(np.asarray(row, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return S


def build_controller_state(raw, split, train_stats, point_preds, y_true,
                           target_rows, q, t):
    """Single-step state s_t (design doc §8.3 signature). Row ``t`` of the matrix."""
    return build_state_matrix(raw, train_stats, point_preds, y_true, target_rows, q)[t]
