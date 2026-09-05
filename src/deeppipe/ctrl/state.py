"""Controller state features."""

import numpy as np
import pandas as pd

from deeppipe.config.schema import TARGET, CtrlConfig

FEATURE_NAMES = [
    "vix_level",
    "vix_change",
    "rvol_5",
    "rvol_20",
    "mom_5",
    "mom_20",
    "ood_distance",
    "roll_abs_resid",
]
STATE_DIM = len(FEATURE_NAMES)
_Z_COLS = ["vix", "dvix", "rv5", "rv20", "mom5", "mom20"]


def _feature_frame(raw, ctrl_cfg: CtrlConfig | None = None):
    cfg = ctrl_cfg or CtrlConfig()
    close = raw[TARGET].astype(float)
    ret = close.pct_change()
    vix = (
        raw["INDIA_VIX_Close"].astype(float)
        if "INDIA_VIX_Close" in raw
        else pd.Series(np.zeros(len(raw)), index=raw.index)
    )
    return pd.DataFrame(
        {
            "vix": vix,
            "dvix": vix.diff(),
            "rv5": ret.rolling(cfg.rv_short).std(),
            "rv20": ret.rolling(cfg.rv_long).std(),
            "mom5": ret.rolling(cfg.rv_short).sum(),
            "mom20": ret.rolling(cfg.rv_long).sum(),
        }
    )


def compute_train_stats(raw, tr_end, ctrl_cfg: CtrlConfig | None = None):
    ff = _feature_frame(raw, ctrl_cfg).iloc[:tr_end]
    st = {}
    for c in _Z_COLS:
        mu = float(ff[c].mean())
        sd = float(ff[c].std())
        st[c + "_mu"] = 0.0 if np.isnan(mu) else mu
        st[c + "_sd"] = sd if (sd and not np.isnan(sd)) else 1.0
    st["max_train_close"] = float(raw[TARGET].iloc[:tr_end].max())
    return st


def build_state_matrix(
    raw,
    train_stats,
    point_preds,
    y_true,
    target_rows,
    q,
    ctrl_cfg: CtrlConfig | None = None,
):
    cfg = ctrl_cfg or CtrlConfig()
    ff = _feature_frame(raw, cfg)
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
        d = min(max(rr - 1, 0), n_rows - 1)
        lo = max(0, j - cfg.resid_win)
        f8 = float(resid[lo:j].mean()) / q if j > 0 else 0.0
        row = [
            z("vix", d),
            z("dvix", d),
            z("rv5", d),
            z("rv20", d),
            z("mom5", d),
            z("mom20", d),
            (close[d] - max_tc) / max_tc,
            f8,
        ]
        S[j] = np.nan_to_num(np.asarray(row, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return S
