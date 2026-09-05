"""Width-gated economic backtest."""

import numpy as np

TRADING_DAYS = 252


def _max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def width_gated_backtest(widths, returns, w_ref=None, trailing=20):
    widths = np.asarray(widths, dtype=float)
    returns = np.asarray(returns, dtype=float)
    n = min(len(widths), len(returns))
    widths, returns = widths[:n], returns[:n]
    if n == 0:
        return dict(n=0, ann_return=0.0, ann_vol=0.0, sharpe=0.0, max_drawdown=0.0)

    if w_ref is None:
        w_ref_series = np.empty(n)
        for t in range(n):
            lo = max(0, t - trailing)
            window = widths[lo:t] if t > 0 else widths[:1]
            w_ref_series[t] = np.median(window)
    else:
        w_ref_series = np.full(n, float(w_ref))

    pos = np.clip(w_ref_series / np.where(widths == 0, np.nan, widths), 0.0, 1.0)
    pos = np.nan_to_num(pos, nan=0.0)
    strat = pos * returns
    equity = np.cumprod(1.0 + strat)

    mean, sd = float(np.mean(strat)), float(np.std(strat))
    ann_return = mean * TRADING_DAYS
    ann_vol = sd * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0.0
    return dict(
        n=int(n),
        ann_return=float(ann_return),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        max_drawdown=_max_drawdown(equity),
        final_equity=float(equity[-1]),
        mean_position=float(np.mean(pos)),
    )


def returns_from_prices(y_true):
    y = np.asarray(y_true, dtype=float)
    r = np.zeros_like(y)
    r[:-1] = (y[1:] - y[:-1]) / y[:-1]
    return r
