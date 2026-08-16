"""Outer (eval-time) economic backtest — width-gated position sizing.

Design doc §7.2. We never optimise P&L; we optimise calibration and *report*
economic value as a consequence. Daily, long-only, causal:

    w_t     = 2 * m_t * q_hat            # today's interval width
    p_t     = clip(w_ref / w_t, 0, 1)    # narrow => confident => larger position
    r_{t+1} = p_t * (NIFTY daily return) # tomorrow's return

``w_ref`` defaults to the trailing median width (causal). Reports annualised
return, vol, Sharpe (rf=0) and max drawdown.
"""
import numpy as np

TRADING_DAYS = 252


def _max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def width_gated_backtest(widths, returns, w_ref=None, trailing=20):
    """Width-gated long-only backtest.

    Parameters
    ----------
    widths : per-step interval widths (2*m_t*q), aligned to ``returns``.
    returns : NIFTY daily returns realised *after* each interval is formed
              (already lagged by the caller so the strategy is causal).
    w_ref : reference width; if None, uses the causal trailing median.
    """
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
    """Realised next-day returns from the actual close series (causal lag)."""
    y = np.asarray(y_true, dtype=float)
    r = np.zeros_like(y)
    r[:-1] = (y[1:] - y[:-1]) / y[:-1]
    return r
