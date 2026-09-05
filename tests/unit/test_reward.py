"""Unit tests for Winkler reward and backtest."""

import numpy as np

from deeppipe.ctrl.backtest import returns_from_prices, width_gated_backtest
from deeppipe.ctrl.reward import interval_score, interval_score_norm


def test_interval_score_inside_is_width():
    s = interval_score(y=100.0, lo=90.0, hi=110.0, p=0.90)
    assert abs(s - 20.0) < 1e-9


def test_interval_score_penalty_below():
    p = 0.90
    alpha = 1 - p
    s = interval_score(y=80.0, lo=90.0, hi=110.0, p=p)
    expected = 20.0 + (2 / alpha) * (90.0 - 80.0)
    assert abs(s - expected) < 1e-9


def test_interval_score_proper_scoring_gaussian():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50_000)
    p = 0.90
    z = 1.6448536269514722
    correct = interval_score(y, -z, z, p).mean()
    too_wide = interval_score(y, -2 * z, 2 * z, p).mean()
    too_narrow = interval_score(y, -0.5 * z, 0.5 * z, p).mean()
    assert correct < too_wide
    assert correct < too_narrow


def test_interval_score_norm():
    s = interval_score(50.0, 40.0, 60.0, p=0.9)
    assert abs(interval_score_norm(50.0, 40.0, 60.0, q=10.0, p=0.9) - s / 10.0) < 1e-9


def test_backtest_invariants():
    widths = np.full(50, 100.0)
    rets = returns_from_prices(np.linspace(100, 120, 51)[:50])
    out = width_gated_backtest(widths, rets, w_ref=100.0)
    assert out["n"] == 50
    assert abs(out["mean_position"] - 1.0) < 1e-9
    assert out["max_drawdown"] <= 0.0
