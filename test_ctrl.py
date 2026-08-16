"""M1 sanity checks for the DeepPIPE-Ctrl layer (design doc §9).

Run with:  python test_ctrl.py   (or: pytest test_ctrl.py)

Covers:
  * interval_score against a known closed-form (width-only when y is inside)
  * interval_score proper-scoring behaviour (correct interval beats too-wide/narrow)
  * make_bandit_dataset reward-table shape and the m=1 column reproducing E9
  * policy fit/act returns a valid multiplier from the action grid
  * width_gated_backtest basic invariants
"""
import numpy as np

import config
from reward import interval_score, interval_score_norm
from backtest import width_gated_backtest, returns_from_prices


def test_interval_score_inside_is_width():
    # y strictly inside => score is exactly the width (no penalty terms)
    s = interval_score(y=100.0, lo=90.0, hi=110.0, p=0.90)
    assert abs(s - 20.0) < 1e-9, s


def test_interval_score_penalty_below():
    # y below lower bound: width + (2/alpha)(lo - y)
    p = 0.90
    alpha = 1 - p
    s = interval_score(y=80.0, lo=90.0, hi=110.0, p=p)
    expected = 20.0 + (2 / alpha) * (90.0 - 80.0)
    assert abs(s - expected) < 1e-9, (s, expected)


def test_interval_score_proper_scoring_gaussian():
    # For a standard normal, the true central 90% interval should score better
    # (lower) on average than an over-wide or over-narrow one.
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200_000)
    p = 0.90
    z = 1.6448536269514722  # 95th percentile of N(0,1)
    correct = interval_score(y, -z, z, p).mean()
    too_wide = interval_score(y, -2 * z, 2 * z, p).mean()
    too_narrow = interval_score(y, -0.5 * z, 0.5 * z, p).mean()
    assert correct < too_wide, (correct, too_wide)
    assert correct < too_narrow, (correct, too_narrow)


def test_interval_score_norm():
    s = interval_score(50.0, 40.0, 60.0, p=0.9)
    assert abs(interval_score_norm(50.0, 40.0, 60.0, q=10.0, p=0.9) - s / 10.0) < 1e-9


def test_backtest_invariants():
    widths = np.full(50, 100.0)
    rets = returns_from_prices(np.linspace(100, 120, 51)[:50])
    out = width_gated_backtest(widths, rets, w_ref=100.0)
    assert out["n"] == 50
    # constant width, w_ref == width => position clipped to 1.0 everywhere
    assert abs(out["mean_position"] - 1.0) < 1e-9
    assert out["max_drawdown"] <= 0.0


def _synthetic_raw(n=500, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    close = 8000 + np.cumsum(rng.normal(5, 40, n))
    df = pd.DataFrame({
        "Date": dates,
        "NIFTY_Open": close + rng.normal(0, 10, n),
        "NIFTY_High": close + np.abs(rng.normal(20, 10, n)),
        "NIFTY_Low": close - np.abs(rng.normal(20, 10, n)),
        "NIFTY_Close": close,
        "NIFTY_Volume": rng.integers(1e5, 1e6, n),
        "INDIA_VIX_Close": 15 + np.abs(rng.normal(0, 3, n)),
        "SP500_Close": 3000 + np.cumsum(rng.normal(1, 15, n)),
        "NASDAQ_Close": 9000 + np.cumsum(rng.normal(1, 30, n)),
        "DOW_Close": 25000 + np.cumsum(rng.normal(1, 80, n)),
        "NIKKEI_Close": 20000 + np.cumsum(rng.normal(1, 60, n)),
        "HANGSENG_Close": 26000 + np.cumsum(rng.normal(1, 70, n)),
        "BRENT_Close": 60 + np.cumsum(rng.normal(0, 0.5, n)),
        "GOLD_Close": 1200 + np.cumsum(rng.normal(0, 2, n)),
        "USDINR_Close": 70 + np.cumsum(rng.normal(0, 0.05, n)),
    })
    return df


def test_bandit_and_policy_smoke(tmp_path=None):
    """End-to-end on tiny synthetic data with a shrunk config (fast)."""
    import pandas as pd
    from data import load_raw, build_splits
    from train import train_deeppipe
    from dataset import make_bandit_dataset, conformal_q, predict_split
    from policies import fit_policy, act
    from rollout import roll_controller

    # shrink the model/training so the smoke test runs in seconds
    config.SEQ_LEN = 20
    config.WARMUP_EP = 2
    config.FULL_EP = 3
    import data as _data
    _data.SEQ_LEN = 20  # data.py imported SEQ_LEN by value

    df = _synthetic_raw(400)
    csv = "._synthetic_market.csv"
    df.to_csv(csv, index=False)
    raw = load_raw(csv)
    n = len(raw)
    tr_end = int(n * 0.70)
    va_end = tr_end + int(n * 0.15)
    sp = build_splits(raw, config.MAIN_FEATURES, tr_end, va_end)

    model = train_deeppipe(sp, True, seed=1)
    q, nc = conformal_q(model, sp)
    assert q > 0 and nc > 0

    D = make_bandit_dataset(raw, sp, model, q, tr_end,
                            split="Xte", ysplit="yte",
                            row_offset=va_end + config.SEQ_LEN)
    N = len(D["y"])
    assert D["R"].shape == (N, len(config.CTRL_ACTIONS))
    assert D["S"].shape[1] == 8

    # m=1 column of the reward table reproduces the static-conformal interval score
    from reward import interval_score
    m1 = config.CTRL_ACTIONS.index(1.0)
    static_score = interval_score(D["y"], D["point"] - q, D["point"] + q).mean()
    assert abs((-D["R"][:, m1] * q).mean() - static_score) < 1e-6

    for tier in ("greedy", "linucb", "lints", "mlp"):
        pol = fit_policy(D, tier=tier)
        mt = act(pol, D["S"][0])
        assert mt in config.CTRL_ACTIONS
        roll = roll_controller(D, pol)
        assert set(["PICP", "MPIW", "mean_interval_score"]).issubset(roll["metrics"])

    import os
    os.remove(csv)
    print("smoke: bandit table", D["R"].shape, "state dim", D["S"].shape[1], "OK")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll DeepPIPE-Ctrl sanity checks passed.")
