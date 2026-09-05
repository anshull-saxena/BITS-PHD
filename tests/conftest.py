"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest

from deeppipe.config.schema import CtrlConfig, DataConfig, RunConfig, TrainConfig


@pytest.fixture
def synthetic_market_df():
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    close = 8000 + np.cumsum(rng.normal(5, 40, n))
    return pd.DataFrame(
        {
            "Date": dates,
            "NIFTY_Open": close + rng.normal(0, 10, n),
            "NIFTY_High": close + np.abs(rng.normal(20, 10, n)),
            "NIFTY_Low": close - np.abs(rng.normal(20, 10, n)),
            "NIFTY_Close": close,
            "NIFTY_Volume": rng.integers(100_000, 1_000_000, n),
            "INDIA_VIX_Close": 15 + np.abs(rng.normal(0, 3, n)),
            "SP500_Close": 3000 + np.cumsum(rng.normal(1, 15, n)),
            "NASDAQ_Close": 9000 + np.cumsum(rng.normal(1, 30, n)),
            "DOW_Close": 25000 + np.cumsum(rng.normal(1, 80, n)),
            "NIKKEI_Close": 20000 + np.cumsum(rng.normal(1, 60, n)),
            "HANGSENG_Close": 26000 + np.cumsum(rng.normal(1, 70, n)),
            "BRENT_Close": 60 + np.cumsum(rng.normal(0, 0.5, n)),
            "GOLD_Close": 1200 + np.cumsum(rng.normal(0, 2, n)),
            "USDINR_Close": 70 + np.cumsum(rng.normal(0, 0.05, n)),
        }
    )


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_market_df):
    path = tmp_path / "market_data.csv"
    synthetic_market_df.to_csv(path, index=False)
    return path


@pytest.fixture
def fast_run_config(synthetic_csv, tmp_path):
    return RunConfig(
        data_path=synthetic_csv,
        out_dir=tmp_path / "outputs",
        seed=1,
        data=DataConfig(),
        train=TrainConfig(seq_len=20, warmup_ep=2, full_ep=3, batch=16),
        ctrl=CtrlConfig(tiers=["greedy"], seed=1),
        write_figures=False,
        write_zip=False,
    )
