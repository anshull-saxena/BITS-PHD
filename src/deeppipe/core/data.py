"""Data loading, feature engineering, sequence building and splitting."""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from deeppipe.config.schema import TARGET, TrainConfig


def load_raw(data_path):
    """Load the market CSV and engineer technical indicators."""
    raw = pd.read_csv(data_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").drop_duplicates(subset="Date").ffill().bfill()
    c = raw["NIFTY_Close"]
    d = c.diff()
    gain = d.where(d > 0, 0.0).rolling(14).mean()
    loss = (-d.where(d < 0, 0.0)).rolling(14).mean()
    rs = gain / loss
    raw["RSI"] = 100 - (100 / (1 + rs))
    raw["EMA20"] = c.ewm(span=20, adjust=False).mean()
    raw["EMA50"] = c.ewm(span=50, adjust=False).mean()
    raw["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    raw["MACD_SIGNAL"] = raw["MACD"].ewm(span=9, adjust=False).mean()
    raw["RETURN"] = c.pct_change()
    raw["VOLATILITY"] = raw["RETURN"].rolling(10).std()
    return raw.dropna().reset_index(drop=True)


class TSData(Dataset):
    """Simple (X, y) tensor dataset."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_sequences(mat, ti, seq_len):
    """Turn a 2D feature matrix into (windows, next-step-target) pairs."""
    X, y = [], []
    for i in range(len(mat) - seq_len):
        X.append(mat[i : i + seq_len])
        y.append(mat[i + seq_len, ti])
    return np.array(X), np.array(y)


def build_splits(raw, features, tr_end, va_end, train_cfg: TrainConfig | None = None):
    """Scale and window the data into train/val/test splits."""
    cfg = train_cfg or TrainConfig()
    seq_len = cfg.seq_len
    ti = features.index(TARGET)
    tr, va, te = raw.iloc[:tr_end], raw.iloc[tr_end:va_end], raw.iloc[va_end:]
    sc = MinMaxScaler()
    trf = sc.fit_transform(tr[features])
    vaf = sc.transform(va[features])
    tef = sc.transform(te[features])
    Xtr, ytr = make_sequences(trf, ti, seq_len)
    Xva, yva = make_sequences(vaf, ti, seq_len)
    Xte, yte = make_sequences(tef, ti, seq_len)
    vix = te["INDIA_VIX_Close"].values[seq_len:] if "INDIA_VIX_Close" in te else None
    return dict(
        Xtr=Xtr,
        ytr=ytr,
        Xva=Xva,
        yva=yva,
        Xte=Xte,
        yte=yte,
        scaler=sc,
        tgt_idx=ti,
        features=features,
        vix_test=vix,
        _tr_end=tr_end,
        _va_end=va_end,
        _seq_len=seq_len,
    )


def inv_target(v, sc, features, ti):
    """Invert the MinMax scaling for the target column only."""
    v = np.asarray(v).reshape(-1)
    d = np.zeros((len(v), len(features)))
    d[:, ti] = v
    return sc.inverse_transform(d)[:, ti]
