"""Configuration constants and feature-set definitions.

Ported from Cell 2 of DeepPIPE_Colab_AllInOne.ipynb.
"""
import time

import torch

# --- Data / training hyper-parameters ---
DATA_PATH = "market_data.csv"
TARGET = "NIFTY_Close"
SEQ_LEN = 60
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.2
BATCH = 32
LR = 5e-4
WARMUP_EP = 40
FULL_EP = 120
BETA = 0.5
CONF = 0.90
LAMBDA = 15.0
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Feature groups ---
F_NIFTY = ["NIFTY_Open", "NIFTY_High", "NIFTY_Low", "NIFTY_Close", "NIFTY_Volume"]
F_VIX = ["INDIA_VIX_Close"]
F_GLOBAL = ["SP500_Close", "NASDAQ_Close", "DOW_Close", "NIKKEI_Close", "HANGSENG_Close"]
F_COMFX = ["BRENT_Close", "GOLD_Close", "USDINR_Close"]
F_TECH = ["RSI", "EMA20", "EMA50", "MACD", "MACD_SIGNAL", "RETURN", "VOLATILITY"]

FEATURE_SETS = {
    "nifty_only": F_NIFTY,
    "plus_global": F_NIFTY + F_VIX + F_GLOBAL,
    "core14": F_NIFTY + F_VIX + F_GLOBAL + F_COMFX,
    "full21": F_NIFTY + F_VIX + F_GLOBAL + F_COMFX + F_TECH,
}
MAIN_FEATURES = FEATURE_SETS["core14"]


def log(m):
    """Timestamped stdout logger."""
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
