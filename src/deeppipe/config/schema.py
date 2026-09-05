"""Typed configuration dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --- Feature groups (constants) ---
TARGET = "NIFTY_Close"
F_NIFTY = ["NIFTY_Open", "NIFTY_High", "NIFTY_Low", "NIFTY_Close", "NIFTY_Volume"]
F_VIX = ["INDIA_VIX_Close"]
F_GLOBAL = ["SP500_Close", "NASDAQ_Close", "DOW_Close", "NIKKEI_Close", "HANGSENG_Close"]
F_COMFX = ["BRENT_Close", "GOLD_Close", "USDINR_Close"]
F_TECH = ["RSI", "EMA20", "EMA50", "MACD", "MACD_SIGNAL", "RETURN", "VOLATILITY"]

FEATURE_SETS: dict[str, list[str]] = {
    "nifty_only": F_NIFTY,
    "plus_global": F_NIFTY + F_VIX + F_GLOBAL,
    "core14": F_NIFTY + F_VIX + F_GLOBAL + F_COMFX,
    "full21": F_NIFTY + F_VIX + F_GLOBAL + F_COMFX + F_TECH,
}
MAIN_FEATURES = FEATURE_SETS["core14"]


@dataclass
class DataConfig:
    target: str = TARGET
    train_split: float = 0.70
    val_split: float = 0.15
    feature_set: str = "core14"

    @property
    def features(self) -> list[str]:
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature_set {self.feature_set!r}")
        return FEATURE_SETS[self.feature_set]


@dataclass
class TrainConfig:
    seq_len: int = 60
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    batch: int = 32
    lr: float = 5e-4
    warmup_ep: int = 40
    full_ep: int = 120
    beta: float = 0.5
    conf: float = 0.90
    lam: float = 15.0
    device: str = "auto"  # "auto", "cpu", "cuda"

    def resolve_device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class CtrlConfig:
    actions: list[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5])
    rv_short: int = 5
    rv_long: int = 20
    resid_win: int = 20
    tiers: list[str] = field(
        default_factory=lambda: ["greedy", "linucb", "lints", "mlp"]
    )
    seed: int = 42


@dataclass
class RunConfig:
    data_path: Path = Path("data/market_data.csv")
    out_dir: Path = Path("outputs")
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ctrl: CtrlConfig = field(default_factory=CtrlConfig)
    write_figures: bool = True
    write_zip: bool = True
    checkpoint_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        def _conv(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _conv(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_conv(v) for v in obj]
            return obj

        return _conv(asdict(self))
