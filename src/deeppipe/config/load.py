"""Load YAML configuration into typed dataclasses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from deeppipe.config.schema import CtrlConfig, DataConfig, RunConfig, TrainConfig


def _merge_dataclass(cls, data: dict[str, Any] | None):
    if not data:
        return cls()
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in fields})


def load_run_config(
    yaml_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Load ``RunConfig`` from YAML plus optional flat overrides."""
    raw: dict[str, Any] = {}
    if yaml_path and yaml_path.is_file():
        with open(yaml_path) as fh:
            raw = yaml.safe_load(fh) or {}

    if overrides:
        for key, val in overrides.items():
            if key in ("data", "train", "ctrl") and isinstance(val, dict):
                raw.setdefault(key, {}).update(val)
            else:
                raw[key] = val

    data = _merge_dataclass(DataConfig, raw.get("data"))
    train = _merge_dataclass(TrainConfig, raw.get("train"))
    ctrl = _merge_dataclass(CtrlConfig, raw.get("ctrl"))

    cfg = RunConfig(
        data_path=Path(raw.get("data_path", "data/market_data.csv")),
        out_dir=Path(raw.get("out_dir", "outputs")),
        seed=int(raw.get("seed", 42)),
        data=data,
        train=train,
        ctrl=ctrl,
        write_figures=bool(raw.get("write_figures", True)),
        write_zip=bool(raw.get("write_zip", True)),
        checkpoint_path=Path(raw["checkpoint_path"]) if raw.get("checkpoint_path") else None,
    )
    return cfg
