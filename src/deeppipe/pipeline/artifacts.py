"""Artifact directory management."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

from deeppipe.config.schema import RunConfig


def prepare_run_dir(run_cfg: RunConfig, tag: str) -> Path:
    """Create a timestamped run directory under ``run_cfg.out_dir``."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = run_cfg.out_dir / f"{stamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    with open(run_dir / "config.resolved.yaml", "w") as fh:
        yaml.safe_dump(run_cfg.to_dict(), fh, default_flow_style=False, sort_keys=False)
    return run_dir


def write_results_zip(run_dir: Path, results_name: str, figures_subdir: str = "figures") -> Path:
    """Bundle figures + results JSON into a zip archive."""
    zip_base = run_dir / "deeppipe_outputs"
    figures_dir = run_dir / figures_subdir
    results_path = run_dir / results_name
    if figures_dir.is_dir():
        shutil.make_archive(str(zip_base), "zip", run_dir, figures_subdir)
    if results_path.is_file():
        with zipfile.ZipFile(str(zip_base) + ".zip", "a") as z:
            z.write(results_path, arcname=results_name)
    return Path(str(zip_base) + ".zip")


def load_results_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)
