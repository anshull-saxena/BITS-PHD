"""DeepPIPE CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from deeppipe.config.load import load_run_config
from deeppipe.pipeline.ctrl import CtrlPipeline
from deeppipe.pipeline.reproduce import ReproducePipeline
from deeppipe.reporting.figures import make_figures
from deeppipe.reporting.tables import print_tables

app = typer.Typer(help="DeepPIPE NIFTY 50 — paper reproduction and adaptive calibration.")


def _default_config_path(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / name


@app.command()
def reproduce(
    data: Path = typer.Option(
        Path("data/market_data.csv"), "--data", help="Path to market_data.csv"
    ),
    out_dir: Path = typer.Option(Path("outputs"), "--out-dir", help="Output root directory"),
    config: Optional[Path] = typer.Option(None, "--config", help="YAML config file"),
    seed: int = typer.Option(42, "--seed"),
    no_figures: bool = typer.Option(False, "--no-figures"),
    no_zip: bool = typer.Option(False, "--no-zip"),
    reuse_results: bool = typer.Option(False, "--reuse-results"),
):
    """Run experiments E1-E10 (paper reproduction)."""
    cfg_path = config or _default_config_path("reproduce.yaml")
    run_cfg = load_run_config(
        cfg_path, overrides={"data_path": data, "out_dir": out_dir, "seed": seed}
    )
    run_cfg.data_path = data
    run_cfg.out_dir = out_dir
    run_cfg.seed = seed
    run_cfg.write_figures = not no_figures
    run_cfg.write_zip = not no_zip
    ReproducePipeline(run_cfg).run(reuse_results=reuse_results)


@app.command()
def ctrl(
    data: Path = typer.Option(Path("data/market_data.csv"), "--data"),
    out_dir: Path = typer.Option(Path("outputs"), "--out-dir"),
    config: Optional[Path] = typer.Option(None, "--config"),
    seed: int = typer.Option(42, "--seed"),
    reuse_results: bool = typer.Option(False, "--reuse-results"),
):
    """Run experiments E11-E14 (DeepPIPE-Ctrl)."""
    cfg_path = config or _default_config_path("ctrl.yaml")
    run_cfg = load_run_config(
        cfg_path, overrides={"data_path": data, "out_dir": out_dir, "seed": seed}
    )
    run_cfg.data_path = data
    run_cfg.out_dir = out_dir
    run_cfg.ctrl.seed = seed
    CtrlPipeline(run_cfg).run(reuse_results=reuse_results)


@app.command()
def figures(
    results: Path = typer.Argument(..., help="Path to results_all.json"),
    out_dir: Path = typer.Option(Path("figures"), "--out-dir"),
):
    """Re-render figures from an existing results JSON."""
    with open(results) as fh:
        R = json.load(fh)
    make_figures(R, out_dir=str(out_dir))
    print_tables(R)


@app.command(name="config")
def config_show(
    config: Optional[Path] = typer.Option(None, "--config"),
    profile: str = typer.Option("reproduce", "--profile", help="reproduce or ctrl"),
):
    """Print resolved configuration."""
    cfg_path = config or _default_config_path(f"{profile}.yaml")
    run_cfg = load_run_config(cfg_path)
    typer.echo(yaml.safe_dump(run_cfg.to_dict(), default_flow_style=False, sort_keys=False))


def main():
    app()


if __name__ == "__main__":
    main()
