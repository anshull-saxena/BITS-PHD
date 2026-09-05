"""Paper reproduction pipeline (E1-E10)."""

from __future__ import annotations

from deeppipe.config.schema import RunConfig
from deeppipe.experiments.reproduce import run_all
from deeppipe.logging import setup_logging
from deeppipe.pipeline.artifacts import prepare_run_dir, write_results_zip
from deeppipe.reporting.figures import make_figures
from deeppipe.reporting.tables import print_tables


class ReproducePipeline:
    def __init__(self, run_cfg: RunConfig):
        self.run_cfg = run_cfg

    def run(self, reuse_results: bool = False) -> dict:
        run_dir = prepare_run_dir(self.run_cfg, "reproduce")
        setup_logging(run_dir)
        results_path = run_dir / "results_all.json"
        figures_dir = run_dir / "figures"

        if reuse_results and results_path.is_file():
            from deeppipe.pipeline.artifacts import load_results_json

            results = load_results_json(results_path)
        else:
            if not self.run_cfg.data_path.is_file():
                raise FileNotFoundError(f"Data file not found: {self.run_cfg.data_path}")
            results = run_all(self.run_cfg, results_path=results_path)

        print_tables(results)

        if self.run_cfg.write_figures:
            make_figures(results, out_dir=str(figures_dir))

        if self.run_cfg.write_zip:
            zip_path = write_results_zip(run_dir, "results_all.json")
            print(f"Wrote {results_path} and {zip_path}")

        return results
