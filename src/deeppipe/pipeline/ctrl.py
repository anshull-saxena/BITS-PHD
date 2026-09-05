"""Controller experiments pipeline (E11-E14)."""

from __future__ import annotations

from deeppipe.config.schema import RunConfig
from deeppipe.experiments.ctrl import run_ctrl
from deeppipe.logging import setup_logging
from deeppipe.pipeline.artifacts import load_results_json, prepare_run_dir
from deeppipe.reporting.tables import print_ctrl_tables


class CtrlPipeline:
    def __init__(self, run_cfg: RunConfig):
        self.run_cfg = run_cfg

    def run(self, reuse_results: bool = False) -> dict:
        run_dir = prepare_run_dir(self.run_cfg, "ctrl")
        setup_logging(run_dir)
        results_path = run_dir / "results_ctrl.json"

        if reuse_results and results_path.is_file():
            results = load_results_json(results_path)
        else:
            if not self.run_cfg.data_path.is_file():
                raise FileNotFoundError(f"Data file not found: {self.run_cfg.data_path}")
            results = run_ctrl(self.run_cfg, results_path=results_path)

        print_ctrl_tables(results, conf=self.run_cfg.train.conf)
        print(f"Wrote {results_path}")
        return results
