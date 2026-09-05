# DeepPIPE NIFTY 50

Installable Python package reproducing the DeepPIPE NIFTY 50 paper (E1–E10) and the
DeepPIPE-Ctrl adaptive calibration layer (E11–E14).

## Install

```bash
cd BITS-PHD
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Place `market_data.csv` in `data/` (see [data/README.md](data/README.md)).

## CLI

```bash
# Paper reproduction (~40–60 min on CPU with full config)
deeppipe reproduce --data data/market_data.csv --out-dir outputs

# Controller experiments
deeppipe ctrl --data data/market_data.csv --out-dir outputs

# Re-render figures from saved JSON
deeppipe figures outputs/<run>/results_all.json --out-dir figures

# Show resolved config
deeppipe config --profile reproduce
```

Each run writes a timestamped directory under `outputs/`:

```
outputs/2026-09-05_153000_reproduce/
  config.resolved.yaml
  results_all.json
  figures/fig1..fig6.png
  deeppipe_outputs.zip
  run.log
```

## Library API

```python
from deeppipe.config import RunConfig, load_run_config
from deeppipe.core import DeepPIPE, load_raw, train_deeppipe
from deeppipe.ctrl import fit_policy, roll_controller
from deeppipe.pipeline import ReproducePipeline, CtrlPipeline
```

## Development

```bash
make test    # unit + integration smoke tests
make lint    # ruff
```

## Documentation

- [docs/prior_work.md](docs/prior_work.md) — paper and notebook background
- [docs/design.md](docs/design.md) — DeepPIPE-Ctrl design
- [MIGRATION.md](MIGRATION.md) — old script → new CLI mapping

## Package layout

```
src/deeppipe/
  core/          # DeepPIPE model, training, conformal (E1–E10)
  ctrl/          # RL interval-width controller (E11–E14)
  experiments/   # run_all, run_ctrl
  pipeline/      # orchestration + artifacts
  reporting/     # tables and figures
  cli/           # Typer entry point
```
