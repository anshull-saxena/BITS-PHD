# Migration guide

The flat root-level scripts have been replaced by the `deeppipe` package.

| Old | New |
|-----|-----|
| `python main.py --data X --out-dir Y` | `deeppipe reproduce --data X --out-dir Y` |
| `python run_ctrl.py --data X --out-dir Y` | `deeppipe ctrl --data X --out-dir Y` |
| `python main.py --reuse-results` | Re-run with existing JSON in run dir, or `deeppipe figures results_all.json` |
| `python test_ctrl.py` | `pytest tests/` |
| `import config` | `from deeppipe.config import RunConfig, TrainConfig` |
| `from data import load_raw` | `from deeppipe.core import load_raw` |
| `from policies import fit_policy` | `from deeppipe.ctrl import fit_policy` |

Configuration defaults live in `configs/reproduce.yaml` and `configs/ctrl.yaml`.
