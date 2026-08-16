# DeepPIPE NIFTY 50 — Reproduction (Python package)

Converted from the single Colab notebook `DeepPIPE_Colab_AllInOne.ipynb` into a
plain Python project. Reproduces **every table and figure** in the paper
*"Restoring Interval Calibration under Distribution Shift: A Two-Phase Training
Fix for DeepPIPE on the NIFTY 50."* All numbers are computed, never hard-coded.

## Layout

| File | Notebook cell | Purpose |
|------|---------------|---------|
| `config.py`      | Cell 2 | Constants, feature sets, device, logger |
| `data.py`        | Cell 2 | CSV loading, feature engineering, sequences/splits |
| `model.py`       | Cell 2 | `DeepPIPE` network, PI loss, metrics |
| `train.py`       | Cell 3 | Training/eval loops, RNN baselines |
| `stats.py`       | Cell 3 | DM test, bootstrap CIs, Kupiec/Christoffersen, aggregation |
| `experiments.py` | Cell 4 | `run_all()` — experiments E1–E10 → `RESULTS` + JSON |
| `report.py`      | Cell 5 | `print_tables()` — all result tables |
| `figures.py`     | Cell 6 | `make_figures()` — fig1..fig6 |
| `main.py`        | Cells 1 & 7 | CLI: load CSV → run → print → figures → ZIP |

## Setup

```bash
cd DeepPIPE_NIFTY50
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --data /path/to/market_data.csv --out-dir outputs
```

Runtime is ~40–60 min on CPU (faster with a CUDA GPU). Outputs:

- `outputs/results_all.json` — every computed metric
- `outputs/figures/fig1..fig6.png`
- `outputs/deeppipe_outputs.zip` — figures + results bundled

Useful flags: `--no-figures`, `--no-zip`, `--reuse-results` (re-render
tables/figures from an existing JSON without re-training).

### Input data

`market_data.csv` needs a `Date` column plus:
`NIFTY_Open, NIFTY_High, NIFTY_Low, NIFTY_Close, NIFTY_Volume,
INDIA_VIX_Close, SP500_Close, NASDAQ_Close, DOW_Close, NIKKEI_Close,
HANGSENG_Close, BRENT_Close, GOLD_Close, USDINR_Close`.
Technical indicators (RSI, EMA20/50, MACD, etc.) are derived automatically.

## Notes (from the paper)

- `E6` (trained-alpha) intentionally shows coverage does **not** respond to the
  nominal level; `E9` (split-conformal) is the post-hoc fix that makes it
  respond. That contrast is the point.
- `fig4` may show two-phase coverage decaying late in Phase 2 — expected, and
  why early stopping is used.

## Difference from the notebook

The only behavioural change is I/O: Colab's `files.upload()` /
`files.download()` are replaced by a `--data` path and files written to
`--out-dir`. All model, training, and statistics code is a faithful port.
