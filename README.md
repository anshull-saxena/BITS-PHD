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

## DeepPIPE-Ctrl — RL interval-width controller

An **additive** layer (design doc `DeepPIPE_Ctrl_Design.md`) that upgrades the
static split-conformal step (E9) from *one scalar multiplier* to a *learned,
state-dependent policy*. It does **not** touch the DeepPIPE architecture or
retraining — it plugs into the E9 seam:

```python
# E9 (static):    loo, hii = pte - qq,       pte + qq
# Ctrl (adaptive): loo, hii = pte - m_t*qq,  pte + m_t*qq   # m_t = policy(state_t)
```

The controller is trained (inner objective) to minimise the **Winkler interval
score** — a strictly proper scoring rule targeting *conditional* calibration —
and evaluated (outer objective) with a width-gated backtest. P&L is never
optimised, only reported.

| File | Role | Design §|
|------|------|--------|
| `reward.py`    | Winkler interval score (inner reward) | §4.4 |
| `state.py`     | 8 fold-local exogenous state features s_t (OOD distance is the G2 centrepiece) | §4.2 |
| `dataset.py`   | `make_bandit_dataset()` — offline full reward table `{S, A, R}` | §4.1 |
| `policies.py`  | T0 `GreedyConstant` / T1 `LinUCB`,`LinTS` / T2 `MLPPolicy` | §5 |
| `baselines.py` | `static_conformal` (m=1), `conformal_pid` (hand-tuned adaptive bar) | §6 |
| `rollout.py`   | `roll_controller()` — per-step intervals + inner metrics | §7.1 |
| `backtest.py`  | `width_gated_backtest()` — Sharpe / MaxDD | §7.2 |
| `run_ctrl.py`  | driver E11–E14 → `results_ctrl.json` | §8.3 |
| `test_ctrl.py` | M1 sanity checks (closed-form interval score, reward-table parity, policy smoke) | §9 |

**Action grid:** `{0.5, 0.75, 1.0, 1.0…}` multipliers on the base half-width
(`config.CTRL_ACTIONS`). Because `ŷ`, `q̂` and realised `y` are all known after
the fact, every action's reward is observable offline — v1 is offline policy
learning against a proper scoring rule (no off-policy correction). The policy is
fit on the **validation-period** bandit table (which precedes test → no
leakage) and rolled forward on the untouched test window.

### Run

```bash
python run_ctrl.py --data /path/to/market_data.csv --out-dir outputs
```

Produces `outputs/results_ctrl.json` and prints the E11–E14 tables:

- **E11** inner calibration: T0/T1/T2 vs static-conformal vs PID
  (PICP, MPIW, interval score, Kupiec p, mean multiplier)
- **E12** VIX-width correlation (G1 check — expect the paper's −0.50 sign to
  flip toward positive)
- **E13** rolling-window calibration across folds (controller vs static)
- **E14** width-gated backtest (annualised return, vol, Sharpe, max drawdown)

### Tests

```bash
python test_ctrl.py      # or: pytest test_ctrl.py
```
