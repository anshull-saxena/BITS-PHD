# DeepPIPE Prior Work — The Paper & The Notebook, Explained

**Status:** Reference / explainer (v1)
**Companion to:** `DeepPIPE_Ctrl_Design.md` (the RL layer this work is built on)
**Date:** 2026-08-16

This document explains the work that already exists in the repository, *before* the RL controller. It covers two artifacts:

| Artifact | File | What it is |
|---|---|---|
| The research paper | `DeepPipe_V1.pdf` | 29-page paper (July 2026). The motivation, the fix, the results, and the residual gaps. |
| The reproduction notebook | `DeepPIPE_Colab_AllInOne.ipynb` | A single self-contained Colab notebook that reproduces **every** table and figure in the paper. |

The RL framework in `DeepPIPE_Ctrl_Design.md` is an *additive layer* on top of the notebook — this document is the "prior work" that layer attaches to.

---

## 1. The paper (`DeepPipe_V1.pdf`)

**Title:** *Restoring Interval Calibration under Distribution Shift: A Two-Phase Training Fix for DeepPIPE on the NIFTY 50*
**Author:** Raunak Goyal, under Prof. Jabez Christopher and Prof. Vasan Arunachalam (CSE dept), July 2026.

### 1.1 The problem it tackles

- **Task:** next-day **prediction intervals** (PIs) for the NIFTY 50 index — not just a point forecast, but a lower/upper bound that should contain the true close with a specified confidence.
- **Model under test:** **DeepPIPE** (Wang et al. 2020), a *distribution-free* encoder–decoder LSTM with **three output heads** (point, lower bound, upper bound), trained with a **hybrid loss** = point MAE + interval width + coverage penalty.
- **The setting that breaks it:** **distribution shift.** The training data tops out at NIFTY **24,613**; most of the test period lies **above** that, up to **26,329**. The model must forecast price levels it never saw in training.

### 1.2 The failure (single-phase training)

The original DeepPIPE trains all three heads at once with the hybrid loss. Under the OOD regime:

| Metric | Result |
|---|---|
| Mean PICP (5 seeds) | **0.32 ± 0.27** |
| Worst seed | **0.00** (zero coverage — no test point inside the interval) |

The intervals don't literally shrink to zero width (they stay a few hundred points wide). The problem is subtler: they are **simultaneously narrow and badly centered**, because the loss prefers minimizing width over maintaining coverage once the point forecast drifts from the truth. This is an **optimization failure**, not an architecture one.

### 1.3 The fix (two-phase training)

The paper's contribution is a training-procedure change — **no architecture change, no new parameters**:

1. **Phase 1 — point-only warm-up.** Train only the point head with plain MAE (the interval heads are dormant).
2. **Phase 2 — full hybrid loss.** Activate the interval heads and train with the full loss (point + width + coverage penalty), with early stopping.

Result:

| Metric | Single-phase | Two-phase |
|---|---|---|
| Mean PICP (5 seeds) | 0.32 ± 0.27 | **0.99 ± 0.01** |
| Point error (MAE) | — | also **improves** (not just coverage) |

That's a **+0.67 absolute** coverage gain, consistent across all five seeds.

### 1.4 What the ablations found

- **`β` (point-prediction weight) matters far more than `λ` (coverage penalty).** Sweeping `β` from 0.1→0.7 moves PICP 0.703→0.995→(down); sweeping `λ` across 1/10/15/50 leaves PICP stuck at ~0.995. The point head is the actual calibration lever.
- **Smaller feature sets generalize better under shift.** The 14-feature "core" set (NIFTY OHLCV + VIX + global indices + commodity/FX) beats the 21-feature set that adds technical indicators (RSI, MACD, EMA, etc.).

### 1.5 Statistical validation

- **Diebold–Mariano** test + **bootstrap confidence intervals** confirm the two-phase point forecast is significantly better than Naive / Moving-Average / Linear Regression / LSTM / GRU baselines.
- **Kupiec POF** and **Christoffersen independence** tests assess calibration of the interval coverage.

### 1.6 The three residual gaps (the paper's own honest conclusion)

The paper ends by admitting what the fix does *not* solve. These are the three gaps the RL layer targets (see `DeepPIPE_Ctrl_Design.md` §1):

| # | Gap | Evidence in the paper |
|---|---|---|
| **G1** | Intervals are **volatility-blind** | width↔VIX correlation = **−0.50** (wrong sign); width ~flat across low/mid/high VIX buckets (~3861/3838/3824) |
| **G2** | Coverage **collapses under regime breaks** | rolling-window Fold 1 (price 21.8k→26.0k) → PICP = **0.01** |
| **G3** | **Over-coverage**, unresponsive to requested level | realized coverage ≈ **0.995** whether trained for 80/90/95% |

### 1.7 The paper's own future-work prescription

The conclusion explicitly calls for: *"integrating the proposed approach with adaptive uncertainty estimation techniques, such as conformal prediction and online calibration methods"*, plus *"volatility-aware learning mechanisms"*. The RL controller is a direct, named continuation of that prescription.

**The key framing to hold onto:** the paper shows a **training-time fix** (two-phase) and a **calibration-time fix** (split-conformal) are *complementary*. Two-phase stops the collapse; split-conformal makes coverage track the target. The RL controller upgrades the second of these from a static scalar to a state-dependent policy.

---

## 2. The notebook (`DeepPIPE_Colab_AllInOne.ipynb`)

A **single-file, self-contained Colab notebook** that reproduces every number and figure in the paper — "all numbers computed, never hard-coded." It is the codebase the RL layer integrates into.

### 2.1 How to run

1. Run **Cell 1**, upload `market_data.csv` when prompted.
2. **Runtime → Run all** (~40–60 min on Colab CPU, faster on GPU).
3. Tables print inline; figures render inline; `results_all.json` + a ZIP of figures download at the end.

### 2.2 Cell-by-cell map

| Cell | Content |
|---|---|
| 0–1 | Markdown intro + data upload |
| 2 | **Core machinery** — imports, config, feature sets, `load_raw`, `TSData`, `make_sequences`, `build_splits`, `inv_target`, `DeepPIPE`, `pipe_loss`, `set_seed`, `_y`, `point_metrics`, `interval_metrics` |
| 3 | Training / eval / baselines / statistics — `train_deeppipe`, `eval_deeppipe`, `SimpleRNN`, `train_rnn`, `run_baselines`, `diebold_mariano`, bootstrap CIs, `kupiec_pof`, `christoffersen_ind`, `agg` |
| 4 | **Runs all experiments E1–E10** and dumps `results_all.json` |
| 5 | Prints all tables |
| 6 | Generates figures fig1–fig6 |
| 7 | Downloads results + figures ZIP |

### 2.3 The experiment blocks (E1–E10)

| Block | Produces (paper item) |
|---|---|
| **E1** | Main comparison table — DeepPIPE (two-phase) vs Naive / MovingAvg5 / LinearReg / LSTM / GRU |
| **E2** | Failure vs fix over 5 seeds, + `λ` and `β` ablations |
| **E3** | Feature-set ablation (nifty_only / plus_global / core14 / full21) |
| **E4** | Multi-seed robustness (5 seeds, two-phase) |
| **E5** | **Rolling-window (walk-forward), 5 folds** — the distribution-shift stress test |
| **E6** | Calibration (Kupiec / Christoffersen) at trained-alpha 80/90/95% |
| **E7** | Diebold–Mariano (MAE-consistent) + bootstrap CIs |
| **E8** | VIX adaptiveness (width↔VIX correlation + buckets) |
| **E9** | **Split-conformal calibration** — the integration seam for the RL controller |
| **E10** | Training dynamics + warm-up sweep |

### 2.4 Config & key symbols (reused by the RL layer, *not* redefined)

```python
TARGET='NIFTY_Close'; SEQ_LEN=60; HIDDEN=64; LAYERS=2; DROPOUT=0.2
BATCH=32; LR=5e-4; WARMUP_EP=40; FULL_EP=120
BETA=0.5; CONF=0.90; LAMBDA=15.0; TRAIN_SPLIT=0.70; VAL_SPLIT=0.15

FEATURE_SETS = {
    'nifty_only': NIFTY OHLCV,
    'plus_global': NIFTY + VIX + global indices,
    'core14':     NIFTY + VIX + global + commodity/FX,   # MAIN_FEATURES
    'full21':     core14 + technical indicators,
}
MAIN_FEATURES = FEATURE_SETS['core14']   # 14 features
```

Functions: `load_raw`, `TSData`, `make_sequences`, `build_splits`, `inv_target`, `DeepPIPE` (the 3-head encoder–decoder LSTM), `pipe_loss`, `set_seed`, `_y`, `point_metrics`, `interval_metrics`, `train_deeppipe`, `eval_deeppipe`, `run_baselines`, `diebold_mariano`, `bootstrap_mae_ci`, `bootstrap_diff_ci`, `kupiec_pof`, `christoffersen_ind`, `agg`, `eval_epoch`, `instrumented`.

### 2.5 The integration seam — E9 (split-conformal)

This is where the RL controller lives. Today it's a static, one-global-multiplier step:

```python
# E9 today:
for tau in (0.80, 0.90, 0.95):
    kk = min(nc, int(np.ceil((nc+1)*tau)))
    qq = float(np.sort(cal)[kk-1])          # fold-local val-residual quantile → half-width
    loo, hii = pte - qq, pte + qq           # static: one qq for the whole test set
```

The controller replaces `loo/hii = pte ∓ qq` with `loo_t/hii_t = pte_t ∓ m_t·qq` — a *learned, per-step* multiplier. The full spec is in `DeepPIPE_Ctrl_Design.md` §8.

### 2.6 The data (`market_data.csv`)

~1,676 trading days (2020–2026) across ~1,690 rows, 50 columns. Feeds into the four feature sets via `load_raw()` which also derives technical indicators (RSI, EMA20/50, MACD, RETURN, VOLATILITY). The paper's headline shift: train price range up to **24,613**, test up to **26,329**.

---

## 3. How the three artifacts fit together

```
market_data.csv  ──►  DeepPIPE_Colab_AllInOne.ipynb  ──►  results_all.json + fig1..fig6
                              │
                     DeepPipe_V1.pdf   (motivation + three residual gaps G1–G3)
                              │
                              └──►  DeepPIPE_Ctrl_Design.md  (RL layer → adaptive width controller)
```

- **The paper** supplies the *why* (three quantitative gaps) and the *where* (the split-conformal seam, the rolling-window harness).
- **The notebook** supplies the *what* (frozen two-phase DeepPIPE forecaster, fold-local `q̂_α`, metrics, walk-forward eval).
- **The RL design** upgrades the notebook's one-line static `qq` into a state-dependent multiplier policy, trained on a proper scoring rule, evaluated with the same metrics the paper already uses.
