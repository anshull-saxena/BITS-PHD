# DeepPIPE-Ctrl — RL Framework for Adaptive Prediction-Interval Calibration

**Status:** Design + implementation plan (v1)
**Author:** — (project doc)
**Date:** 2026-08-16
**Scope:** Additive RL layer on top of the existing two-phase DeepPIPE pipeline (`DeepPIPE_Colab_AllInOne.ipynb`). No changes to the DeepPIPE architecture or retraining.

---

## 1. Why this exists

The paper (`DeepPipe_V1.pdf`) ends by naming three *quantitative* gaps in the two-phase DeepPIPE result:

| # | Gap (from paper) | Evidence in the paper |
|---|---|---|
| G1 | Intervals are **volatility-blind** | width↔VIX correlation = **−0.50** (wrong sign); width nearly flat across low/mid/high VIX buckets (~3861/3838/3824) |
| G2 | Coverage **collapses under regime breaks** | rolling-window Fold 1 (21.8k→26.0k) → PICP = **0.01** |
| G3 | **Over-coverage**, unresponsive to requested level | realized coverage ≈ 0.995 whether trained for 80/90/95% |

The paper's own "future work" prescribes *adaptive calibration, conformal/online calibration, volatility-aware learning*. Their split-conformal step (E9 in the notebook) fixes G3 only partially and *statically* — one global multiplier for the whole test set.

**DeepPIPE-Ctrl turns that static multiplier into a learned, state-dependent policy.** It is the split-conformal step upgraded from "one scalar" to "one policy," and it directly targets G1–G3.

### 1.1 Positioning (what it is and isn't)

- **Is:** an *online interval-width controller* — at each time step it observes market state and rescales the prediction interval around the (already trained) DeepPIPE point forecast.
- **Is not:** a new forecaster, a trading agent (the trading layer is a *downstream evaluation*, not the RL objective), or a training meta-controller.

---

## 2. Layered objective

The framework has two objectives, deliberately separated:

- **Inner (train-time) objective — statistical calibration.** The RL policy is *trained* to minimize the expected **Winkler interval score** (a strictly proper scoring rule). This is the learning signal.
- **Outer (eval-time) objective — economic utility.** The *calibrated* intervals feed a transparent position-sizing backtest; we then show better calibration → better risk-adjusted returns (Sharpe / max drawdown).

The causal claim is clean: **we never optimize P&L directly.** We optimize calibration; economic value is a consequence we *report*, not a reward we chase. This keeps the framework a direct extension of the paper (evaluable with the same PICP/MPIW/Kupiec/Christoffersen metrics) while still answering the "so what?" question.

---

## 3. Architecture

```
                  (trained, frozen)
 DeepPIPE two-phase point predictor  ──►  ŷ_{t+1}   (from train_deeppipe / eval_deeppipe)
                                              │
 base conformal half-width  q̂_α  (fold-local validation-residual quantile, E9 seam)
                                              │
              ┌───────────────────────────────┴──────────────────────────────┐
              ▼                                                              ▼
      state  s_t  (exogenous market state)                         base interval  [ŷ − q̂_α, ŷ + q̂_α]
              │                                                              │
              ▼                                                              │
        RL controller  π(a_t | s_t)  ──►  multiplier  m_t ∈ {0.5, 0.75, 1.0, 1.25, 1.5}
              │                                                              │
              └───────────────────────────────┬──────────────────────────────┘
                                              ▼
                       adaptive interval  [ŷ − m_t·q̂_α, ŷ + m_t·q̂_α]
                                              │
                 ┌────────────────────────────┴────────────────────────────┐
                 ▼                                                        ▼
        (inner) Winkler interval score  r_t                     (outer) width-gated backtest
```

The controller replaces one line of the notebook's E9 code:

```python
# E9 (static):  loo = pte - qq ;  hii = pte + qq
# Ctrl (adaptive): per-step  loo_t = pte_t - m_t*qq ;  hii_t = pte_t + m_t*qq
```

---

## 4. RL formulation

### 4.1 Why this is a bandit, not a full MDP

The action `m_t` only rescales the interval width; it does **not** affect the market's evolution, the point forecast, or the next state. State transitions are driven by exogenous market dynamics, independent of the controller's action. Therefore each step is an independent contextual decision:

**Contextual bandit** with context `s_t`, action `a_t`, reward `r_t = −interval_score(s_t, a_t)`.

Crucially, because `ŷ_t`, `q̂_α`, and the realized `y_{t+1}` are all known after the fact, **the reward of every action is fully observable offline** — no counterfactual / importance-weighted off-policy estimation is needed for v1. The bandit dataset is a *complete reward table* `R ∈ R^{N × |A|}`.

This is the single most important de-risking property of the design: v1 reduces to *offline supervised policy learning against a proper scoring rule*, with the RL framing adding (a) the sequential/regret semantics and (b) a clean upgrade path to true online adaptation in v2.

### 4.2 State space  s_t ∈ ℝ^d

All features are **exogenous w.r.t. the controller's action** (no "recent coverage error" feedback — see §7.4 for why and when to relax this). All z-scores/statistics are computed **fold-locally from training data only** (no leakage), consistent with the notebook's `MinMaxScaler` discipline.

| # | Feature | Formula (fold-local) | Targets |
|---|---|---|---|
| 1 | VIX level | `(VIX_t − μ_train)/σ_train` | G1 volatility signal |
| 2 | VIX daily change | `(ΔVIX_t − μ)/σ` | G1 |
| 3 | Realized vol 5d | `std(returns_{t−4:t})` | G1 |
| 4 | Realized vol 20d | `std(returns_{t−19:t})` | G1 |
| 5 | Momentum 5d | `Σ returns_{t−4:t}` | regime |
| 6 | Momentum 20d | `Σ returns_{t−19:t}` | regime |
| 7 | **OOD distance** | `(Close_t − max_train_Close)/max_train_Close` | G2 — the exact collapse driver the paper identifies |
| 8 | Rolling \|residual\| | trailing mean of `|y − ŷ|` | expected error magnitude |

Feature 7 is the design's centerpiece: it is a normalized, live flag for "are we beyond the training price range?", which the paper showed is the regime where coverage collapses.

### 4.3 Action space

```
A = {0.5, 0.75, 1.0, 1.25, 1.5}      # multiplier on the base conformal half-width q̂_α
```

Discrete and small is a deliberate choice: (i) ~190 out-of-sample points cannot support continuous control; (ii) the grid is interpretable ("today is a 1.25× day"); (iii) it maps cleanly onto LinUCB/Thompson sampling as well as a softmax policy.

### 4.4 Reward (inner) — the Winkler interval score

For target coverage `p` (notebook `CONF`, default `0.90`) and tail `α = 1 − p`:

```
S(L, U, y) = (U − L)  +  (2/α)(L − y)·𝟙[y < L]  +  (2/α)(y − U)·𝟙[y > U]

r_t = − S(L_t, U_t, y_{t+1})        # RL maximizes reward ⇒ minimizes interval score
```

**Why this and not a hand-rolled "coverage − width" proxy:** `S` is a *strictly proper scoring rule* (Gneiting & Raftery 2007). Its expectation over `y` is minimized *uniquely* by the interval whose bounds are the `α/2` and `1−α/2` quantiles of the conditional distribution — i.e. the correct `p`-level interval with minimal width. Maximizing `−S` therefore targets **conditional** calibration (the right interval *at each state*), which is strictly stronger than the paper's *unconditional* PICP. This is exactly the property G3 is missing.

**Scale note:** widths are in index points (thousands), so raw rewards are large negative. Normalize rewards by `q̂_α` (report in "units of base half-width") for numerical stability; monotone scaling does not change the argmax.

---

## 5. Policy tiers (v1)

Three tiers, increasing in capacity, all fit **offline** on the full reward table.

- **T0 — Greedy constant.** Learn the single best multiplier `m* = argmin_a Σ_t S(...)` on training data. Trivially answers: *"is a constant ≠ 1.0 already better than static conformal?"* (If yes, the paper's static conformal was suboptimal even without adaptivity.)
- **T1 — Linear bandit (recommended floor).** `LinUCB` / `LinTS` over handcrafted `φ(s)` (the §4.2 features, possibly with `s × a` interactions). Sample-efficient, interpretable, closed-form updates.
- **T2 — MLP policy.** `π_θ(a | s)` (softmax over A) trained either by reward-regression + argmax (direct method — exact here) or REINFORCE on the bandit dataset. Capacity to learn nonlinear state→multiplier maps (e.g., "wide only when OOD distance > 0 AND realized vol is high").

**v2 (offline / online RL):** escalate to a short-horizon MDP with a coverage-error memory state, using `CQL` (offline) or online nonstationary bandit updates, when §7.4's exogenous-state limit binds. Also supports asymmetric intervals (separate lower/upper multipliers) and ACI-style level selection.

---

## 6. Baselines the controller must beat

A learned controller is only interesting if it clears the *cheap, non-learned* bar. All baselines share the same `ŷ` and `q̂_α`:

1. **Static split-conformal** — the paper's E9 (`m_t = 1` always). This is the thing being upgraded.
2. **Fixed hybrid interval** — raw DeepPIPE output (E1).
3. **Conformal PID** (Angelopoulos et al. 2023; cf. Gibbs & Candès 2021) — the hand-tuned *adaptive* baseline:

```text
e_t     = p_target − EMA_coverage_t          # coverage error (trailing EMA of hit indicator)
m_{t+1} = clip( m_t + k_p·e_t + k_i·Σe + k_d·Δe_t , 0.5, 1.5 )
```

If T2 can't beat the PID, the honest conclusion is "a hand-tuned controller suffices" — and we report it as such (§8.2 commits to honest reporting).

---

## 7. Evaluation protocol

Reuse the notebook's **walk-forward harness** (E5): 5 folds, expanding window.

Per fold:
1. Train two-phase DeepPIPE on the fold's train split (`train_deeppipe(..., two_phase=True)`).
2. Compute base `q̂_α` from the fold's *validation* residuals (split-conformal, same as E9 — fold-local, no leakage).
3. Build the **bandit dataset** for the fold's *training* period (state + all-action reward table).
4. Fit the policy (T0/T1/T2) on that fold's bandit data.
5. Roll forward over the fold's *test* period: observe `s_t`, pick `a_t`, form the interval, record metrics.
6. Run baselines (static conformal, PID) on the same fold.

### 7.1 Inner metrics (calibration)

- `PICP`, `MPIW` (reuse `interval_metrics`)
- mean interval score (normalized)
- `kupiec_pof`, `christoffersen_ind` (reuse existing functions)
- **VIX-width correlation** (paper's E8 metric) — we expect the sign to flip from −0.50 toward positive, directly evidencing G1 fix

### 7.2 Outer metrics (economic)

**Width-gated position sizing** (default flavor). Daily, long-only, causal:

```text
w_t     = 2 · m_t · q̂_α                      # today's interval width
p_t     = clip( w_ref / w_t , 0, 1 )         # narrow ⇒ confident ⇒ larger position
r_{t+1} = p_t · (NIFTY daily return)         # tomorrow's return
```

`w_ref` is the trailing median width. Report **annualized return, vol, Sharpe (rf=0), max drawdown**, per-fold and aggregate, for each controller tier and baseline. The headline comparison: *does the adaptive interval's strategy beat the static-conformal interval's strategy on Sharpe/MaxDD?*

Optional second flavor (documented, not built first): use the lower/upper bounds as **stop / take-profit** levels.

---

## 8. Integration with the existing notebook

### 8.1 The exact seam

The controller lives where E9 currently hardcodes one `qq`:

```python
# E9 today:
for tau in (0.80,0.90,0.95):
    kk = min(nc, int(np.ceil((nc+1)*tau)))
    qq = float(np.sort(cal)[kk-1])
    loo, hii = pte - qq, pte + qq      # static
```

becomes, in the controller rollout,

```python
m = policy(state_t)                     # learned multiplier
loo_t, hii_t = pte_t - m*qq, pte_t + m*qq
```

### 8.2 Reused symbols (do not redefine)

`load_raw`, `build_splits`, `make_sequences`, `train_deeppipe`, `eval_deeppipe`, `inv_target`, `interval_metrics`, `point_metrics`, `kupiec_pof`, `christoffersen_ind`, `set_seed`, `_y`, `TSData`, `DeepPIPE`, and config `CONF`, `SEQ_LEN`, `TRAIN_SPLIT`, `VAL_SPLIT`, `MAIN_FEATURES`.

### 8.3 New components (function signatures)

```python
# state.py
def build_controller_state(raw, split, train_stats, point_preds, t) -> np.ndarray   # §4.2 features

# reward.py
def interval_score(y, lo, hi, p=CONF) -> float          # Winkler score, §4.4
def interval_score_norm(y, lo, hi, q, p=CONF) -> float  # / q̂ for stability

# dataset.py
def make_bandit_dataset(sp, model, q, alpha, p=CONF) -> dict
    # returns {'S': (N,d), 'A': list, 'R': (N, |A|)}  full reward table

# policies.py
class GreedyConstant: ...                               # T0
class LinUCB: ...                                       # T1
class LinTS: ...                                        # T1
class MLPPolicy(nn.Module): ...                         # T2
def fit_policy(D, tier='linucb') -> policy
def act(policy, s_t) -> float                           # -> multiplier m_t

# baselines.py
def static_conformal(...)                               # m_t = 1
def conformal_pid(...)                                  # §6.3 PID

# rollout.py
def roll_controller(sp, model, policy, q, p=CONF) -> dict   # per-step intervals + inner metrics

# backtest.py
def width_gated_backtest(widths, returns, w_ref) -> dict    # outer metrics

# run.py  (E11–E14)
#   E11: T0/T1/T2 vs static-conformal vs PID  (inner metrics)
#   E12: VIX-width correlation  (G1 check)
#   E13: rolling-window calibration across folds
#   E14: width-gated outer backtest (Sharpe/MaxDD)
```

### 8.4 File layout

Two options; recommendation is **(A) for v1** to match the existing single-file Colab workflow.

```
(A) extend notebook          (B) modular (cleaner long-term)
├── notebook + new cells      ├── core.py        (extract Cell 4+6 machinery)
│   "RL controller"            ├── deeppipe_ctrl/{state,reward,dataset,policies,baselines,rollout,backtest}.py
│   "E11–E14"                  └── run.py + deeppipe_ctrl.ipynb (thin driver)
```

`core.py` extraction is a mechanical copy of the notebook's Cell 4 + Cell 6; both the notebook and the RL package then import from it. Do this only when (B) is chosen.

---

## 9. Implementation milestones

- **M1 — Foundations (offline, no learning yet).** Extract core (if B); implement `state.py`, `reward.py`, `dataset.py`; unit-check `interval_score` against a known closed-form case (e.g. a Gaussian where the true 90% interval is analytically known) and `make_bandit_dataset` against E9's static `qq`.
- **M2 — v1 policies + inner eval.** T0/T1/T2; walk-forward roll; inner metrics vs static-conformal and PID. This is the scientific core — G1–G3 verdict.
- **M3 — Outer backtest.** `width_gated_backtest`; Sharpe/MaxDD comparison across tiers/baselines.
- **M4 — v2 (optional).** Coverage-error memory state + offline RL (CQL) / online nonstationary bandit; asymmetric lower/upper multipliers.
- **M5 — Reproducibility.** Seed discipline (`set_seed`), results → `results_ctrl.json`, figure parity with notebook Cell 6.

---

## 10. Risks, limits, open questions

1. **Data volume** — ~1,113 train / 191 val / 192 test points. The full reward table mitigates this (no off-policy noise), but the policy still sees few samples. Mitigation: low-dim handcrafted `φ(s)`, strong regularization, `LinUCB` as the floor.
2. **Does it beat PID?** Genuinely open. Pre-committed to reporting a negative result if that's what falls out (matches the paper's honest-reporting ethos).
3. **Leakage** — `q̂_α` and feature z-scores must be fold-local. Enforced by construction in §7.
4. **Exogenous-state limit** — dropping "recent coverage error" from `s_t` keeps the bandit clean but removes the agent's memory of its own misses. If G2 needs that memory, escalate to v2 MDP. Flagged now, not discovered later.
5. **Point-forecast bias is out of scope** — the controller rescales *width* around a possibly-biased `ŷ` (the paper's Figure 1 lag). It improves **calibration, not location**. This is a stated, intentional limit.

---

## 11. References

- Wang et al. (2020), *DeepPIPE* — Neurocomputing 397.
- Gneiting & Raftery (2007), *Strictly proper scoring rules, prediction, and estimation* — JASA (proper scoring rule property of the interval score).
- Gibbs & Candès (2021), *Adaptive Conformal Inference* — NeurIPS (ACI / online level adjustment).
- Angelopoulos, Candès & Tibshirani (2023), *Conformal PID control for time series prediction* (PID baseline).
- Romano, Patterson & Candès (2019), *Conformalized Quantile Regression* (split-conformal seam).
