"""Run experiments E11-E14 (DeepPIPE-Ctrl)."""

import json
import time

from deeppipe.config.schema import RunConfig
from deeppipe.core.conformal import conformal_q
from deeppipe.core.data import build_splits, load_raw, make_sequences
from deeppipe.core.train import train_deeppipe
from deeppipe.ctrl.backtest import returns_from_prices, width_gated_backtest
from deeppipe.ctrl.baselines import conformal_pid, static_conformal
from deeppipe.ctrl.dataset import make_bandit_dataset
from deeppipe.ctrl.policies import TIER_LABELS, fit_policy
from deeppipe.ctrl.rollout import eval_intervals, roll_controller
from deeppipe.logging import log


def _bandit(raw, sp, model, q, tr_end, split, ysplit, row_offset, run_cfg):
    return make_bandit_dataset(
        raw,
        sp,
        model,
        q,
        tr_end,
        split=split,
        ysplit=ysplit,
        row_offset=row_offset,
        actions=run_cfg.ctrl.actions,
        train_cfg=run_cfg.train,
        ctrl_cfg=run_cfg.ctrl,
    )


def _fit_all_tiers(D_fit, run_cfg: RunConfig):
    policies = {}
    for tier in run_cfg.ctrl.tiers:
        name = TIER_LABELS.get(tier, tier)
        try:
            policies[name] = fit_policy(D_fit, tier=tier, seed=run_cfg.ctrl.seed)
        except Exception as exc:
            log(f"  tier {name} failed to fit: {exc}")
    return policies


def _strip(metrics):
    return {k: v for k, v in metrics.items() if k != "multiplier_hist"}


def _rolling(raw, n, run_cfg: RunConfig):
    K = 5
    start = int(n * 0.55)
    fs = (n - start) // K
    folds = []
    features = run_cfg.data.features
    seq_len = run_cfg.train.seq_len
    for k in range(K):
        ts = start + k * fs
        te = ts + fs if k < K - 1 else n
        trk = int(ts * 0.85)
        vak = ts
        if trk <= seq_len + 10 or (te - ts) <= seq_len + 5:
            continue
        spk = build_splits(raw, features, trk, vak, run_cfg.train)
        tdf = raw.iloc[ts:te]
        tef = spk["scaler"].transform(tdf[features])
        Xk, yk = make_sequences(tef, spk["tgt_idx"], seq_len)
        if len(Xk) < 10:
            continue
        spk["Xte"], spk["yte"] = Xk, yk
        model = train_deeppipe(
            spk, True, train_cfg=run_cfg.train, seed=run_cfg.ctrl.seed, warmup_ep=30, full_ep=80
        )
        q, _ = conformal_q(model, spk, train_cfg=run_cfg.train)
        try:
            D_fit = _bandit(raw, spk, model, q, trk, "Xva", "yva", trk + seq_len, run_cfg)
            D_test = _bandit(raw, spk, model, q, trk, "Xte", "yte", ts + seq_len, run_cfg)
            pol = fit_policy(D_fit, tier="linucb", seed=run_cfg.ctrl.seed)
            ctrl = roll_controller(D_test, pol, p=run_cfg.train.conf)["metrics"]
            sc = static_conformal(D_test["point"], q)
            stat = eval_intervals(D_test["y"], sc["lo"], sc["hi"], q, sc["m"], p=run_cfg.train.conf)
        except Exception as exc:
            log(f"  fold {k} failed: {exc}")
            continue
        folds.append(
            dict(
                fold=k,
                test_period=[str(tdf.Date.min().date()), str(tdf.Date.max().date())],
                price_range=[float(tdf.NIFTY_Close.min()), float(tdf.NIFTY_Close.max())],
                controller_linucb=dict(
                    PICP=ctrl["PICP"],
                    MPIW=ctrl["MPIW"],
                    mean_interval_score=ctrl["mean_interval_score"],
                ),
                static_conformal=dict(
                    PICP=stat["PICP"],
                    MPIW=stat["MPIW"],
                    mean_interval_score=stat["mean_interval_score"],
                ),
            )
        )
    return dict(folds=folds)


def run_ctrl(run_cfg: RunConfig, results_path=None):
    RESULTS = {"meta": {"seed": run_cfg.ctrl.seed, "actions": list(run_cfg.ctrl.actions)}}
    t0 = time.time()
    features = run_cfg.data.features
    seq_len = run_cfg.train.seq_len

    raw = load_raw(run_cfg.data_path)
    n = len(raw)
    tr_end = int(n * run_cfg.data.train_split)
    va_end = tr_end + int(n * run_cfg.data.val_split)
    sp = build_splits(raw, features, tr_end, va_end, run_cfg.train)
    RESULTS["meta"].update(total_days=int(n), train_end=tr_end, val_end=va_end)
    log(f"{n} days | train_end={tr_end} val_end={va_end}")

    log("Ctrl: training frozen two-phase DeepPIPE")
    model = train_deeppipe(sp, True, train_cfg=run_cfg.train, seed=run_cfg.ctrl.seed)
    q, nc = conformal_q(model, sp, train_cfg=run_cfg.train)
    RESULTS["meta"].update(q_halfwidth=q, n_calibration=int(nc))
    log(f"Ctrl: base half-width q={q:.1f} from {nc} validation residuals")

    D_fit = _bandit(raw, sp, model, q, tr_end, "Xva", "yva", tr_end + seq_len, run_cfg)
    D_test = _bandit(raw, sp, model, q, tr_end, "Xte", "yte", va_end + seq_len, run_cfg)
    vix_test = sp["vix_test"]
    p_target = run_cfg.train.conf

    policies = _fit_all_tiers(D_fit, run_cfg)

    log("E11: inner calibration — tiers vs static-conformal vs PID")
    e11 = {}
    for name, pol in policies.items():
        e11[name] = _strip(roll_controller(D_test, pol, p=p_target, vix=vix_test)["metrics"])

    sc = static_conformal(D_test["point"], q)
    e11["static_conformal"] = _strip(
        eval_intervals(D_test["y"], sc["lo"], sc["hi"], q, sc["m"], p=p_target, vix=vix_test)
    )
    pid = conformal_pid(D_test["point"], D_test["y"], q, p=p_target)
    e11["conformal_pid"] = _strip(
        eval_intervals(D_test["y"], pid["lo"], pid["hi"], q, pid["m"], p=p_target, vix=vix_test)
    )
    RESULTS["E11_inner"] = e11

    log("E12: VIX-width correlation (G1 check)")
    RESULTS["E12_vix_width"] = {
        name: m.get("corr_width_vix") for name, m in e11.items() if "corr_width_vix" in m
    }

    log("E13: rolling-window calibration across folds")
    RESULTS["E13_rolling"] = _rolling(raw, n, run_cfg)

    log("E14: width-gated backtest (Sharpe / MaxDD)")
    e14 = {}
    fwd_ret = returns_from_prices(D_test["y"])
    for name, pol in policies.items():
        roll = roll_controller(D_test, pol, p=p_target, vix=vix_test)
        widths = 2.0 * roll["m"] * q
        e14[name] = width_gated_backtest(widths, fwd_ret)
    e14["static_conformal"] = width_gated_backtest(2.0 * sc["m"] * q, fwd_ret)
    e14["conformal_pid"] = width_gated_backtest(2.0 * pid["m"] * q, fwd_ret)
    RESULTS["E14_backtest"] = e14

    RESULTS["meta"]["runtime_min"] = round((time.time() - t0) / 60, 2)
    if results_path:
        with open(results_path, "w") as fh:
            json.dump(RESULTS, fh, indent=2, default=float)
        log(f'DONE in {RESULTS["meta"]["runtime_min"]} min. Saved {results_path}')
    return RESULTS
