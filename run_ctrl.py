"""DeepPIPE-Ctrl driver — experiments E11-E14 (design doc §8.3).

Additive RL layer on top of the frozen two-phase DeepPIPE point predictor. It
reuses the notebook/package machinery (load_raw, build_splits, train_deeppipe,
eval_deeppipe, interval_metrics, kupiec_pof, christoffersen_ind, set_seed,
config.*) and plugs a learned interval-width policy into the E9 conformal seam.

    E11  T0/T1/T2 vs static-conformal vs PID   (inner calibration metrics)
    E12  VIX-width correlation                 (G1 check: sign flips toward +)
    E13  rolling-window calibration across folds
    E14  width-gated outer backtest            (Sharpe / MaxDD)

Fitting protocol (no leakage): the policy is fit on the *validation-period*
bandit table (states + full reward table, all observable offline) which
precedes the test window; q_hat is the fold-local validation-residual quantile
(exactly E9); the policy then rolls forward over the untouched test window.
"""
import json
import time

import numpy as np

import config
from config import (MAIN_FEATURES, TRAIN_SPLIT, VAL_SPLIT, SEQ_LEN,
                    CTRL_ACTIONS, log)
from data import load_raw, build_splits, make_sequences
from train import train_deeppipe
from dataset import make_bandit_dataset, conformal_q
from policies import fit_policy
from rollout import roll_controller, eval_intervals
from baselines import static_conformal, conformal_pid
from backtest import width_gated_backtest, returns_from_prices

TIERS = [("T0_greedy", "greedy"), ("T1_linucb", "linucb"),
         ("T1_lints", "lints"), ("T2_mlp", "mlp")]


def _bandit(raw, sp, model, q, tr_end, split, ysplit, row_offset):
    return make_bandit_dataset(raw, sp, model, q, tr_end,
                               split=split, ysplit=ysplit,
                               row_offset=row_offset, actions=CTRL_ACTIONS)


def _fit_all_tiers(D_fit):
    policies = {}
    for name, tier in TIERS:
        try:
            policies[name] = fit_policy(D_fit, tier=tier)
        except Exception as exc:  # keep the run going; record the failure
            log(f"  tier {name} failed to fit: {exc}")
    return policies


def _strip(metrics):
    """Drop bulky per-step histograms for the top-level comparison table."""
    return {k: v for k, v in metrics.items() if k != "multiplier_hist"}


def run_ctrl(data_path, results_path="results_ctrl.json", seed=42):
    RESULTS = {"meta": {"seed": seed, "actions": list(CTRL_ACTIONS)}}
    t0 = time.time()

    raw = load_raw(data_path)
    n = len(raw)
    tr_end = int(n * TRAIN_SPLIT)
    va_end = tr_end + int(n * VAL_SPLIT)
    sp = build_splits(raw, MAIN_FEATURES, tr_end, va_end)
    RESULTS["meta"].update(total_days=int(n), train_end=tr_end, val_end=va_end)
    log(f"{n} days | train_end={tr_end} val_end={va_end}")

    # ---- frozen point predictor + base conformal half-width (E9 seam) ----
    log("Ctrl: training frozen two-phase DeepPIPE")
    model = train_deeppipe(sp, True, seed=seed)
    q, nc = conformal_q(model, sp)
    RESULTS["meta"].update(q_halfwidth=q, n_calibration=int(nc))
    log(f"Ctrl: base half-width q={q:.1f} from {nc} validation residuals")

    # ---- bandit datasets: fit on validation window, roll on test window ----
    D_fit = _bandit(raw, sp, model, q, tr_end, "Xva", "yva", tr_end + SEQ_LEN)
    D_test = _bandit(raw, sp, model, q, tr_end, "Xte", "yte", va_end + SEQ_LEN)
    vix_test = sp["vix_test"]

    policies = _fit_all_tiers(D_fit)

    # ================= E11: inner metrics — tiers vs baselines ==============
    log("E11: inner calibration — tiers vs static-conformal vs PID")
    e11 = {}
    for name, pol in policies.items():
        e11[name] = _strip(roll_controller(D_test, pol, vix=vix_test)["metrics"])

    sc = static_conformal(D_test["point"], q)
    e11["static_conformal"] = _strip(eval_intervals(
        D_test["y"], sc["lo"], sc["hi"], q, sc["m"], vix=vix_test))
    pid = conformal_pid(D_test["point"], D_test["y"], q)
    e11["conformal_pid"] = _strip(eval_intervals(
        D_test["y"], pid["lo"], pid["hi"], q, pid["m"], vix=vix_test))
    RESULTS["E11_inner"] = e11

    # ================= E12: VIX-width correlation (G1) ======================
    log("E12: VIX-width correlation (G1 check)")
    RESULTS["E12_vix_width"] = {
        name: m.get("corr_width_vix") for name, m in e11.items()
        if "corr_width_vix" in m
    }

    # ================= E13: rolling-window calibration ======================
    log("E13: rolling-window calibration across folds")
    RESULTS["E13_rolling"] = _rolling(raw, n, seed)

    # ================= E14: width-gated outer backtest ======================
    log("E14: width-gated backtest (Sharpe / MaxDD)")
    e14 = {}
    fwd_ret = returns_from_prices(D_test["y"])
    for name, pol in policies.items():
        roll = roll_controller(D_test, pol, vix=vix_test)
        widths = 2.0 * roll["m"] * q
        e14[name] = width_gated_backtest(widths, fwd_ret)
    e14["static_conformal"] = width_gated_backtest(2.0 * sc["m"] * q, fwd_ret)
    e14["conformal_pid"] = width_gated_backtest(2.0 * pid["m"] * q, fwd_ret)
    RESULTS["E14_backtest"] = e14

    RESULTS["meta"]["runtime_min"] = round((time.time() - t0) / 60, 2)
    json.dump(RESULTS, open(results_path, "w"), indent=2, default=float)
    log(f'DONE in {RESULTS["meta"]["runtime_min"]} min. Saved {results_path}')
    return RESULTS


def _rolling(raw, n, seed):
    """Walk-forward: fit on each fold's val window, roll on its test window."""
    K = 5
    start = int(n * 0.55)
    fs = (n - start) // K
    folds = []
    for k in range(K):
        ts = start + k * fs
        te = ts + fs if k < K - 1 else n
        trk = int(ts * 0.85)
        vak = ts
        if trk <= SEQ_LEN + 10 or (te - ts) <= SEQ_LEN + 5:
            continue
        spk = build_splits(raw, MAIN_FEATURES, trk, vak)
        # custom test window for this fold
        tdf = raw.iloc[ts:te]
        tef = spk["scaler"].transform(tdf[MAIN_FEATURES])
        Xk, yk = make_sequences(tef, spk["tgt_idx"], SEQ_LEN)
        if len(Xk) < 10:
            continue
        spk["Xte"], spk["yte"] = Xk, yk
        model = train_deeppipe(spk, True, seed=seed, warmup_ep=30, full_ep=80)
        q, _ = conformal_q(model, spk)
        try:
            D_fit = _bandit(raw, spk, model, q, trk, "Xva", "yva", trk + SEQ_LEN)
            D_test = _bandit(raw, spk, model, q, trk, "Xte", "yte", ts + SEQ_LEN)
            pol = fit_policy(D_fit, tier="linucb")
            ctrl = roll_controller(D_test, pol)["metrics"]
            sc = static_conformal(D_test["point"], q)
            stat = eval_intervals(D_test["y"], sc["lo"], sc["hi"], q, sc["m"])
        except Exception as exc:
            log(f"  fold {k} failed: {exc}")
            continue
        folds.append(dict(
            fold=k,
            test_period=[str(tdf.Date.min().date()), str(tdf.Date.max().date())],
            price_range=[float(tdf.NIFTY_Close.min()), float(tdf.NIFTY_Close.max())],
            controller_linucb=dict(PICP=ctrl["PICP"], MPIW=ctrl["MPIW"],
                                   mean_interval_score=ctrl["mean_interval_score"]),
            static_conformal=dict(PICP=stat["PICP"], MPIW=stat["MPIW"],
                                  mean_interval_score=stat["mean_interval_score"]),
        ))
    return dict(folds=folds)


def print_ctrl_tables(R):
    """Compact E11-E14 report (design doc §7)."""
    print("=" * 74)
    print("DeepPIPE-Ctrl  |  base half-width q =", round(R["meta"].get("q_halfwidth", 0), 1))
    print("=" * 74)

    print("\nE11 — INNER CALIBRATION (target p = %.2f)" % config.CONF)
    print(f"{'policy':<20}{'PICP':>8}{'MPIW':>9}{'IntScore':>10}{'Kupiec_p':>10}{'meanM':>8}")
    for name, m in R["E11_inner"].items():
        print(f"{name:<20}{m['PICP']:>8.3f}{m['MPIW']:>9.0f}"
              f"{m['mean_interval_score']:>10.3f}{m['kupiec']['p']:>10.3f}"
              f"{m.get('mean_multiplier', float('nan')):>8.2f}")

    print("\nE12 — VIX-WIDTH CORRELATION (G1: expect sign flip toward +)")
    for name, c in R.get("E12_vix_width", {}).items():
        print(f"  {name:<20} corr(width,VIX) = {c:+.3f}")

    print("\nE13 — ROLLING-WINDOW (controller LinUCB vs static conformal)")
    for f in R.get("E13_rolling", {}).get("folds", []):
        c, s = f["controller_linucb"], f["static_conformal"]
        print(f"  fold{f['fold']} {f['test_period'][0]}..{f['test_period'][1]}"
              f"  ctrl PICP={c['PICP']:.3f} S={c['mean_interval_score']:.2f}"
              f" | static PICP={s['PICP']:.3f} S={s['mean_interval_score']:.2f}")

    print("\nE14 — WIDTH-GATED BACKTEST (long-only, causal)")
    print(f"{'policy':<20}{'AnnRet':>9}{'AnnVol':>9}{'Sharpe':>9}{'MaxDD':>9}")
    for name, b in R["E14_backtest"].items():
        print(f"{name:<20}{b['ann_return']:>9.3f}{b['ann_vol']:>9.3f}"
              f"{b['sharpe']:>9.3f}{b['max_drawdown']:>9.3f}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run DeepPIPE-Ctrl experiments E11-E14.")
    ap.add_argument("--data", default=config.DATA_PATH, help="Path to market_data.csv")
    ap.add_argument("--out-dir", default=".", help="Output directory")
    ap.add_argument("--results", default="results_ctrl.json", help="Results JSON name")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-results", action="store_true")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, args.results)
    if args.reuse_results and os.path.isfile(results_path):
        with open(results_path) as fh:
            R = json.load(fh)
    else:
        if not os.path.isfile(args.data):
            raise SystemExit(f"Data file not found: {args.data}")
        R = run_ctrl(args.data, results_path=results_path, seed=args.seed)
    print_ctrl_tables(R)


if __name__ == "__main__":
    main()
