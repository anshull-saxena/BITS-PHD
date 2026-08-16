"""Run all experiments E1-E10 and build the RESULTS dictionary.

Ported from Cell 4 of DeepPIPE_Colab_AllInOne.ipynb.
"""
import json
import time

import numpy as np

import config
from config import (MAIN_FEATURES, FEATURE_SETS, TRAIN_SPLIT, VAL_SPLIT,
                    SEQ_LEN, WARMUP_EP, device, log)
from data import load_raw, build_splits, make_sequences, TSData, inv_target
from model import (DeepPIPE, pipe_loss, set_seed, _y, point_metrics,
                   interval_metrics)
from train import train_deeppipe, eval_deeppipe, run_baselines
from stats import (diebold_mariano, bootstrap_mae_ci, bootstrap_diff_ci,
                   kupiec_pof, christoffersen_ind, agg)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def _eval_epoch(model, loader):
    model.eval()
    tm = tp = tw = nb = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), _y(yb.to(device))
            o = model(xb)
            tm += F.l1_loss(o["point"], yb).item()
            cap = ((yb >= o["lower"]) & (yb <= o["upper"])).float()
            tp += cap.mean().item()
            tw += (o["upper"] - o["lower"]).mean().item()
            nb += 1
    return tm / nb, tp / nb, tw / nb


def _instrumented(sp, two_phase, seed=42, warmup_ep=WARMUP_EP, full_ep=config.FULL_EP):
    """Training loop that records per-epoch validation PICP (for E10 dynamics)."""
    import copy
    set_seed(seed)
    model = DeepPIPE(sp["Xtr"].shape[2]).to(device)
    tl = DataLoader(TSData(sp["Xtr"], sp["ytr"]), batch_size=config.BATCH, shuffle=True)
    vl = DataLoader(TSData(sp["Xva"], sp["yva"]), batch_size=config.BATCH, shuffle=False)
    h = {"epoch": [], "phase": [], "val_picp": []}
    g = 0
    if two_phase:
        opt = optim.Adam(model.parameters(), lr=config.LR)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=warmup_ep, eta_min=1e-5)
        best, bs, pat = float("inf"), None, 0
        for ep in range(warmup_ep):
            model.train()
            for xb, yb in tl:
                xb, yb = xb.to(device), _y(yb.to(device))
                opt.zero_grad()
                loss = F.l1_loss(model(xb)["point"], yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            mae, picp, _ = _eval_epoch(model, vl)
            g += 1
            h["epoch"].append(g)
            h["phase"].append(1)
            h["val_picp"].append(picp)
            if mae < best:
                best, bs, pat = mae, copy.deepcopy(model.state_dict()), 0
            else:
                pat += 1
            if pat >= 20:
                break
        model.load_state_dict(bs)
    opt = optim.AdamW(model.parameters(), lr=config.LR * (0.5 if two_phase else 1.0), weight_decay=1e-5)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15)
    best, bs, pat = float("inf"), None, 0
    for ep in range(full_ep):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), _y(yb.to(device))
            opt.zero_grad()
            loss, _, _, _ = pipe_loss(model(xb), yb)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                l, _, _, _ = pipe_loss(model(xb.to(device)), _y(yb.to(device)))
                vloss += l.item()
        vloss /= len(vl)
        mae, picp, _ = _eval_epoch(model, vl)
        g += 1
        h["epoch"].append(g)
        h["phase"].append(2)
        h["val_picp"].append(picp)
        sch.step(vloss)
        if vloss < best:
            best, bs, pat = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            pat += 1
        if pat >= 25:
            break
    return h


def run_all(data_path, results_path="results_all.json"):
    """Execute experiments E1-E10 and persist RESULTS to JSON."""
    RESULTS = {"meta": {"device": str(device), "torch": torch.__version__}}
    t0 = time.time()
    raw = load_raw(data_path)
    n = len(raw)
    tr_end = int(n * TRAIN_SPLIT)
    va_end = tr_end + int(n * VAL_SPLIT)
    RESULTS["meta"].update(
        total_days=int(n), train_end=tr_end, val_end=va_end,
        train_price_range=[float(raw.NIFTY_Close[:tr_end].min()), float(raw.NIFTY_Close[:tr_end].max())],
        test_price_range=[float(raw.NIFTY_Close[va_end:].min()), float(raw.NIFTY_Close[va_end:].max())],
    )
    sp = build_splits(raw, MAIN_FEATURES, tr_end, va_end)
    log(f'{n} days | train<= {RESULTS["meta"]["train_price_range"][1]:.0f}, '
        f'test up to {RESULTS["meta"]["test_price_range"][1]:.0f}')

    # E1 main + E7 significance (MAE-consistent DM) + bootstrap
    log("E1: main comparison")
    m = train_deeppipe(sp, True, seed=42)
    y, p, lo, hi = eval_deeppipe(m, sp)
    dp = dict(**point_metrics(y, p), **interval_metrics(y, lo, hi))
    base = run_baselines(sp, y, seed=42, include_rnn=True)
    RESULTS["E1_main"] = dict(
        deeppipe=dp,
        baselines={k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in base.items()},
    )
    log("E7: Diebold-Mariano (abs-loss) + bootstrap")
    dm = {}
    for k, v in base.items():
        dabs, pabs = diebold_mariano(y, p, v["pred"], power=1)
        md, cl, ch = bootstrap_diff_ci(y, p, v["pred"])
        dm[k] = dict(DM_abs=dabs, p_abs=pabs, mae_diff=md, mae_diff_ci=[cl, ch])
    RESULTS["E7_significance"] = dict(
        dm_abs_and_bootstrap=dm,
        deeppipe_mae_bootstrap95=list(bootstrap_mae_ci(y, p)),
    )

    # E6 calibration across nominal levels (trained-alpha)
    log("E6: calibration (trained alpha) 80/90/95")
    calib = {}
    for c in (0.80, 0.90, 0.95):
        config.CONF = c
        mc = train_deeppipe(sp, True, seed=42)
        yy, pp, ll, hh = eval_deeppipe(mc, sp)
        cap = (yy >= ll) & (yy <= hh)
        klr, kp, kx, kn = kupiec_pof(cap, c)
        clr, cpv = christoffersen_ind(cap)
        calib[str(int(c * 100))] = dict(
            target=c, **interval_metrics(yy, ll, hh),
            kupiec=dict(LR=klr, p=kp, failures=kx, n=kn),
            christoffersen=dict(LR=clr, p=cpv),
        )
    config.CONF = 0.90
    RESULTS["E6_calibration"] = calib

    # E8 VIX adaptiveness (uses E1 preds)
    log("E8: VIX adaptiveness")
    vix = sp["vix_test"]
    if vix is not None and len(vix) >= len(hi):
        vix = vix[:len(hi)]
        width = hi - lo
        corr = float(np.corrcoef(vix, width)[0, 1])
        q = np.quantile(vix, [0.33, 0.66])
        bk = {}
        for lab, mask in (("low", vix <= q[0]), ("mid", (vix > q[0]) & (vix <= q[1])), ("high", vix > q[1])):
            if mask.sum() > 0:
                bk[lab] = dict(n=int(mask.sum()), mean_vix=float(vix[mask].mean()),
                               mean_width=float(width[mask].mean()))
        RESULTS["E8_vix"] = dict(corr_width_vix=corr, buckets=bk)

    # E2 fail vs fix over 5 seeds + lambda + beta ablations
    log("E2: fail vs fix, 5 seeds")
    ab = {"single_phase": [], "two_phase": []}
    for seed in (42, 1, 2, 3, 4):
        for mode, tp in (("single_phase", False), ("two_phase", True)):
            mm = train_deeppipe(sp, tp, seed=seed)
            yy, pp, ll, hh = eval_deeppipe(mm, sp)
            ab[mode].append(dict(seed=seed, **point_metrics(yy, pp), **interval_metrics(yy, ll, hh)))
    RESULTS["E2_failfix"] = dict(
        seeds=[42, 1, 2, 3, 4], single_phase=ab["single_phase"], two_phase=ab["two_phase"],
        single_phase_agg=agg(ab["single_phase"]), two_phase_agg=agg(ab["two_phase"]),
    )
    log("E2b: lambda sweep")
    RESULTS["E2b_lambda"] = {}
    for lam in (1.0, 10.0, 15.0, 50.0):
        mm = train_deeppipe(sp, True, seed=42, lam=lam)
        yy, pp, ll, hh = eval_deeppipe(mm, sp)
        RESULTS["E2b_lambda"][str(lam)] = dict(**point_metrics(yy, pp), **interval_metrics(yy, ll, hh))
    log("E2c: beta sweep")
    RESULTS["E2c_beta"] = {}
    for b in (0.1, 0.3, 0.5, 0.7):
        mm = train_deeppipe(sp, True, seed=42, beta=b)
        yy, pp, ll, hh = eval_deeppipe(mm, sp)
        RESULTS["E2c_beta"][str(b)] = dict(**point_metrics(yy, pp), **interval_metrics(yy, ll, hh))

    # E3 feature ablation
    log("E3: feature ablation")
    RESULTS["E3_features"] = {}
    for name, feats in FEATURE_SETS.items():
        spf = build_splits(raw, feats, tr_end, va_end)
        mm = train_deeppipe(spf, True, seed=42)
        yy, pp, ll, hh = eval_deeppipe(mm, spf)
        RESULTS["E3_features"][name] = dict(n_features=len(feats), **point_metrics(yy, pp), **interval_metrics(yy, ll, hh))

    # E4 robustness (5 seeds, two-phase)
    log("E4: robustness")
    rows = []
    for s in (42, 1, 2, 3, 4):
        mm = train_deeppipe(sp, True, seed=s)
        yy, pp, ll, hh = eval_deeppipe(mm, sp)
        rows.append(dict(seed=s, **point_metrics(yy, pp), **interval_metrics(yy, ll, hh)))
    RESULTS["E4_robustness"] = dict(rows=rows, agg=agg(rows))

    # E5 rolling window
    log("E5: rolling-window")
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
        tdf = raw.iloc[ts:te]
        tef = spk["scaler"].transform(tdf[MAIN_FEATURES])
        Xk, yk = make_sequences(tef, spk["tgt_idx"], SEQ_LEN)
        if len(Xk) < 10:
            continue
        spk["Xte"], spk["yte"] = Xk, yk
        mm = train_deeppipe(spk, True, seed=42, warmup_ep=30, full_ep=80)
        yy, pp, ll, hh = eval_deeppipe(mm, spk)
        bk = run_baselines(spk, yy, seed=42, include_rnn=False)
        folds.append(dict(
            fold=k, test_period=[str(tdf.Date.min().date()), str(tdf.Date.max().date())],
            price_range=[float(tdf.NIFTY_Close.min()), float(tdf.NIFTY_Close.max())],
            deeppipe=dict(**point_metrics(yy, pp), **interval_metrics(yy, ll, hh)),
            naive=point_metrics(yy, bk["Naive"]["pred"]),
        ))
    RESULTS["E5_rolling"] = dict(folds=folds)

    # E9 split-conformal calibration
    log("E9: split-conformal")
    mconf = train_deeppipe(sp, True, seed=42)

    def _pp(kx, ky):
        mconf.eval()
        yt, pr = [], []
        dl = DataLoader(TSData(sp[kx], sp[ky]), batch_size=config.BATCH, shuffle=False)
        with torch.no_grad():
            for xb, yb in dl:
                o = mconf(xb.to(device))
                yt.extend(yb.numpy().flatten())
                pr.extend(o["point"].cpu().numpy().flatten())
        s, f, ti = sp["scaler"], sp["features"], sp["tgt_idx"]
        return inv_target(yt, s, f, ti), inv_target(pr, s, f, ti)

    yv, pv = _pp("Xva", "yva")
    yte, pte = _pp("Xte", "yte")
    cal = np.abs(yv - pv)
    nc = len(cal)
    conf = {}
    for tau in (0.80, 0.90, 0.95):
        kk = min(nc, int(np.ceil((nc + 1) * tau)))
        qq = float(np.sort(cal)[kk - 1])
        loo = pte - qq
        hii = pte + qq
        cap = (yte >= loo) & (yte <= hii)
        klr, kp, kx, kn = kupiec_pof(cap, tau)
        conf[str(int(tau * 100))] = dict(
            target=tau, q_halfwidth=qq, **interval_metrics(yte, loo, hii),
            kupiec=dict(LR=klr, p=kp, failures=kx, n=kn),
        )
    RESULTS["E9_conformal"] = dict(n_calibration=int(nc), levels=conf, mean_test_price=float(np.mean(yte)))

    # E10 training dynamics + warm-up sweep
    log("E10: training dynamics + warm-up sweep")
    RESULTS["E10_dynamics"] = {
        "single_phase": _instrumented(sp, False),
        "two_phase": _instrumented(sp, True),
        "warmup_epochs": WARMUP_EP,
    }
    RESULTS["E10_warmup"] = {}
    for wep in (0, 10, 20, 40):
        mm = train_deeppipe(sp, wep > 0, seed=42, warmup_ep=max(wep, 1))
        yy, pp, ll, hh = eval_deeppipe(mm, sp)
        RESULTS["E10_warmup"][str(wep)] = dict(**point_metrics(yy, pp), **interval_metrics(yy, ll, hh))

    # store test predictions for the interval-band figure
    RESULTS["predictions"] = {
        "actual": [float(v) for v in y],
        "point": [float(v) for v in p],
        "lower": [float(v) for v in lo],
        "upper": [float(v) for v in hi],
    }

    RESULTS["meta"]["runtime_min"] = round((time.time() - t0) / 60, 1)
    json.dump(RESULTS, open(results_path, "w"), indent=2, default=float)
    log(f'DONE in {RESULTS["meta"]["runtime_min"]} min. Saved {results_path}')
    return RESULTS
