"""Run experiments E1-E10."""

import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from deeppipe.config.schema import FEATURE_SETS, RunConfig, TrainConfig
from deeppipe.core.conformal import conformal_intervals
from deeppipe.core.data import TSData, build_splits, load_raw, make_sequences
from deeppipe.core.model import DeepPIPE, _y, interval_metrics, pipe_loss, point_metrics, set_seed
from deeppipe.core.stats import (
    agg,
    bootstrap_diff_ci,
    bootstrap_mae_ci,
    christoffersen_ind,
    diebold_mariano,
    kupiec_pof,
)
from deeppipe.core.train import eval_deeppipe, run_baselines, train_deeppipe
from deeppipe.logging import log


def _eval_epoch(model, loader, train_cfg: TrainConfig):
    device = train_cfg.resolve_device()
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


def _instrumented(sp, two_phase, run_cfg: RunConfig, seed=42, warmup_ep=None, full_ep=None):
    train_cfg = run_cfg.train
    warmup_ep = warmup_ep if warmup_ep is not None else train_cfg.warmup_ep
    full_ep = full_ep if full_ep is not None else train_cfg.full_ep
    device = train_cfg.resolve_device()
    set_seed(seed)
    model = DeepPIPE(sp["Xtr"].shape[2], train_cfg).to(device)
    tl = DataLoader(TSData(sp["Xtr"], sp["ytr"]), batch_size=train_cfg.batch, shuffle=True)
    vl = DataLoader(TSData(sp["Xva"], sp["yva"]), batch_size=train_cfg.batch, shuffle=False)
    h = {"epoch": [], "phase": [], "val_picp": []}
    g = 0
    if two_phase:
        opt = optim.Adam(model.parameters(), lr=train_cfg.lr)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=warmup_ep, eta_min=1e-5)
        best, bs, pat = float("inf"), None, 0
        for _ in range(warmup_ep):
            model.train()
            for xb, yb in tl:
                xb, yb = xb.to(device), _y(yb.to(device))
                opt.zero_grad()
                loss = F.l1_loss(model(xb)["point"], yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            mae, picp, _ = _eval_epoch(model, vl, train_cfg)
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
    opt = optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr * (0.5 if two_phase else 1.0),
        weight_decay=1e-5,
    )
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15)
    best, bs, pat = float("inf"), None, 0
    for _ in range(full_ep):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), _y(yb.to(device))
            opt.zero_grad()
            loss, _, _, _ = pipe_loss(model(xb), yb, train_cfg)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                loss_val, _, _, _ = pipe_loss(
                    model(xb.to(device)), _y(yb.to(device)), train_cfg
                )
                vloss += loss_val.item()
        vloss /= len(vl)
        mae, picp, _ = _eval_epoch(model, vl, train_cfg)
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


def run_all(run_cfg: RunConfig, results_path=None):
    """Execute experiments E1-E10 and persist RESULTS to JSON."""
    train_cfg = run_cfg.train
    data_cfg = run_cfg.data
    features = data_cfg.features
    device = train_cfg.resolve_device()

    RESULTS = {"meta": {"device": str(device), "torch": torch.__version__}}
    t0 = time.time()
    raw = load_raw(run_cfg.data_path)
    n = len(raw)
    tr_end = int(n * data_cfg.train_split)
    va_end = tr_end + int(n * data_cfg.val_split)
    RESULTS["meta"].update(
        total_days=int(n),
        train_end=tr_end,
        val_end=va_end,
        train_price_range=[
            float(raw.NIFTY_Close[:tr_end].min()),
            float(raw.NIFTY_Close[:tr_end].max()),
        ],
        test_price_range=[
            float(raw.NIFTY_Close[va_end:].min()),
            float(raw.NIFTY_Close[va_end:].max()),
        ],
    )
    sp = build_splits(raw, features, tr_end, va_end, train_cfg)
    log(
        f'{n} days | train<= {RESULTS["meta"]["train_price_range"][1]:.0f}, '
        f'test up to {RESULTS["meta"]["test_price_range"][1]:.0f}'
    )

    log("E1: main comparison")
    m = train_deeppipe(sp, True, train_cfg=train_cfg, seed=run_cfg.seed)
    y, p, lo, hi = eval_deeppipe(m, sp, train_cfg)
    dp = dict(**point_metrics(y, p), **interval_metrics(y, lo, hi))
    base = run_baselines(sp, y, seed=run_cfg.seed, include_rnn=True, train_cfg=train_cfg)
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

    log("E6: calibration (trained alpha) 80/90/95")
    calib = {}
    for c in (0.80, 0.90, 0.95):
        e6_cfg = TrainConfig(**{**train_cfg.__dict__, "conf": c})
        mc = train_deeppipe(sp, True, train_cfg=e6_cfg, seed=run_cfg.seed)
        yy, pp, ll, hh = eval_deeppipe(mc, sp, e6_cfg)
        cap = (yy >= ll) & (yy <= hh)
        klr, kp, kx, kn = kupiec_pof(cap, c)
        clr, cpv = christoffersen_ind(cap)
        calib[str(int(c * 100))] = dict(
            target=c,
            **interval_metrics(yy, ll, hh),
            kupiec=dict(LR=klr, p=kp, failures=kx, n=kn),
            christoffersen=dict(LR=clr, p=cpv),
        )
    RESULTS["E6_calibration"] = calib

    log("E8: VIX adaptiveness")
    vix = sp["vix_test"]
    if vix is not None and len(vix) >= len(hi):
        vix = vix[: len(hi)]
        width = hi - lo
        corr = float(np.corrcoef(vix, width)[0, 1])
        qv = np.quantile(vix, [0.33, 0.66])
        bk = {}
        for lab, mask in (
            ("low", vix <= qv[0]),
            ("mid", (vix > qv[0]) & (vix <= qv[1])),
            ("high", vix > qv[1]),
        ):
            if mask.sum() > 0:
                bk[lab] = dict(
                    n=int(mask.sum()),
                    mean_vix=float(vix[mask].mean()),
                    mean_width=float(width[mask].mean()),
                )
        RESULTS["E8_vix"] = dict(corr_width_vix=corr, buckets=bk)

    log("E2: fail vs fix, 5 seeds")
    ab = {"single_phase": [], "two_phase": []}
    for seed in (42, 1, 2, 3, 4):
        for mode, tp in (("single_phase", False), ("two_phase", True)):
            mm = train_deeppipe(sp, tp, train_cfg=train_cfg, seed=seed)
            yy, pp, ll, hh = eval_deeppipe(mm, sp, train_cfg)
            ab[mode].append(
                dict(seed=seed, **point_metrics(yy, pp), **interval_metrics(yy, ll, hh))
            )
    RESULTS["E2_failfix"] = dict(
        seeds=[42, 1, 2, 3, 4],
        single_phase=ab["single_phase"],
        two_phase=ab["two_phase"],
        single_phase_agg=agg(ab["single_phase"]),
        two_phase_agg=agg(ab["two_phase"]),
    )

    log("E2b: lambda sweep")
    RESULTS["E2b_lambda"] = {}
    for lam in (1.0, 10.0, 15.0, 50.0):
        mm = train_deeppipe(sp, True, train_cfg=train_cfg, lam=lam, seed=run_cfg.seed)
        yy, pp, ll, hh = eval_deeppipe(mm, sp, train_cfg)
        RESULTS["E2b_lambda"][str(lam)] = dict(
            **point_metrics(yy, pp), **interval_metrics(yy, ll, hh)
        )

    log("E2c: beta sweep")
    RESULTS["E2c_beta"] = {}
    for b in (0.1, 0.3, 0.5, 0.7):
        b_cfg = TrainConfig(**{**train_cfg.__dict__, "beta": b})
        mm = train_deeppipe(sp, True, train_cfg=b_cfg, seed=run_cfg.seed)
        yy, pp, ll, hh = eval_deeppipe(mm, sp, b_cfg)
        RESULTS["E2c_beta"][str(b)] = dict(
            **point_metrics(yy, pp), **interval_metrics(yy, ll, hh)
        )

    log("E3: feature ablation")
    RESULTS["E3_features"] = {}
    for name, feats in FEATURE_SETS.items():
        spf = build_splits(raw, feats, tr_end, va_end, train_cfg)
        mm = train_deeppipe(spf, True, train_cfg=train_cfg, seed=run_cfg.seed)
        yy, pp, ll, hh = eval_deeppipe(mm, spf, train_cfg)
        RESULTS["E3_features"][name] = dict(
            n_features=len(feats), **point_metrics(yy, pp), **interval_metrics(yy, ll, hh)
        )

    log("E4: robustness")
    rows = []
    for s in (42, 1, 2, 3, 4):
        mm = train_deeppipe(sp, True, train_cfg=train_cfg, seed=s)
        yy, pp, ll, hh = eval_deeppipe(mm, sp, train_cfg)
        rows.append(dict(seed=s, **point_metrics(yy, pp), **interval_metrics(yy, ll, hh)))
    RESULTS["E4_robustness"] = dict(rows=rows, agg=agg(rows))

    log("E5: rolling-window")
    K = 5
    start = int(n * 0.55)
    fs = (n - start) // K
    folds = []
    seq_len = train_cfg.seq_len
    for k in range(K):
        ts = start + k * fs
        te = ts + fs if k < K - 1 else n
        trk = int(ts * 0.85)
        vak = ts
        if trk <= seq_len + 10 or (te - ts) <= seq_len + 5:
            continue
        spk = build_splits(raw, features, trk, vak, train_cfg)
        tdf = raw.iloc[ts:te]
        tef = spk["scaler"].transform(tdf[features])
        Xk, yk = make_sequences(tef, spk["tgt_idx"], seq_len)
        if len(Xk) < 10:
            continue
        spk["Xte"], spk["yte"] = Xk, yk
        mm = train_deeppipe(
            spk, True, train_cfg=train_cfg, seed=run_cfg.seed, warmup_ep=30, full_ep=80
        )
        yy, pp, ll, hh = eval_deeppipe(mm, spk, train_cfg)
        bk = run_baselines(spk, yy, seed=run_cfg.seed, include_rnn=False, train_cfg=train_cfg)
        folds.append(
            dict(
                fold=k,
                test_period=[str(tdf.Date.min().date()), str(tdf.Date.max().date())],
                price_range=[float(tdf.NIFTY_Close.min()), float(tdf.NIFTY_Close.max())],
                deeppipe=dict(**point_metrics(yy, pp), **interval_metrics(yy, ll, hh)),
                naive=point_metrics(yy, bk["Naive"]["pred"]),
            )
        )
    RESULTS["E5_rolling"] = dict(folds=folds)

    log("E9: split-conformal")
    mconf = train_deeppipe(sp, True, train_cfg=train_cfg, seed=run_cfg.seed)
    RESULTS["E9_conformal"] = conformal_intervals(mconf, sp, train_cfg=train_cfg)

    log("E10: training dynamics + warm-up sweep")
    RESULTS["E10_dynamics"] = {
        "single_phase": _instrumented(sp, False, run_cfg),
        "two_phase": _instrumented(sp, True, run_cfg),
        "warmup_epochs": train_cfg.warmup_ep,
    }
    RESULTS["E10_warmup"] = {}
    for wep in (0, 10, 20, 40):
        mm = train_deeppipe(
            sp,
            wep > 0,
            train_cfg=train_cfg,
            seed=run_cfg.seed,
            warmup_ep=max(wep, 1),
        )
        yy, pp, ll, hh = eval_deeppipe(mm, sp, train_cfg)
        RESULTS["E10_warmup"][str(wep)] = dict(
            **point_metrics(yy, pp), **interval_metrics(yy, ll, hh)
        )

    RESULTS["predictions"] = {
        "actual": [float(v) for v in y],
        "point": [float(v) for v in p],
        "lower": [float(v) for v in lo],
        "upper": [float(v) for v in hi],
    }

    RESULTS["meta"]["runtime_min"] = round((time.time() - t0) / 60, 1)
    if results_path:
        with open(results_path, "w") as fh:
            json.dump(RESULTS, fh, indent=2, default=float)
        log(f'DONE in {RESULTS["meta"]["runtime_min"]} min. Saved {results_path}')
    return RESULTS
