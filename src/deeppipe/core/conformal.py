"""Split-conformal calibration helpers (E9 seam)."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from deeppipe.config.schema import TrainConfig
from deeppipe.core.data import TSData, inv_target
from deeppipe.core.model import interval_metrics
from deeppipe.core.stats import kupiec_pof


def predict_split(model, sp, kx, ky, train_cfg: TrainConfig | None = None):
    """Return (y_true, point_pred) in price units for split keys kx/ky."""
    cfg = train_cfg or TrainConfig()
    device = cfg.resolve_device()
    model.eval()
    yt, pr = [], []
    dl = DataLoader(TSData(sp[kx], sp[ky]), batch_size=cfg.batch, shuffle=False)
    with torch.no_grad():
        for xb, yb in dl:
            o = model(xb.to(device))
            yt.extend(yb.numpy().flatten())
            pr.extend(o["point"].cpu().numpy().flatten())
    s, f, ti = sp["scaler"], sp["features"], sp["tgt_idx"]
    return inv_target(yt, s, f, ti), inv_target(pr, s, f, ti)


def conformal_q(model, sp, tau=None, train_cfg: TrainConfig | None = None):
    """Split-conformal base half-width from validation residuals."""
    cfg = train_cfg or TrainConfig()
    if tau is None:
        tau = cfg.conf
    yv, pv = predict_split(model, sp, "Xva", "yva", cfg)
    cal = np.abs(yv - pv)
    nc = len(cal)
    kk = min(nc, int(np.ceil((nc + 1) * tau)))
    return float(np.sort(cal)[kk - 1]), nc


def conformal_intervals(model, sp, levels=(0.80, 0.90, 0.95), train_cfg: TrainConfig | None = None):
    """Run split-conformal calibration at multiple nominal levels (E9)."""
    cfg = train_cfg or TrainConfig()
    yv, pv = predict_split(model, sp, "Xva", "yva", cfg)
    yte, pte = predict_split(model, sp, "Xte", "yte", cfg)
    cal = np.abs(yv - pv)
    nc = len(cal)
    conf = {}
    for tau in levels:
        kk = min(nc, int(np.ceil((nc + 1) * tau)))
        qq = float(np.sort(cal)[kk - 1])
        loo = pte - qq
        hii = pte + qq
        cap = (yte >= loo) & (yte <= hii)
        klr, kp, kx, kn = kupiec_pof(cap, tau)
        conf[str(int(tau * 100))] = dict(
            target=tau,
            q_halfwidth=qq,
            **interval_metrics(yte, loo, hii),
            kupiec=dict(LR=klr, p=kp, failures=kx, n=kn),
        )
    return dict(n_calibration=int(nc), levels=conf, mean_test_price=float(np.mean(yte)))
