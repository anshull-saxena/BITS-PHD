"""Training and evaluation loops for DeepPIPE and RNN baselines."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from deeppipe.config.schema import TrainConfig
from deeppipe.core.data import TSData, inv_target
from deeppipe.core.model import DeepPIPE, _y, pipe_loss, point_metrics, set_seed


def train_deeppipe(
    sp,
    two_phase=True,
    train_cfg: TrainConfig | None = None,
    beta=None,
    lam=None,
    conf=None,
    warmup_ep=None,
    full_ep=None,
    seed=42,
):
    cfg = train_cfg or TrainConfig()
    if beta is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "beta": beta})
    if lam is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "lam": lam})
    if conf is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "conf": conf})
    warmup_ep = warmup_ep if warmup_ep is not None else cfg.warmup_ep
    full_ep = full_ep if full_ep is not None else cfg.full_ep
    device = cfg.resolve_device()

    set_seed(seed)
    model = DeepPIPE(sp["Xtr"].shape[2], cfg).to(device)
    tl = DataLoader(TSData(sp["Xtr"], sp["ytr"]), batch_size=cfg.batch, shuffle=True)
    vl = DataLoader(TSData(sp["Xva"], sp["yva"]), batch_size=cfg.batch, shuffle=False)
    if two_phase:
        opt = optim.Adam(model.parameters(), lr=cfg.lr)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=warmup_ep, eta_min=1e-5)
        best, bs, pat = float("inf"), None, 0
        for _ in range(warmup_ep):
            model.train()
            for xb, yb in tl:
                xb, yb = xb.to(device), _y(yb.to(device))
                opt.zero_grad()
                o = model(xb)
                loss = F.l1_loss(o["point"], yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                for xb, yb in vl:
                    vloss += F.l1_loss(model(xb.to(device))["point"], _y(yb.to(device))).item()
            vloss /= len(vl)
            if vloss < best:
                best, bs, pat = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                pat += 1
            if pat >= 20:
                break
        model.load_state_dict(bs)
    opt = optim.AdamW(
        model.parameters(), lr=cfg.lr * (0.5 if two_phase else 1.0), weight_decay=1e-5
    )
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15)
    best, bs, pat = float("inf"), None, 0
    for _ in range(full_ep):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), _y(yb.to(device))
            opt.zero_grad()
            o = model(xb)
            loss, _, _, _ = pipe_loss(o, yb, cfg)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                loss_val, _, _, _ = pipe_loss(model(xb.to(device)), _y(yb.to(device)), cfg)
                vloss += loss_val.item()
        vloss /= len(vl)
        sch.step(vloss)
        if vloss < best:
            best, bs, pat = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            pat += 1
        if pat >= 25:
            break
    if bs is not None:
        model.load_state_dict(bs)
    return model


def eval_deeppipe(model, sp, train_cfg: TrainConfig | None = None):
    cfg = train_cfg or TrainConfig()
    device = cfg.resolve_device()
    model.eval()
    yt, pr, lo, hi = [], [], [], []
    tl = DataLoader(TSData(sp["Xte"], sp["yte"]), batch_size=cfg.batch, shuffle=False)
    with torch.no_grad():
        for xb, yb in tl:
            o = model(xb.to(device))
            yt.extend(yb.numpy().flatten())
            pr.extend(o["point"].cpu().numpy().flatten())
            lo.extend(o["lower"].cpu().numpy().flatten())
            hi.extend(o["upper"].cpu().numpy().flatten())
    s, f, ti = sp["scaler"], sp["features"], sp["tgt_idx"]
    return (
        inv_target(yt, s, f, ti),
        inv_target(pr, s, f, ti),
        inv_target(lo, s, f, ti),
        inv_target(hi, s, f, ti),
    )


class SimpleRNN(nn.Module):
    """Plain LSTM/GRU point-forecast baseline."""

    def __init__(self, in_size, kind="lstm", train_cfg: TrainConfig | None = None):
        cfg = train_cfg or TrainConfig()
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(
            in_size, cfg.hidden, cfg.layers, batch_first=True, dropout=cfg.dropout
        )
        self.fc = nn.Linear(cfg.hidden, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1])


def train_rnn(sp, kind="lstm", epochs=80, seed=42, train_cfg: TrainConfig | None = None):
    cfg = train_cfg or TrainConfig()
    device = cfg.resolve_device()
    set_seed(seed)
    m = SimpleRNN(sp["Xtr"].shape[2], kind, cfg).to(device)
    opt = optim.Adam(m.parameters(), lr=cfg.lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    tl = DataLoader(TSData(sp["Xtr"], sp["ytr"]), batch_size=cfg.batch, shuffle=True)
    vl = DataLoader(TSData(sp["Xva"], sp["yva"]), batch_size=cfg.batch, shuffle=False)
    best, st = float("inf"), None
    for _ in range(epochs):
        m.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.l1_loss(m(xb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sch.step()
        m.eval()
        vl_ = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                vl_ += F.l1_loss(m(xb.to(device)).squeeze(-1), yb.to(device)).item()
        vl_ /= len(vl)
        if vl_ < best:
            best, st = vl_, copy.deepcopy(m.state_dict())
    m.load_state_dict(st)
    m.eval()
    preds = []
    tl2 = DataLoader(TSData(sp["Xte"], sp["yte"]), batch_size=cfg.batch, shuffle=False)
    with torch.no_grad():
        for xb, _ in tl2:
            preds.extend(m(xb.to(device)).squeeze(-1).cpu().numpy().flatten())
    return inv_target(preds, sp["scaler"], sp["features"], sp["tgt_idx"])


def run_baselines(sp, y_true, seed=42, include_rnn=True, train_cfg: TrainConfig | None = None):
    cfg = train_cfg or TrainConfig()
    s, f, ti = sp["scaler"], sp["features"], sp["tgt_idx"]
    n = len(y_true)
    out = {}
    out["Naive"] = dict(pred=inv_target(sp["Xte"][:n, -1, ti], s, f, ti))
    out["MovingAvg5"] = dict(pred=inv_target(sp["Xte"][:n, -5:, ti].mean(axis=1), s, f, ti))
    lr = LinearRegression().fit(sp["Xtr"].reshape(len(sp["Xtr"]), -1), sp["ytr"].ravel())
    out["LinearReg"] = dict(
        pred=inv_target(lr.predict(sp["Xte"].reshape(len(sp["Xte"]), -1))[:n], s, f, ti)
    )
    if include_rnn:
        for k in ("lstm", "gru"):
            out[k.upper()] = dict(pred=train_rnn(sp, k, seed=seed, train_cfg=cfg))
    for k in out:
        out[k].update(point_metrics(y_true, out[k]["pred"]))
    return out
