"""Training and evaluation loops for DeepPIPE and RNN baselines.

Ported from Cell 3 of DeepPIPE_Colab_AllInOne.ipynb.

Note: CONF is read from the ``config`` module at call time (``config.CONF``)
rather than imported by value, so experiments that mutate the nominal
confidence level (E6) are picked up by the training loop exactly as in the
original notebook.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression

import config
from config import BATCH, LR, WARMUP_EP, FULL_EP, BETA, LAMBDA, HIDDEN, LAYERS, DROPOUT, device
from data import TSData, inv_target
from model import DeepPIPE, pipe_loss, set_seed, _y, point_metrics


def train_deeppipe(sp, two_phase=True, beta=BETA, lam=LAMBDA,
                   warmup_ep=WARMUP_EP, full_ep=FULL_EP, seed=42):
    set_seed(seed)
    model = DeepPIPE(sp["Xtr"].shape[2]).to(device)
    tl = DataLoader(TSData(sp["Xtr"], sp["ytr"]), batch_size=BATCH, shuffle=True)
    vl = DataLoader(TSData(sp["Xva"], sp["yva"]), batch_size=BATCH, shuffle=False)
    if two_phase:
        opt = optim.Adam(model.parameters(), lr=LR)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=warmup_ep, eta_min=1e-5)
        best, bs, pat = float("inf"), None, 0
        for ep in range(warmup_ep):
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
    opt = optim.AdamW(model.parameters(), lr=LR * (0.5 if two_phase else 1.0), weight_decay=1e-5)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15)
    best, bs, pat = float("inf"), None, 0
    for ep in range(full_ep):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), _y(yb.to(device))
            opt.zero_grad()
            o = model(xb)
            loss, _, _, _ = pipe_loss(o, yb, beta, config.CONF, lam)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                l, _, _, _ = pipe_loss(model(xb.to(device)), _y(yb.to(device)), beta, config.CONF, lam)
                vloss += l.item()
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


def eval_deeppipe(model, sp):
    model.eval()
    yt, pr, lo, hi = [], [], [], []
    tl = DataLoader(TSData(sp["Xte"], sp["yte"]), batch_size=BATCH, shuffle=False)
    with torch.no_grad():
        for xb, yb in tl:
            o = model(xb.to(device))
            yt.extend(yb.numpy().flatten())
            pr.extend(o["point"].cpu().numpy().flatten())
            lo.extend(o["lower"].cpu().numpy().flatten())
            hi.extend(o["upper"].cpu().numpy().flatten())
    s, f, ti = sp["scaler"], sp["features"], sp["tgt_idx"]
    return (inv_target(yt, s, f, ti), inv_target(pr, s, f, ti),
            inv_target(lo, s, f, ti), inv_target(hi, s, f, ti))


class SimpleRNN(nn.Module):
    """Plain LSTM/GRU point-forecast baseline."""

    def __init__(self, in_size, kind="lstm"):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(in_size, HIDDEN, LAYERS, batch_first=True, dropout=DROPOUT)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1])


def train_rnn(sp, kind="lstm", epochs=80, seed=42):
    set_seed(seed)
    m = SimpleRNN(sp["Xtr"].shape[2], kind).to(device)
    opt = optim.Adam(m.parameters(), lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    tl = DataLoader(TSData(sp["Xtr"], sp["ytr"]), batch_size=BATCH, shuffle=True)
    vl = DataLoader(TSData(sp["Xva"], sp["yva"]), batch_size=BATCH, shuffle=False)
    best, st = float("inf"), None
    for ep in range(epochs):
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
    tl2 = DataLoader(TSData(sp["Xte"], sp["yte"]), batch_size=BATCH, shuffle=False)
    with torch.no_grad():
        for xb, _ in tl2:
            preds.extend(m(xb.to(device)).squeeze(-1).cpu().numpy().flatten())
    return inv_target(preds, sp["scaler"], sp["features"], sp["tgt_idx"])


def run_baselines(sp, y_true, seed=42, include_rnn=True):
    s, f, ti = sp["scaler"], sp["features"], sp["tgt_idx"]
    n = len(y_true)
    out = {}
    out["Naive"] = dict(pred=inv_target(sp["Xte"][:n, -1, ti], s, f, ti))
    out["MovingAvg5"] = dict(pred=inv_target(sp["Xte"][:n, -5:, ti].mean(axis=1), s, f, ti))
    lr = LinearRegression().fit(sp["Xtr"].reshape(len(sp["Xtr"]), -1), sp["ytr"].ravel())
    out["LinearReg"] = dict(pred=inv_target(lr.predict(sp["Xte"].reshape(len(sp["Xte"]), -1))[:n], s, f, ti))
    if include_rnn:
        for k in ("lstm", "gru"):
            out[k.upper()] = dict(pred=train_rnn(sp, k, seed=seed))
    for k in out:
        out[k].update(point_metrics(y_true, out[k]["pred"]))
    return out
