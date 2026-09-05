"""DeepPIPE model, prediction-interval loss and metric helpers."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from deeppipe.config.schema import TrainConfig


class DeepPIPE(nn.Module):
    """Encoder-decoder LSTM producing a point forecast plus lower/upper bounds."""

    def __init__(self, in_size, train_cfg: TrainConfig | None = None):
        cfg = train_cfg or TrainConfig()
        super().__init__()
        hidden, layers, dropout = cfg.hidden, cfg.layers, cfg.dropout
        self.enc = nn.LSTM(
            in_size, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0
        )
        self.dec = nn.LSTM(
            in_size, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0
        )
        self.ln = nn.LayerNorm(hidden)
        self.do = nn.Dropout(dropout)
        self.point = nn.Linear(hidden, 1)
        self.lo = nn.Linear(hidden, 1)
        self.hi = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, c) = self.enc(x)
        out, _ = self.dec(x[:, -1:, :], (h, c))
        z = self.do(self.ln(out[:, 0, :]))
        p = self.point(z)
        return {
            "point": p,
            "lower": p - F.softplus(self.lo(z)),
            "upper": p + F.softplus(self.hi(z)),
        }


def pipe_loss(o, y, train_cfg: TrainConfig | None = None, beta=None, conf=None, lam=None):
    """Combined point-error + interval-width + coverage-penalty loss."""
    cfg = train_cfg or TrainConfig()
    beta = beta if beta is not None else cfg.beta
    conf = conf if conf is not None else cfg.conf
    lam = lam if lam is not None else cfg.lam
    p, lo, hi = o["point"], o["lower"], o["upper"]
    l_pe = F.l1_loss(p, y)
    cap = ((y >= lo) & (y <= hi)).float()
    picp = cap.mean()
    mpiw = ((hi - lo) * cap).mean()
    pen = torch.relu(torch.tensor(conf, device=y.device) - picp) ** 2
    return beta * l_pe + mpiw + lam * pen, l_pe, picp, mpiw


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def _y(yb):
    """Ensure the target tensor has a trailing feature dimension."""
    return yb.unsqueeze(1) if yb.dim() == 1 else yb


def point_metrics(y, p):
    return dict(
        MAE=float(mean_absolute_error(y, p)),
        RMSE=float(np.sqrt(mean_squared_error(y, p))),
        MAPE=float(np.mean(np.abs((y - p) / y)) * 100),
        R2=float(r2_score(y, p)),
    )


def interval_metrics(y, lo, hi):
    cap = (y >= lo) & (y <= hi)
    picp = float(np.mean(cap))
    mpiw = float(np.mean(hi[cap] - lo[cap])) if cap.sum() > 0 else 0.0
    return dict(
        PICP=picp,
        MPIW=mpiw,
        MPIW_all=float(np.mean(hi - lo)),
        captured=int(cap.sum()),
        n=int(len(y)),
    )
