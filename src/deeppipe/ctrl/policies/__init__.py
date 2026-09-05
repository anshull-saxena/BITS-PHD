"""Policy tiers T0/T1/T2 for DeepPIPE-Ctrl."""

import numpy as np
import torch
import torch.nn as nn

from deeppipe.core.model import set_seed


def _phi(S):
    S = np.atleast_2d(np.asarray(S, dtype=float))
    return np.hstack([S, np.ones((S.shape[0], 1))])


class GreedyConstant:
    def __init__(self):
        self.A = None
        self.best_idx = 0
        self.m = 1.0

    def fit(self, D):
        self.A = list(D["A"])
        mean_reward = D["R"].mean(axis=0)
        self.best_idx = int(np.argmax(mean_reward))
        self.m = float(self.A[self.best_idx])
        return self

    def act(self, s):
        return self.m


class _LinContextual:
    def __init__(self, alpha=1.0, ridge=1.0, seed=42):
        self.alpha = alpha
        self.ridge = ridge
        self.seed = seed
        self.A = None
        self.Ainv = None
        self.theta = None

    def fit(self, D):
        self.A = list(D["A"])
        X = _phi(D["S"])
        R = D["R"]
        n_arms = len(self.A)
        d = X.shape[1]
        self.Ainv = np.zeros((n_arms, d, d))
        self.theta = np.zeros((n_arms, d))
        for a in range(n_arms):
            M = self.ridge * np.eye(d) + X.T @ X
            Minv = np.linalg.inv(M)
            self.Ainv[a] = Minv
            self.theta[a] = Minv @ (X.T @ R[:, a])
        return self


class LinUCB(_LinContextual):
    def act(self, s):
        x = _phi(s)[0]
        mean = self.theta @ x
        bonus = self.alpha * np.array(
            [np.sqrt(x @ self.Ainv[a] @ x) for a in range(len(self.A))]
        )
        return float(self.A[int(np.argmax(mean + bonus))])


class LinTS(_LinContextual):
    def __init__(self, alpha=1.0, ridge=1.0, seed=42):
        super().__init__(alpha, ridge, seed)
        self._rng = np.random.default_rng(seed)

    def act(self, s):
        x = _phi(s)[0]
        vals = np.empty(len(self.A))
        for a in range(len(self.A)):
            cov = (self.alpha ** 2) * self.Ainv[a]
            theta_s = self._rng.multivariate_normal(self.theta[a], cov)
            vals[a] = theta_s @ x
        return float(self.A[int(np.argmax(vals))])


class MLPPolicy(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.A = None

    def forward(self, x):
        return self.net(x)

    def fit(self, D, epochs=300, lr=1e-2, seed=42, weight_decay=1e-3):
        set_seed(seed)
        self.A = list(D["A"])
        X = torch.FloatTensor(np.asarray(D["S"], dtype=float))
        R = torch.FloatTensor(np.asarray(D["R"], dtype=float))
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        self.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self(X), R)
            loss.backward()
            opt.step()
        self.eval()
        return self

    def act(self, s):
        with torch.no_grad():
            x = torch.FloatTensor(np.atleast_2d(np.asarray(s, dtype=float)))
            q = self(x)[0].numpy()
        return float(self.A[int(np.argmax(q))])


_TIERS = {
    "greedy": GreedyConstant,
    "t0": GreedyConstant,
    "linucb": LinUCB,
    "lints": LinTS,
    "mlp": MLPPolicy,
    "t2": MLPPolicy,
}

TIER_LABELS = {
    "greedy": "T0_greedy",
    "linucb": "T1_linucb",
    "lints": "T1_lints",
    "mlp": "T2_mlp",
}


def fit_policy(D, tier="linucb", **kwargs):
    key = tier.lower()
    if key not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(_TIERS)}")
    cls = _TIERS[key]
    if cls is MLPPolicy:
        pol = MLPPolicy(in_dim=np.asarray(D["S"]).shape[1], n_actions=len(D["A"]))
        return pol.fit(D, **kwargs)
    if cls is GreedyConstant:
        return cls().fit(D)
    return cls(**kwargs).fit(D)


def act(policy, s_t):
    return policy.act(s_t)
