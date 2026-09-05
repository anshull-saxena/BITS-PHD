"""Unit tests for controller policies."""

import numpy as np

from deeppipe.ctrl.policies import act, fit_policy


def test_policy_fit_act():
    N, d, nA = 30, 8, 5
    D = {
        "S": np.random.default_rng(0).standard_normal((N, d)),
        "A": [0.5, 0.75, 1.0, 1.25, 1.5],
        "R": np.random.default_rng(1).standard_normal((N, nA)),
    }
    for tier in ("greedy", "linucb", "lints", "mlp"):
        pol = fit_policy(D, tier=tier, seed=0)
        m = act(pol, D["S"][0])
        assert m in D["A"]
