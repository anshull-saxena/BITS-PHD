"""DeepPIPE-Ctrl — adaptive interval-width controller."""

from deeppipe.ctrl.backtest import returns_from_prices, width_gated_backtest
from deeppipe.ctrl.dataset import make_bandit_dataset
from deeppipe.ctrl.policies import act, fit_policy
from deeppipe.ctrl.reward import interval_score, interval_score_norm
from deeppipe.ctrl.rollout import eval_intervals, roll_controller

__all__ = [
    "act",
    "eval_intervals",
    "fit_policy",
    "interval_score",
    "interval_score_norm",
    "make_bandit_dataset",
    "returns_from_prices",
    "roll_controller",
    "width_gated_backtest",
]
