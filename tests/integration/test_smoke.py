"""Integration smoke tests on synthetic data."""

from deeppipe.config.schema import CtrlConfig, TrainConfig
from deeppipe.core.data import build_splits, load_raw
from deeppipe.core.train import train_deeppipe
from deeppipe.ctrl.dataset import conformal_q, make_bandit_dataset
from deeppipe.ctrl.policies import act, fit_policy
from deeppipe.ctrl.reward import interval_score
from deeppipe.ctrl.rollout import roll_controller
from deeppipe.experiments.ctrl import run_ctrl
from deeppipe.experiments.reproduce import run_all
from deeppipe.pipeline.ctrl import CtrlPipeline
from deeppipe.pipeline.reproduce import ReproducePipeline


def test_bandit_rejects_nonpositive_q(fast_run_config):
    import pytest

    raw = load_raw(fast_run_config.data_path)
    n = len(raw)
    tr_end = int(n * fast_run_config.data.train_split)
    va_end = tr_end + int(n * fast_run_config.data.val_split)
    sp = build_splits(raw, fast_run_config.data.features, tr_end, va_end, fast_run_config.train)
    model = train_deeppipe(sp, True, train_cfg=fast_run_config.train, seed=1)

    with pytest.raises(ValueError, match="must be positive"):
        make_bandit_dataset(
            raw, sp, model, 0.0, tr_end,
            train_cfg=fast_run_config.train, ctrl_cfg=fast_run_config.ctrl,
        )


def test_bandit_smoke(fast_run_config):
    raw = load_raw(fast_run_config.data_path)
    n = len(raw)
    tr_end = int(n * fast_run_config.data.train_split)
    va_end = tr_end + int(n * fast_run_config.data.val_split)
    sp = build_splits(raw, fast_run_config.data.features, tr_end, va_end, fast_run_config.train)
    model = train_deeppipe(sp, True, train_cfg=fast_run_config.train, seed=1)
    q, nc = conformal_q(model, sp, train_cfg=fast_run_config.train)
    assert q > 0 and nc > 0

    D = make_bandit_dataset(
        raw,
        sp,
        model,
        q,
        tr_end,
        split="Xte",
        ysplit="yte",
        row_offset=va_end + fast_run_config.train.seq_len,
        train_cfg=fast_run_config.train,
        ctrl_cfg=fast_run_config.ctrl,
    )
    assert D["R"].shape == (len(D["y"]), len(fast_run_config.ctrl.actions))
    assert D["S"].shape[1] == 8

    m1 = fast_run_config.ctrl.actions.index(1.0)
    static_score = interval_score(D["y"], D["point"] - q, D["point"] + q).mean()
    assert abs((-D["R"][:, m1] * q).mean() - static_score) < 1e-6

    pol = fit_policy(D, tier="greedy", seed=1)
    assert act(pol, D["S"][0]) in fast_run_config.ctrl.actions
    roll = roll_controller(D, pol, p=fast_run_config.train.conf)
    assert "PICP" in roll["metrics"]


def test_run_all_smoke(fast_run_config, tmp_path):
    results_path = tmp_path / "results_all.json"
    # Minimal subset: override to run faster by using tiny epochs already in fixture
    fast_cfg = fast_run_config
    fast_cfg.train = TrainConfig(
        seq_len=20, warmup_ep=1, full_ep=1, batch=16, conf=0.90
    )
    R = run_all(fast_cfg, results_path=results_path)
    assert "E1_main" in R
    assert results_path.is_file()


def test_run_ctrl_smoke(fast_run_config, tmp_path):
    fast_cfg = fast_run_config
    fast_cfg.train = TrainConfig(seq_len=20, warmup_ep=1, full_ep=1, batch=16)
    fast_cfg.ctrl = CtrlConfig(tiers=["greedy"], seed=1)
    results_path = tmp_path / "results_ctrl.json"
    R = run_ctrl(fast_cfg, results_path=results_path)
    assert "E11_inner" in R
    assert results_path.is_file()


def test_reproduce_pipeline(fast_run_config):
    fast_run_config.train = TrainConfig(seq_len=20, warmup_ep=1, full_ep=1, batch=16)
    R = ReproducePipeline(fast_run_config).run()
    assert "E1_main" in R


def test_ctrl_pipeline(fast_run_config):
    fast_run_config.train = TrainConfig(seq_len=20, warmup_ep=1, full_ep=1, batch=16)
    fast_run_config.ctrl = CtrlConfig(tiers=["greedy"], seed=1)
    R = CtrlPipeline(fast_run_config).run()
    assert "E11_inner" in R
