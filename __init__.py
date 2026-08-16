"""DeepPIPE NIFTY 50 reproduction package.

Converted from DeepPIPE_Colab_AllInOne.ipynb into a plain Python package.

Modules:
    config       -- constants, feature sets, device, logger
    data         -- CSV loading, feature engineering, sequence/split builders
    model        -- DeepPIPE network, PI loss, metrics
    train        -- training/eval loops and RNN baselines
    stats        -- Diebold-Mariano, bootstrap CIs, Kupiec/Christoffersen, agg
    experiments  -- run_all(): experiments E1-E10 -> RESULTS dict + JSON
    report       -- print_tables(): all result tables
    figures      -- make_figures(): fig1..fig6
    main         -- CLI entry point

DeepPIPE-Ctrl (additive RL interval-width controller, design doc §8.3):
    reward       -- Winkler interval score (inner reward)
    state        -- controller state features s_t (fold-local, exogenous)
    dataset      -- make_bandit_dataset(): offline reward table {S, A, R}
    policies     -- T0 GreedyConstant / T1 LinUCB,LinTS / T2 MLPPolicy
    baselines    -- static_conformal, conformal_pid
    rollout      -- roll_controller(): per-step intervals + inner metrics
    backtest     -- width_gated_backtest(): outer economic metrics
    run_ctrl     -- run_ctrl(): experiments E11-E14 -> results_ctrl.json
"""
