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
"""
