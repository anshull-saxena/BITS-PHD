"""Print all result tables.

Ported from Cell 5 of DeepPIPE_Colab_AllInOne.ipynb.
"""


def print_tables(R):
    print("=" * 66)
    print("MAIN COMPARISON (E1)")
    print("=" * 66)
    print(f"{'Model':<22}{'MAE':>9}{'RMSE':>9}{'MAPE%':>8}{'R2':>8}{'PICP':>8}{'MPIW':>8}")
    d = R["E1_main"]["deeppipe"]
    for k, v in R["E1_main"]["baselines"].items():
        print(f"{k:<22}{v['MAE']:>9.1f}{v['RMSE']:>9.1f}{v['MAPE']:>8.2f}{v['R2']:>8.3f}{'-':>8}{'-':>8}")
    print(f"{'DeepPIPE(two-phase)':<22}{d['MAE']:>9.1f}{d['RMSE']:>9.1f}"
          f"{d['MAPE']:>8.2f}{d['R2']:>8.3f}{d['PICP']:>8.3f}{d['MPIW']:>8.0f}")

    print("\n" + "=" * 66)
    print("FAILURE vs FIX (E2, 5 seeds)")
    print("=" * 66)
    sa = R["E2_failfix"]["single_phase_agg"]
    ta = R["E2_failfix"]["two_phase_agg"]
    print(f"single-phase: MAE {sa['MAE']['mean']:.0f}+/-{sa['MAE']['std']:.0f} | "
          f"PICP {sa['PICP']['mean']:.2f}+/-{sa['PICP']['std']:.2f}")
    print("  per-seed PICP:", [round(r["PICP"], 3) for r in R["E2_failfix"]["single_phase"]])
    print(f"two-phase:    MAE {ta['MAE']['mean']:.0f}+/-{ta['MAE']['std']:.0f} | "
          f"PICP {ta['PICP']['mean']:.2f}+/-{ta['PICP']['std']:.2f}")
    print("  per-seed PICP:", [round(r["PICP"], 3) for r in R["E2_failfix"]["two_phase"]])

    print("\n" + "=" * 66)
    print("ABLATIONS (E2b lambda, E2c beta, E3 features)")
    print("=" * 66)
    print("lambda:", {k: (round(v["PICP"], 3), round(v["MPIW"])) for k, v in R["E2b_lambda"].items()})
    print("beta  :", {k: (round(v["PICP"], 3), round(v["MAE"])) for k, v in R["E2c_beta"].items()})
    for k, v in R["E3_features"].items():
        print(f"  {k:<12}({v['n_features']:>2}f) MAE={v['MAE']:.0f} PICP={v['PICP']:.3f} MPIW={v['MPIW']:.0f}")

    print("\n" + "=" * 66)
    print("WARM-UP SWEEP (E10)")
    print("=" * 66)
    for k, v in R["E10_warmup"].items():
        print(f"  warmup={k:>2}: MAE={v['MAE']:.0f} PICP={v['PICP']:.3f} MPIW={v['MPIW']:.0f}")

    print("\n" + "=" * 66)
    print("ROLLING WINDOW (E5)")
    print("=" * 66)
    for f in R["E5_rolling"]["folds"]:
        print(f"  fold{f['fold']} {f['test_period'][0]}..{f['test_period'][1]} "
              f"PICP={f['deeppipe']['PICP']:.3f} MAE={f['deeppipe']['MAE']:.0f} "
              f"(naive {f['naive']['MAE']:.0f})")

    print("\n" + "=" * 66)
    print("SIGNIFICANCE (E7, MAE-consistent DM + bootstrap diff CI)")
    print("=" * 66)
    for k, v in R["E7_significance"]["dm_abs_and_bootstrap"].items():
        print(f"  vs {k:<12} DM={v['DM_abs']:>7.2f} p={v['p_abs']:.1e} | "
              f"MAEdiff={v['mae_diff']:>6.0f} CI[{v['mae_diff_ci'][0]:.0f},{v['mae_diff_ci'][1]:.0f}]")

    print("\n" + "=" * 66)
    print("CONFORMAL (E9) vs TRAINED-ALPHA CALIBRATION (E6)")
    print("=" * 66)
    for k, v in R["E9_conformal"]["levels"].items():
        print(f"  conformal tau={v['target']}: PICP={v['PICP']:.3f} "
              f"MPIW={v['MPIW']:.0f} kupiec_p={v['kupiec']['p']:.3f}")
    for k, v in R["E6_calibration"].items():
        print(f"  trained  a={v['target']}: PICP={v['PICP']:.3f} "
              f"MPIW={v['MPIW']:.0f} kupiec_p={v['kupiec']['p']:.2e}")

    print("\n" + "=" * 66)
    print("VIX ADAPTIVENESS (E8)")
    print("=" * 66)
    print("  corr(width,VIX)=", round(R["E8_vix"]["corr_width_vix"], 3),
          "| buckets:", {k: round(v["mean_width"]) for k, v in R["E8_vix"]["buckets"].items()})
