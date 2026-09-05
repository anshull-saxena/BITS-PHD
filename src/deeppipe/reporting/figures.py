"""Generate all figures (fig1..fig6).

Ported from Cell 6 of DeepPIPE_Colab_AllInOne.ipynb. Uses a non-interactive
Matplotlib backend so it runs headless; figures are written to ``figures/``.
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_figures(R, out_dir="figures", show=False):
    os.makedirs(out_dir, exist_ok=True)

    # fig1: fail vs fix (derive seeds from the actual rows so it can't drift)
    ff = R["E2_failfix"]
    spv = [r["PICP"] for r in ff["single_phase"]]
    tpv = [r["PICP"] for r in ff["two_phase"]]
    seeds = [r["seed"] for r in ff["single_phase"]]
    x = np.arange(len(seeds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.bar(x - w / 2, spv, w, label="Single-phase (original)", color="#c0392b")
    ax.bar(x + w / 2, tpv, w, label="Two-phase (proposed)", color="#27ae60")
    ax.axhline(0.90, ls="--", c="gray", lw=1, label="Target 0.90")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("PICP")
    ax.set_ylim(0, 1.05)
    ax.set_title("Interval collapse vs fix (per seed)")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_failfix.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # fig2: beta sweep
    bs = R["E2c_beta"]
    bx = sorted(float(k) for k in bs)
    bp = [bs[str(b)]["PICP"] for b in bx]
    bm = [bs[str(b)]["MAE"] for b in bx]
    fig, a1 = plt.subplots(figsize=(6.5, 4))
    a1.plot(bx, bp, "o-", color="#2980b9", label="PICP")
    a1.axhline(0.90, ls="--", c="gray", lw=1)
    a1.set_xlabel(r"$\beta$")
    a1.set_ylabel("PICP", color="#2980b9")
    a2 = a1.twinx()
    a2.plot(bx, bm, "s--", color="#e67e22")
    a2.set_ylabel("MAE", color="#e67e22")
    a1.set_title(r"$\beta$ controls calibration")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_beta_sweep.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # fig3: rolling window
    folds = R["E5_rolling"]["folds"]
    lab = [f"F{f['fold']}\n{f['test_period'][0][:7]}" for f in folds]
    picp = [f["deeppipe"]["PICP"] for f in folds]
    pmax = [f["price_range"][1] for f in folds]
    tmax = R["meta"]["train_price_range"][1]
    fig, a1 = plt.subplots(figsize=(7, 4))
    col = ["#27ae60" if p >= 0.85 else "#c0392b" for p in picp]
    a1.bar(range(len(folds)), picp, color=col)
    a1.axhline(0.90, ls="--", c="gray", lw=1)
    a1.set_xticks(range(len(folds)))
    a1.set_xticklabels(lab, fontsize=9)
    a1.set_ylabel("PICP")
    a1.set_ylim(0, 1.05)
    a2 = a1.twinx()
    a2.plot(range(len(folds)), pmax, "k^-", label="Fold max price")
    a2.axhline(tmax, ls=":", c="purple", lw=1.2, label=f"Train max ({tmax:.0f})")
    a2.set_ylabel("Price")
    a1.set_title("Coverage collapses when prices exceed training range")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_rolling.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # fig4: training dynamics
    s = R["E10_dynamics"]["single_phase"]
    t = R["E10_dynamics"]["two_phase"]
    wu = R["E10_dynamics"]["warmup_epochs"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s["epoch"], s["val_picp"], color="#c0392b", label="Single-phase")
    ax.plot(t["epoch"], t["val_picp"], color="#27ae60", label="Two-phase")
    ax.axvline(wu + 0.5, ls=":", c="gray", lw=1.2)
    ax.axhline(0.90, ls="--", c="black", lw=0.8, alpha=0.6, label="Target 0.90")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation PICP")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Validation coverage during training")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_dynamics.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # fig5: interval band
    P = R["predictions"]
    xi = np.arange(len(P["actual"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(xi, P["lower"], P["upper"], color="#27ae60", alpha=0.18, label="90% interval")
    ax.plot(xi, P["actual"], color="#2c3e50", lw=1.3, label="Actual")
    ax.plot(xi, P["point"], "--", color="#e67e22", lw=1.1, label="Point forecast")
    ax.set_xlabel("Test trading day")
    ax.set_ylabel("NIFTY 50 close")
    ax.set_title("Two-phase DeepPIPE: point forecast and 90% interval")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_intervals.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # fig6: warm-up sweep
    sw = R["E10_warmup"]
    ks = sorted(int(k) for k in sw)
    wp = [sw[str(k)]["PICP"] for k in ks]
    wm = [sw[str(k)]["MAE"] for k in ks]
    fig, a1 = plt.subplots(figsize=(6.5, 4))
    a1.plot(ks, wp, "o-", color="#2980b9", label="PICP")
    a1.axhline(0.90, ls="--", c="gray", lw=1)
    a1.set_xlabel("Warm-up epochs")
    a1.set_ylabel("PICP", color="#2980b9")
    a1.set_xticks(ks)
    a2 = a1.twinx()
    a2.plot(ks, wm, "s--", color="#e67e22")
    a2.set_ylabel("MAE", color="#e67e22")
    a1.set_title("Effect of warm-up length on coverage")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_warmup.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    print(f"All 6 figures saved to {out_dir}/.")
