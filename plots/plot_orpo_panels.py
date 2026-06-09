#!/usr/bin/env python
"""Compact publication figure: ORPO cold start for all three models as panels (a)(b)(c).

2x3 grid: top row = loss + NLL; bottom row = preference accuracy + reward margin
(both in [0,1], single axis -- no twin). Columns 1.7B / 3B / 8B; x normalized to
[0,1] training progress; y shared per row; one shared legend. EMA smoothing
(trending curves). Double-column friendly.

"""
import argparse, os
import matplotlib.pyplot as plt
import wandb

import plotcfg

STEP = "train/global_step"
C_LOSS, C_NLL, C_ACC, C_MARGIN = "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"


def ser(api, ent, proj, rid, key):
    h = api.run(f"{ent}/{proj}/{rid}").history(keys=[STEP, key], samples=200000)
    if key not in h.columns:
        import pandas as pd
        return pd.Series(dtype=float)
    h = h.dropna(subset=[STEP, key]).set_index(STEP).sort_index()
    return h.groupby(level=0).mean()[key]


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean() if span and span > 1 else s


def norm_x(s):
    if len(s) > 1:
        lo, hi = s.index.min(), s.index.max()
        s = s.copy(); s.index = (s.index - lo) / (hi - lo)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="YAML config (default plot_config.yaml or $PLOT_CONFIG)")
    p.add_argument("--smooth", type=int, default=15)
    p.add_argument("--out", default="figures/orpo_panels")
    args = p.parse_args()

    cfg = plotcfg.load(args.config)
    ent, proj = cfg["entity"], cfg["project"]
    MODELS = plotcfg.models(cfg, "orpo")
    api = wandb.Api()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), sharex="col", sharey="row")
    for ax in axes.flat:
        ax.grid(True, color="0.85", lw=0.5)
        ax.set_axisbelow(True)
    handles = {}
    for j, m in enumerate(MODELS):
        label, rid = m["panel"], m["ra"]
        g = lambda k: ema(ser(api, ent, proj, rid, k), args.smooth)
        loss, nll = g("train/loss"), g("train/nll_loss")
        acc, margin = g("train/rewards/accuracies"), g("train/rewards/margins")
        h1, = axes[0, j].plot(loss.index, loss.values, color=C_LOSS, lw=1.8, label="ORPO loss")
        h2, = axes[0, j].plot(nll.index, nll.values, color=C_NLL, lw=1.6, label="NLL loss")
        h3, = axes[1, j].plot(acc.index, acc.values, color=C_ACC, lw=1.8, label="Preference accuracy")
        h4, = axes[1, j].plot(margin.index, margin.values, color=C_MARGIN, lw=1.8, label="Reward margin")
        handles = {"ORPO loss": h1, "NLL loss": h2, "Preference accuracy": h3, "Reward margin": h4}
        axes[0, j].set_title(label)

    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy / margin")
    for j in range(3):
        axes[1, j].set_ylim(0, 1.05)
    fig.supxlabel("Training step (ORPO)", y=0.085, fontsize=9)
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()
