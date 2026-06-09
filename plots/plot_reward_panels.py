#!/usr/bin/env python
"""Compact publication figure: GRPO reward for all three models as panels (a)(b)(c).

2x3 grid: top row = total reward, bottom row = the six components; columns are
1.7B / 3B / 8B. x-axis normalized to [0,1] training progress; y shared per row;
one shared component legend. Double-column (\\textwidth) friendly.

"""
import argparse, os
import matplotlib.pyplot as plt
import pandas as pd
import wandb

import plotcfg

STEP = "train/global_step"
TOTAL = "train/rewards/reward_function/mean"
COMPONENTS = [
    ("reward/correctness", "Correctness"),
    ("reward/format", "Format"),
    ("reward/graph_utility", "Graph utility"),
    ("reward/graph_networkx", "Graph (NetworkX)"),
    ("reward/graph_diversity", "Graph diversity"),
    ("reward/graph_structure", "Graph structure"),
]
# Models (labels + run IDs) come from plot_config.yaml via plotcfg.models().


def ser(api, ent, proj, rid, key, maxstep=None, off=0):
    h = api.run(f"{ent}/{proj}/{rid}").history(keys=[STEP, key], samples=200000)
    if key not in h.columns:
        return pd.Series(dtype=float)
    h = h.dropna(subset=[STEP, key]).set_index(STEP).sort_index()
    s = h.groupby(level=0).mean()[key]
    if maxstep is not None:
        s = s[s.index <= maxstep]
    s.index = s.index + off
    return s


def model_series(api, ent, proj, ra, rb, key, max_step_b=1000):
    a = ser(api, ent, proj, ra, key)
    if not rb:
        return a
    off = int(ser(api, ent, proj, ra, key).index.max())
    b = ser(api, ent, proj, rb, key, maxstep=max_step_b, off=off)
    return pd.concat([a, b]).sort_index()


def smooth(s, w):
    return s.rolling(w, center=True, min_periods=1).mean() if w and w > 1 else s


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
    p.add_argument("--out", default="figures/reward_panels")
    args = p.parse_args()

    cfg = plotcfg.load(args.config)
    ent, proj = cfg["entity"], cfg["project"]
    MODELS = plotcfg.models(cfg, "grpo")
    api = wandb.Api()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    })
    comp_colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), sharex="col", sharey="row")
    for ax in axes.flat:
        ax.grid(True, color="0.85", lw=0.5)
        ax.set_axisbelow(True)

    comp_handles = None
    for j, mdl in enumerate(MODELS):
        label, ra, rb = mdl["panel"], mdl["ra"], mdl["rb"]
        # total
        tot = smooth(model_series(api, ent, proj, ra, rb, TOTAL), args.smooth)
        tot_raw = model_series(api, ent, proj, ra, rb, TOTAL)
        axes[0, j].plot(tot_raw.index, tot_raw.values, color="#3060a0", alpha=0.18, lw=0.8)
        axes[0, j].plot(tot.index, tot.values, color="#3060a0", lw=1.8)
        axes[0, j].set_title(label)
        # components
        handles = []
        for i, (key, clabel) in enumerate(COMPONENTS):
            cs = smooth(model_series(api, ent, proj, ra, rb, key), args.smooth)
            ln, = axes[1, j].plot(cs.index, cs.values, color=comp_colors[i], lw=1.4, label=clabel)
            handles.append(ln)
        comp_handles = handles
        print(f"{label}: total {tot.iloc[0]:.3f}->{tot.iloc[-1]:.3f}")

    axes[0, 0].set_ylabel("Total reward")
    axes[1, 0].set_ylabel("Reward component")
    for j in range(3):
        axes[1, j].set_ylim(0, 1.05)
    fig.supxlabel("Training step (GRPO)", y=0.10, fontsize=9)
    fig.legend(comp_handles, [h.get_label() for h in comp_handles], loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()
