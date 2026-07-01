#!/usr/bin/env python
"""Compact publication figure: GRPO reward panels.

Top row = total reward, bottom row = the six components. Columns are determined
from the number of GRPO entries in the YAML config.

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


def model_series(api, ent, proj, ra, rb, key, max_step_b=None, continuation_mode="auto", label=None):
    a = ser(api, ent, proj, ra, key)
    if not rb:
        return a
    b_raw = ser(api, ent, proj, rb, key, maxstep=max_step_b)
    if a.empty:
        return b_raw
    if b_raw.empty:
        return a

    a_max = float(a.index.max())
    b_min = float(b_raw.index.min())
    b_max = float(b_raw.index.max())
    mode = continuation_mode or "auto"
    if mode == "auto":
        # Trainer resumes usually log absolute global steps (e.g. restart from
        # checkpoint-600 starts near step 600). Fresh W&B continuation runs often
        # restart at 0 and need an offset.
        mode = "absolute" if b_min > 0 and b_min <= a_max + max(10.0, 0.05 * a_max) else "offset"

    if mode == "absolute":
        b = b_raw
        if label and key == TOTAL:
            print(f"{label}: continuation uses absolute steps ({ra} max={a_max:.0f}, {rb} {b_min:.0f}->{b_max:.0f})")
    elif mode == "offset":
        b = b_raw.copy()
        b.index = b.index + int(a_max)
        if label and key == TOTAL:
            print(f"{label}: continuation offset by {int(a_max)} ({rb} raw {b_min:.0f}->{b_max:.0f})")
    else:
        raise ValueError(f"unknown continuation mode {continuation_mode!r}")
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
    p.add_argument("--continuation-mode", choices=("auto", "absolute", "offset"), default=None,
                   help="Override GRPO continuation handling for all models.")
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
    ncols = max(1, len(MODELS))
    fig_width = max(4.8, 3.2 * ncols)
    fig, axes = plt.subplots(2, ncols, figsize=(fig_width, 4.5),
                             sharex="col", sharey="row", squeeze=False)
    for ax in axes.flat:
        ax.grid(True, color="0.85", lw=0.5)
        ax.set_axisbelow(True)

    comp_handles = None
    for j, mdl in enumerate(MODELS):
        label, ra, rb = mdl["panel"], mdl["ra"], mdl["rb"]
        max_step_b = mdl.get("max_step_b")
        continuation_mode = args.continuation_mode or mdl.get("continuation_mode", "auto")
        # total
        tot_raw = model_series(api, ent, proj, ra, rb, TOTAL,
                               max_step_b=max_step_b,
                               continuation_mode=continuation_mode,
                               label=label)
        tot = smooth(tot_raw, args.smooth)
        axes[0, j].plot(tot_raw.index, tot_raw.values, color="#3060a0", alpha=0.18, lw=0.8)
        axes[0, j].plot(tot.index, tot.values, color="#3060a0", lw=1.8)
        axes[0, j].set_title(label)
        # components
        handles = []
        for i, (key, clabel) in enumerate(COMPONENTS):
            cs = smooth(model_series(api, ent, proj, ra, rb, key,
                                     max_step_b=max_step_b,
                                     continuation_mode=continuation_mode), args.smooth)
            ln, = axes[1, j].plot(cs.index, cs.values, color=comp_colors[i], lw=1.4, label=clabel)
            handles.append(ln)
        comp_handles = handles
        if not tot.empty:
            print(f"{label}: total {tot.iloc[0]:.3f}->{tot.iloc[-1]:.3f}")

    axes[0, 0].set_ylabel("Total reward")
    axes[1, 0].set_ylabel("Reward component")
    for j in range(ncols):
        axes[1, j].set_ylim(0, 1.05)
    fig.supxlabel("Training step (GRPO)", y=0.10, fontsize=9)
    if comp_handles:
        fig.legend(comp_handles, [h.get_label() for h in comp_handles], loc="lower center",
                   ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()
