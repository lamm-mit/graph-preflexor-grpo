#!/usr/bin/env python
"""Compact publication figure: GRPO generation + optimization dynamics.

2x2 grid, columns are panels (a) and (b):
  (a) reasoning-trace length:   top mean terminated length, bottom truncated %
  (b) optimization diagnostics: top within-group reward std, bottom policy entropy
One line per GRPO entry in the YAML config, x normalized to [0,1] training
progress, one shared model legend. Double-column friendly.

Output: figures/grpo_length_diag.{png,svg,pdf}
"""
import argparse, os
import matplotlib.pyplot as plt
import pandas as pd
import wandb

import plotcfg

STEP = "train/global_step"
# Models (labels + run IDs + colors) come from plot_config.yaml via plotcfg.models().


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
        mode = "absolute" if b_min > 0 and b_min <= a_max + max(10.0, 0.05 * a_max) else "offset"

    if mode == "absolute":
        b = b_raw
        if label:
            print(f"{label}: continuation uses absolute steps ({ra} max={a_max:.0f}, {rb} {b_min:.0f}->{b_max:.0f})")
    elif mode == "offset":
        b = b_raw.copy()
        b.index = b.index + int(a_max)
        if label:
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
    p.add_argument("--smooth", type=int, default=11)
    p.add_argument("--out", default="figures/grpo_length_diag")
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
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), sharex=True)
    for ax in axes.flat:
        ax.grid(True, color="0.85", lw=0.5)
        ax.set_axisbelow(True)
    # panel -> (axis, metric key, scale)
    spec = {
        (0, 0): ("train/completions/mean_terminated_length", 1.0),
        (1, 0): ("train/completions/clipped_ratio", 100.0),
        (0, 1): ("train/reward_std", 1.0),
        (1, 1): ("train/entropy", 1.0),
    }
    handles = []
    for mdl in MODELS:
        label, ra, rb, color = mdl["label"], mdl["ra"], mdl["rb"], mdl["color"]
        max_step_b = mdl.get("max_step_b")
        continuation_mode = args.continuation_mode or mdl.get("continuation_mode", "auto")
        for (r, c), (key, scale) in spec.items():
            s = norm_x(smooth(model_series(api, ent, proj, ra, rb, key,
                                           max_step_b=max_step_b,
                                           continuation_mode=continuation_mode,
                                           label=label if (r, c) == (0, 0) else None),
                              args.smooth))
            ln, = axes[r, c].plot(s.index, s.values * scale, color=color, lw=1.8, label=label)
            if (r, c) == (0, 0):
                handles.append(ln)

    axes[0, 0].set_title("(a) Reasoning-trace length")
    axes[0, 1].set_title("(b) Optimization diagnostics")
    axes[0, 0].set_ylabel("Mean terminated\nlength (tokens)")
    axes[1, 0].set_ylabel("Truncated\ncompletions (%)")
    axes[0, 1].set_ylabel("Within-group\nreward std")
    axes[1, 1].set_ylabel("Policy entropy")
    axes[1, 0].set_ylim(bottom=0)
    fig.supxlabel("Training progress (normalized)", y=0.075, fontsize=9)
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=min(3, max(1, len(handles))), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()
