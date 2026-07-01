#!/usr/bin/env python
"""Compact publication figure for SFT warm-start training.

Uses the `sft` block in the plotting YAML config. The block may be either a
single run:

  sft:
    label: Gemma 4 E4B SFT
    run: wb07zkae

or a list of runs:

  sft:
    - label: SFT step-600 source
      run: wb07zkae

Output: figures/sft_panels.{png,svg,pdf}
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import wandb

import plotcfg

STEP_CANDIDATES = ("train/global_step", "global_step", "_step")
METRICS = [
    ("train_loss", ("train/loss", "loss", "train_loss", "train/train_loss"), "Train loss", "Loss"),
    ("eval_loss", ("eval/loss", "eval_loss", "validation/loss", "train/eval_loss"), "Eval loss", "Loss"),
    ("learning_rate", ("train/learning_rate", "learning_rate", "train/lr", "lr"), "Learning rate", "LR"),
    ("grad_norm", ("train/grad_norm", "grad_norm", "train/global_grad_norm", "global_grad_norm"), "Gradient norm", "Grad norm"),
]
DEFAULT_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")
EXCLUDE_AUTO_COLUMNS = {
    "_runtime", "_timestamp", "_step", "train/global_step", "global_step", "epoch",
}


def sft_runs(cfg):
    if "sft" not in cfg:
        raise SystemExit("[plot_sft_panels] config has no `sft` block")
    entries = cfg["sft"]
    if isinstance(entries, dict):
        entries = [entries]
    out = []
    for i, entry in enumerate(entries):
        if "run" not in entry:
            raise SystemExit("[plot_sft_panels] each `sft` entry needs a `run` id")
        out.append({
            "label": entry.get("label", f"SFT run {i + 1}"),
            "run": entry["run"],
            "color": entry.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
        })
    return out


def history(api, ent, proj, rid, max_rows):
    # Use scan_history first. It is slower than history(), but it is exact and
    # avoids sampled-history surprises where sparse or custom metric keys are
    # omitted from the returned frame.
    run = api.run(f"{ent}/{proj}/{rid}")
    rows = []
    try:
        for i, row in enumerate(run.scan_history(page_size=1000)):
            if i >= max_rows:
                break
            rows.append(row)
    except Exception as exc:
        print(f"{rid}: scan_history failed, falling back to sampled history: {exc}")
    if rows:
        return pd.DataFrame(rows)
    return run.history(samples=max_rows)


def interesting_columns(h):
    needles = ("loss", "lr", "learning", "grad", "norm", "step", "epoch")
    return [c for c in sorted(map(str, h.columns)) if any(n in c.lower() for n in needles)]


def numeric_metric_columns(h):
    cols = []
    for col in h.columns:
        if str(col) in EXCLUDE_AUTO_COLUMNS or str(col).startswith("_"):
            continue
        s = pd.to_numeric(h[col], errors="coerce")
        if s.notna().sum() < 2:
            continue
        cols.append(str(col))
    return cols


def auto_metrics(h):
    cols = numeric_metric_columns(h)
    priority = ("loss", "learning", "lr", "grad", "norm", "eval")

    def score(col):
        name = col.lower()
        hits = [i for i, needle in enumerate(priority) if needle in name]
        return (min(hits) if hits else 99, col)

    selected = sorted(cols, key=score)[:4]
    return [(col, (col,), col, col) for col in selected]


def series_from_history(h, key_candidates):
    metric_key = next((k for k in key_candidates if k in h.columns), None)
    if metric_key is None:
        return pd.Series(dtype=float), None
    step_key = next((k for k in STEP_CANDIDATES if k in h.columns), None)
    if step_key is None:
        step_key = "_step" if "_step" in h.columns else None
    if step_key is None:
        return pd.Series(dtype=float), metric_key
    frame = h[[step_key, metric_key]].copy()
    frame[step_key] = pd.to_numeric(frame[step_key], errors="coerce")
    frame[metric_key] = pd.to_numeric(frame[metric_key], errors="coerce")
    s = frame.dropna(subset=[step_key, metric_key]).set_index(step_key).sort_index()
    if s.empty:
        return pd.Series(dtype=float), metric_key
    return s.groupby(level=0).mean()[metric_key], metric_key


def smooth(s, w):
    return s.rolling(w, center=True, min_periods=1).mean() if w and w > 1 else s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="YAML config (default plot_config.yaml or $PLOT_CONFIG)")
    p.add_argument("--smooth", type=int, default=15)
    p.add_argument("--out", default="figures/sft_panels")
    p.add_argument("--list-keys", action="store_true",
                   help="Print available W&B history columns for the SFT runs and exit.")
    p.add_argument("--max-rows", type=int, default=200000,
                   help="Maximum W&B history rows to scan per SFT run.")
    args = p.parse_args()

    cfg = plotcfg.load(args.config)
    ent, proj = cfg["entity"], cfg["project"]
    runs = sft_runs(cfg)
    api = wandb.Api()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    })
    handles = []
    labels = []
    histories = []
    metric_specs = list(METRICS)
    for run in runs:
        h = history(api, ent, proj, run["run"], args.max_rows)
        histories.append((run, h))
        if args.list_keys:
            print(f"\n{run['label']} ({run['run']}) columns:")
            cols = interesting_columns(h)
            if not cols:
                cols = sorted(map(str, h.columns))
            for col in cols:
                print(f"  {col}")
            print("\nNumeric metric candidates:")
            for col in numeric_metric_columns(h):
                print(f"  {col}")

    if args.list_keys:
        return

    found_preferred = False
    for _, h in histories:
        for _, key_candidates, _, _ in METRICS:
            s, _ = series_from_history(h, key_candidates)
            found_preferred = found_preferred or not s.empty
    if not found_preferred:
        for _, h in histories:
            metric_specs = auto_metrics(h)
            if metric_specs:
                print("No preferred SFT metric names found; auto-plotting numeric metrics:")
                for _, keys, _, _ in metric_specs:
                    print(f"  {keys[0]}")
                break
    if not metric_specs:
        metric_specs = list(METRICS)

    nplots = min(4, len(metric_specs))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=False)
    axes = list(axes.flat)
    for ax in axes:
        ax.grid(True, color="0.85", lw=0.5)
        ax.set_axisbelow(True)

    for run, h in histories:
        plotted = 0
        for ax, (_, key_candidates, title, ylabel) in zip(axes, metric_specs[:nplots]):
            s, found_key = series_from_history(h, key_candidates)
            if s.empty:
                continue
            plotted += 1
            raw = s
            sm = smooth(s, args.smooth)
            ax.plot(raw.index, raw.values, color=run["color"], alpha=0.18, lw=0.8)
            ln, = ax.plot(sm.index, sm.values, color=run["color"], lw=1.8, label=run["label"])
            if title == "Train loss":
                print(f"{run['label']} {found_key}: {sm.iloc[0]:.4g}->{sm.iloc[-1]:.4g}")
            if run["label"] not in labels:
                handles.append(ln)
                labels.append(run["label"])
        if plotted == 0:
            print(f"\n{run['label']} ({run['run']}): no configured SFT metrics found.")
            print("Available relevant columns:")
            for col in interesting_columns(h):
                print(f"  {col}")

    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for ax, panel, (_, _, title, ylabel) in zip(axes, panel_labels, metric_specs[:nplots]):
        ax.set_title(f"{panel} {title}")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training step")
        if not ax.lines:
            ax.text(0.5, 0.5, "metric not logged", ha="center", va="center",
                    transform=ax.transAxes, color="0.45")
    for ax in axes[nplots:]:
        ax.set_axis_off()

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(handles)),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08 if handles else 0, 1, 1))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()
