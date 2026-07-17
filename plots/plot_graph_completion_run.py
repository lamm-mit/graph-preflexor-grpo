#!/usr/bin/env python
"""Download and plot one graph-completion GRPO run from Weights & Biases.

The script is intentionally specific to the graph-completion trainer metrics. It
uses ``scan_history`` (rather than sampled W&B history), keeps one exact row per
trainer step, and summarizes noisy curves with non-overlapping step bins. Figure
lines are bin means and ribbons are +/- one standard error across logged steps in
the bin; faint traces show the unsmoothed observations where useful.

Example
-------
python plots/plot_graph_completion_run.py \
  --run lamm-mit/graph-completion-grpo/lm0bp1n8 \
  --max-step 1400 \
  --bin-size 25 \
  --out-dir plots/figures/graph_completion_lm0bp1n8_step1400 \
  --refresh
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import wandb


STEP = "train/global_step"
TOTAL_REWARD = "train/reward"

POSITIVE_WEIGHTS = {
    "train/graph_completion/format_parse": 0.10,
    "train/graph_completion/schema_structure": 0.10,
    "train/graph_completion/fixed_contract": 0.15,
    "train/graph_completion/node": 0.10,
    "train/graph_completion/edge": 0.15,
    "train/graph_completion/mode_primary": 0.15,
    "train/graph_completion/improvement": 0.10,
    "train/graph_completion/exact_match": 0.15,
}

POSITIVE_COMPONENTS = [
    ("train/graph_completion/format_parse", "Format / parse"),
    ("train/graph_completion/schema_structure", "Schema / structure"),
    ("train/graph_completion/fixed_contract", "Fixed-object contract"),
    ("train/graph_completion/node", "Node quality"),
    ("train/graph_completion/edge", "Edge quality"),
    ("train/graph_completion/mode_primary", "Mode-specific objective"),
    ("train/graph_completion/improvement", "Improvement"),
    ("train/graph_completion/exact_match", "Exact match"),
]

PENALTY_COMPONENTS = [
    ("train/graph_completion/penalty_dangling", "Dangling objects"),
    ("train/graph_completion/penalty_duplicate", "Duplicates"),
    ("train/graph_completion/penalty_excessive_length", "Excessive length"),
    ("train/graph_completion/penalty_forbidden_change", "Forbidden changes"),
    ("train/graph_completion/penalty_no_op", "No-op"),
    ("train/graph_completion/penalty_spurious_content", "Spurious content"),
    ("train/graph_completion/penalty_truncation", "Truncation"),
]

QUALITY_PANELS = [
    (
        "Payload and exactness",
        [
            ("train/graph_completion_metric/structural_validity", "Structurally valid"),
            ("train/graph_completion_metric/fixed_object_exact", "Fixed objects exact"),
            ("train/graph_completion_metric/payload_precision", "Payload precision"),
            ("train/graph_completion_metric/exact_canonical_match", "Exact graph"),
        ],
    ),
    (
        "Node reconstruction",
        [
            ("train/graph_completion_metric/node_f1", "F1"),
            ("train/graph_completion_metric/node_precision", "Precision"),
            ("train/graph_completion_metric/node_recall", "Recall"),
            ("train/graph_completion_metric/node_add_recall", "Addition recall"),
        ],
    ),
    (
        "Edge reconstruction",
        [
            ("train/graph_completion_metric/edge_f1", "F1"),
            ("train/graph_completion_metric/edge_precision", "Precision"),
            ("train/graph_completion_metric/edge_recall", "Recall"),
            ("train/graph_completion_metric/edge_add_recall", "Addition recall"),
        ],
    ),
    (
        "Relation repair and removal",
        [
            ("train/graph_completion_metric/relation_accuracy", "Relation accuracy"),
            ("train/graph_completion_metric/relation_repair_recall", "Repair recall"),
            ("train/graph_completion_metric/wrong_relation_removal_rate", "Wrong-relation removal"),
            ("train/graph_completion_metric/removal_rate", "Removal rate"),
        ],
    ),
]

SUMMARY_METRICS = [
    TOTAL_REWARD,
    "train/reward_std",
    "train/frac_reward_zero_std",
    "train/completions/mean_length",
    "train/completions/clipped_ratio",
    "train/completions/mean_terminated_length",
    "train/graph_completion_metric/node_f1",
    "train/graph_completion_metric/edge_f1",
    "train/graph_completion_metric/relation_accuracy",
    "train/graph_completion_metric/exact_canonical_match",
    "train/graph_completion_metric/fixed_object_exact",
    "train/graph_completion_metric/structural_validity",
    "train/grad_norm",
    "train/loss",
    "train/entropy",
    "train/step_time",
    "train/learning_rate",
    "train/sampling/importance_sampling_ratio/min",
    "train/sampling/importance_sampling_ratio/mean",
    "train/sampling/importance_sampling_ratio/max",
    "train/sampling/sampling_logp_difference/mean",
    "train/sampling/sampling_logp_difference/max",
] + [key for key, _ in POSITIVE_COMPONENTS + PENALTY_COMPONENTS]

COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#6A3D9A",  # purple
    "#333333",  # near black
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="lamm-mit/graph-completion-grpo/lm0bp1n8",
        help="W&B run path: entity/project/run_id.",
    )
    parser.add_argument("--max-step", type=int, default=1400)
    parser.add_argument(
        "--bin-size",
        type=int,
        default=25,
        help="Number of consecutive trainer steps per arithmetic-mean bin.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots/figures/graph_completion_lm0bp1n8_step1400"),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload exact W&B history even when the cached CSV exists.",
    )
    parser.add_argument("--png-dpi", type=int, default=450)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.85,
            "figure.dpi": 150,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def _numeric(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def download_exact_history(
    run_path: str,
    max_step: int,
    cache_path: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, dict]:
    """Return exact per-step training rows and small run metadata."""
    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path)
        frame = frame[pd.to_numeric(frame[STEP], errors="coerce") <= max_step]
        metadata_path = cache_path.with_name("run_metadata.json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        print(f"Loaded {len(frame):,} cached training rows from {cache_path}")
        return frame, metadata

    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    rows: list[dict] = []
    for row in run.scan_history(page_size=1000):
        reward = _numeric(row.get(TOTAL_REWARD))
        step = _numeric(row.get(STEP))
        if reward is None or step is None:
            continue
        if step > max_step:
            # scan_history is chronological, so no later training row is needed.
            break
        compact = {
            key: value
            for key, value in row.items()
            if key.startswith("train/") or key in {"_step", "_timestamp", "_runtime"}
        }
        rows.append(compact)

    if not rows:
        raise RuntimeError(f"No {TOTAL_REWARD!r} rows found for {run_path}")

    frame = pd.DataFrame(rows)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[STEP]).sort_values(STEP)
    # Averaging duplicates is lossless for the normal one-row-per-step case and
    # deterministic if a W&B retry happened to log the same trainer step twice.
    frame = frame.groupby(STEP, as_index=False).mean(numeric_only=True)
    frame = frame[frame[STEP] <= max_step]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)

    metadata = {
        "run_path": run_path,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "url": run.url,
        "requested_max_step": max_step,
        "downloaded_rows": int(len(frame)),
        "downloaded_min_step": int(frame[STEP].min()),
        "downloaded_max_step": int(frame[STEP].max()),
    }
    cache_path.with_name("run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Downloaded {len(frame):,} exact training rows to {cache_path}")
    return frame, metadata


def binned_stats(frame: pd.DataFrame, key: str, bin_size: int) -> pd.DataFrame:
    if key not in frame.columns:
        return pd.DataFrame(columns=["x", "mean", "std", "sem", "n", "bin_start", "bin_end"])
    work = frame[[STEP, key]].copy()
    work[STEP] = pd.to_numeric(work[STEP], errors="coerce")
    work[key] = pd.to_numeric(work[key], errors="coerce")
    work = work.dropna()
    if work.empty:
        return pd.DataFrame(columns=["x", "mean", "std", "sem", "n", "bin_start", "bin_end"])
    # Anchor bins at steps 1..bin_size, bin_size+1..2*bin_size, etc.
    work["bin"] = np.floor((np.maximum(work[STEP], 1) - 1) / bin_size).astype(int)
    grouped = work.groupby("bin")
    out = grouped.agg(
        mean=(key, "mean"),
        std=(key, "std"),
        n=(key, "count"),
    ).reset_index()
    # Use canonical bin coordinates so metrics with occasional missing values
    # align to the same rows in the exported wide table.
    out["bin_start"] = out["bin"] * bin_size + 1
    out["bin_end"] = (out["bin"] + 1) * bin_size
    # Plot at the bin end so an analysis through step N visibly reaches N.
    out["x"] = out["bin_end"]
    out = out.drop(columns="bin")
    out["std"] = out["std"].fillna(0.0)
    out["sem"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    return out


def save_binned_table(frame: pd.DataFrame, bin_size: int, path: Path) -> None:
    parts = []
    for key in sorted(column for column in frame.columns if column.startswith("train/") and column != STEP):
        stats = binned_stats(frame, key, bin_size)
        if stats.empty:
            continue
        stats = stats.set_index(["bin_start", "bin_end"])[["x", "mean", "sem", "n"]]
        stats.columns = pd.MultiIndex.from_product([[key], stats.columns])
        parts.append(stats)
    if not parts:
        return
    table = pd.concat(parts, axis=1).sort_index()
    table.columns = [f"{metric}__{stat}" for metric, stat in table.columns]
    # Consolidate blocks before reset_index inserts the two index columns.
    table = table.copy()
    table.reset_index().to_csv(path, index=False)
    print(f"Wrote {path}")


def style_axis(ax: plt.Axes, max_step: int, xlabel: bool = True) -> None:
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_step)
    tick_spacing = 200 if max_step >= 800 else max(50, int(math.ceil(max_step / 7 / 50) * 50))
    ticks = list(range(0, max_step + 1, tick_spacing))
    if not ticks or ticks[-1] != max_step:
        ticks.append(max_step)
    ax.set_xticks(ticks)
    if xlabel:
        ax.set_xlabel("Training step")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def panel_title(ax: plt.Axes, index: int, title: str) -> None:
    letter = chr(ord("a") + index)
    ax.set_title(f"({letter}) {title}", loc="left", pad=5, fontweight="bold")


def plot_binned_metric(
    ax: plt.Axes,
    frame: pd.DataFrame,
    key: str,
    label: str,
    color: str,
    bin_size: int,
    *,
    raw: bool = False,
    sem: bool = True,
    linewidth: float = 1.65,
) -> pd.DataFrame:
    stats = binned_stats(frame, key, bin_size)
    if stats.empty:
        return stats
    if raw:
        raw_frame = frame[[STEP, key]].dropna()
        ax.plot(
            raw_frame[STEP],
            raw_frame[key],
            color=color,
            alpha=0.10,
            linewidth=0.45,
            zorder=1,
        )
    x = stats["x"].to_numpy(dtype=float)
    mean = stats["mean"].to_numpy(dtype=float)
    err = stats["sem"].to_numpy(dtype=float)
    if sem:
        ax.fill_between(x, mean - err, mean + err, color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(x, mean, color=color, linewidth=linewidth, label=label, zorder=3)
    return stats


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        path = out_dir / f"{stem}.{extension}"
        kwargs = {"dpi": dpi} if extension == "png" else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        print(f"Wrote {path}")
    plt.close(fig)


def plot_training_overview(
    frame: pd.DataFrame,
    max_step: int,
    bin_size: int,
    out_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35), constrained_layout=True)
    axes = axes.flat

    ax = axes[0]
    plot_binned_metric(ax, frame, TOTAL_REWARD, "Shaped reward", COLORS[0], bin_size, raw=True)
    panel_title(ax, 0, "Total reward")
    ax.set_ylabel("Reward")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    mean_stats = plot_binned_metric(
        ax,
        frame,
        "train/completions/mean_length",
        "Mean length",
        COLORS[1],
        bin_size,
        raw=True,
    )
    min_stats = binned_stats(frame, "train/completions/min_length", bin_size)
    max_stats = binned_stats(frame, "train/completions/max_length", bin_size)
    if not mean_stats.empty and len(min_stats) == len(max_stats):
        ax.fill_between(
            mean_stats["x"].to_numpy(dtype=float),
            min_stats["mean"].to_numpy(dtype=float),
            max_stats["mean"].to_numpy(dtype=float),
            color=COLORS[1],
            alpha=0.09,
            linewidth=0,
            label="Mean rollout min-max",
        )
    panel_title(ax, 1, "Completion length")
    ax.set_ylabel("Tokens")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="best")

    ax = axes[2]
    for i, (key, label) in enumerate(
        [
            ("train/graph_completion/node", "Node score"),
            ("train/graph_completion/edge", "Edge score"),
            ("train/graph_completion/mode_primary", "Mode-specific score"),
            ("train/graph_completion/exact_match", "Exact graph"),
        ]
    ):
        plot_binned_metric(ax, frame, key, label, COLORS[i], bin_size)
    panel_title(ax, 2, "Task scores across all completions")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, ncol=2, loc="best")

    ax = axes[3]
    plot_binned_metric(
        ax,
        frame,
        "train/reward_std",
        "Within-group reward SD",
        COLORS[2],
        bin_size,
    )
    plot_binned_metric(
        ax,
        frame,
        "train/frac_reward_zero_std",
        "Fraction zero-variance groups",
        COLORS[4],
        bin_size,
    )
    panel_title(ax, 3, "GRPO reward signal")
    ax.set_ylabel("Value / fraction")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="best")

    for ax in axes:
        style_axis(ax, max_step)
    fig.suptitle(
        f"Graph-completion GRPO training dynamics (means in {bin_size}-step bins; ribbon = ±SEM)",
        fontsize=10.5,
    )
    save_figure(fig, out_dir, f"training_overview_step{max_step}", dpi)


def plot_graph_quality(
    frame: pd.DataFrame,
    max_step: int,
    bin_size: int,
    out_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35), constrained_layout=True)
    for panel_index, (ax, (title, metrics)) in enumerate(zip(axes.flat, QUALITY_PANELS)):
        for metric_index, (key, label) in enumerate(metrics):
            plot_binned_metric(
                ax,
                frame,
                key,
                label,
                COLORS[metric_index],
                bin_size,
                linewidth=1.5,
            )
        panel_title(ax, panel_index, title)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.03)
        ax.legend(frameon=False, ncol=2, loc="best")
        style_axis(ax, max_step)
    fig.suptitle(
        f"Conditional graph quality among valid parsed completions (means in {bin_size}-step bins; ribbon = ±SEM)",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        -0.01,
        "These metrics are emitted only when a completion parses to a schema-valid graph; unconditional task scores are in the overview and reward-component figures.",
        ha="center",
        fontsize=7.3,
        color="0.25",
    )
    save_figure(fig, out_dir, f"graph_quality_step{max_step}", dpi)


def _maybe_log_axis(ax: plt.Axes, values: Iterable[float], threshold: float = 30.0) -> None:
    finite = np.asarray([value for value in values if np.isfinite(value) and value > 0], dtype=float)
    if finite.size and finite.max() / max(np.median(finite), 1e-12) > threshold:
        ax.set_yscale("log")


def plot_optimization_diagnostics(
    frame: pd.DataFrame,
    max_step: int,
    bin_size: int,
    out_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.3), constrained_layout=True)
    axes = list(axes.flat)

    specs = [
        ("train/loss", "Policy loss", "Loss", COLORS[0]),
        ("train/grad_norm", "Gradient norm", "L2 norm", COLORS[1]),
        ("train/entropy", "Token entropy", "Entropy", COLORS[2]),
    ]
    for index, (key, title, ylabel, color) in enumerate(specs):
        stats = plot_binned_metric(axes[index], frame, key, title, color, bin_size, raw=True)
        panel_title(axes[index], index, title)
        axes[index].set_ylabel(ylabel)
        if key == "train/loss":
            axes[index].axhline(0, color="0.35", linewidth=0.6, linestyle="--")
        if key == "train/grad_norm" and not stats.empty:
            _maybe_log_axis(axes[index], stats["mean"])

    ax = axes[3]
    mean_stats = binned_stats(frame, "train/sampling/importance_sampling_ratio/mean", bin_size)
    min_stats = binned_stats(frame, "train/sampling/importance_sampling_ratio/min", bin_size)
    max_stats = binned_stats(frame, "train/sampling/importance_sampling_ratio/max", bin_size)
    if not mean_stats.empty:
        ax.plot(mean_stats["x"], mean_stats["mean"], color=COLORS[3], linewidth=1.65, label="Mean")
        if len(min_stats) == len(max_stats):
            ax.fill_between(
                mean_stats["x"].to_numpy(dtype=float),
                min_stats["mean"].to_numpy(dtype=float),
                max_stats["mean"].to_numpy(dtype=float),
                color=COLORS[3],
                alpha=0.16,
                linewidth=0,
                label="Binned min-max",
            )
        _maybe_log_axis(ax, max_stats["mean"] if not max_stats.empty else mean_stats["mean"], threshold=10.0)
    ax.axhline(1, color="0.35", linewidth=0.7, linestyle="--")
    panel_title(ax, 3, "Importance-sampling ratio")
    ax.set_ylabel("Ratio")
    ax.legend(frameon=False, loc="best")

    ax = axes[4]
    plot_binned_metric(ax, frame, "train/step_time", "Step time", COLORS[4], bin_size, raw=True)
    panel_title(ax, 4, "Training throughput")
    ax.set_ylabel("Seconds per step")
    ax.set_ylim(bottom=0)

    ax = axes[5]
    plot_binned_metric(ax, frame, "train/learning_rate", "Learning rate", COLORS[5], bin_size)
    panel_title(ax, 5, "Learning-rate schedule")
    ax.set_ylabel("Learning rate")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylim(bottom=0)

    for ax in axes:
        style_axis(ax, max_step)
    fig.suptitle(
        f"GRPO optimization diagnostics (means in {bin_size}-step bins; ribbon = ±SEM)",
        fontsize=10.5,
    )
    save_figure(fig, out_dir, f"optimization_diagnostics_step{max_step}", dpi)


def plot_reward_components(
    frame: pd.DataFrame,
    max_step: int,
    bin_size: int,
    out_dir: Path,
    dpi: int,
) -> None:
    components = POSITIVE_COMPONENTS + PENALTY_COMPONENTS
    penalty_upper = 0.02
    for key, _ in PENALTY_COMPONENTS:
        stats = binned_stats(frame, key, bin_size)
        if not stats.empty:
            penalty_upper = max(penalty_upper, float((stats["mean"] + stats["sem"]).max()) * 1.12)

    fig, axes = plt.subplots(4, 4, figsize=(10.6, 8.25), constrained_layout=True)
    axes = list(axes.flat)
    for index, (ax, (key, label)) in enumerate(zip(axes, components)):
        positive = index < len(POSITIVE_COMPONENTS)
        color = COLORS[index % len(COLORS)] if positive else COLORS[1]
        plot_binned_metric(ax, frame, key, label, color, bin_size, raw=True, linewidth=1.45)
        panel_title(ax, index, label)
        ax.set_ylabel("Component" if positive else "Penalty magnitude")
        if positive:
            ax.set_ylim(0, 1.03)
        else:
            ax.set_ylim(0, penalty_upper)
        style_axis(ax, max_step, xlabel=index >= 12)
        if index >= 12:
            ax.set_xticks(np.linspace(0, max_step, 5, dtype=int))
    axes[-1].set_axis_off()
    fig.suptitle(
        f"Complete shaped-reward decomposition (means in {bin_size}-step bins; ribbon = ±SEM)",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        -0.005,
        "Positive panels (a-h) show unweighted component scores; panels (i-o) show already-scaled penalty magnitudes.",
        ha="center",
        fontsize=7.5,
        color="0.25",
    )
    save_figure(fig, out_dir, f"reward_components_all_step{max_step}", dpi)


def plot_reward_change_decomposition(
    analysis: dict,
    out_dir: Path,
    dpi: int,
) -> None:
    impacts = analysis.get("reward_change_decomposition", [])
    if not impacts:
        return
    info = analysis["analysis"]
    initial_lo, initial_hi = info["initial_window"]
    final_lo, final_hi = info["final_window"]
    labels = [item["label"] for item in impacts]
    values = np.asarray([item["impact_on_total_reward_change"] for item in impacts])
    colors = ["#0072B2" if value >= 0 else "#D55E00" for value in values]
    y = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.barh(y, values, color=colors, alpha=0.92, height=0.72)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("Contribution to change in mean total reward")
    ax.set_title(
        f"Reward-change decomposition: steps {initial_lo}-{initial_hi} versus {final_lo}-{final_hi}",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", visible=False)
    max_abs = max(float(np.max(np.abs(values))), 1e-4)
    for yi, value in zip(y, values):
        offset = 0.012 * max_abs
        display_value = 0.0 if abs(value) < 5e-4 else value
        ax.text(
            value + offset if value >= 0 else offset,
            yi,
            f"{display_value:+.3f}" if display_value else "0.000",
            va="center",
            ha="left",
            fontsize=7.2,
        )
    fig.text(
        0.5,
        -0.01,
        "Positive component changes are multiplied by their configured reward weights; penalty changes enter with the opposite sign.",
        ha="center",
        fontsize=7.3,
        color="0.25",
    )
    save_figure(fig, out_dir, f"reward_change_decomposition_step{info['max_step']}", dpi)


def _window_mean(frame: pd.DataFrame, key: str, lo: int, hi: int) -> float | None:
    if key not in frame.columns:
        return None
    values = pd.to_numeric(frame.loc[frame[STEP].between(lo, hi), key], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def build_analysis(frame: pd.DataFrame, max_step: int, bin_size: int, metadata: dict) -> dict:
    min_step = int(frame[STEP].min())
    final_start = max(min_step, max_step - 99)
    initial_end = min(max_step, min_step + 99)
    metrics = {}
    for key in SUMMARY_METRICS:
        if key not in frame.columns:
            continue
        initial = _window_mean(frame, key, min_step, initial_end)
        final = _window_mean(frame, key, final_start, max_step)
        if initial is None or final is None:
            continue
        metrics[key] = {
            "initial_100_step_mean": initial,
            "final_100_step_mean": final,
            "absolute_change": final - initial,
            "relative_change_percent": ((final - initial) / abs(initial) * 100.0) if initial != 0 else None,
        }

    reward_bins = binned_stats(frame, TOTAL_REWARD, bin_size)
    reward_peak = {}
    if not reward_bins.empty:
        peak = reward_bins.loc[reward_bins["mean"].idxmax()]
        reward_peak = {
            "bin_start": int(peak["bin_start"]),
            "bin_end": int(peak["bin_end"]),
            "mean": float(peak["mean"]),
            "sem": float(peak["sem"]),
        }

    component_changes = []
    for key, label in POSITIVE_COMPONENTS + PENALTY_COMPONENTS:
        entry = metrics.get(key)
        if entry:
            component_changes.append(
                {
                    "metric": key,
                    "label": label,
                    "initial": entry["initial_100_step_mean"],
                    "final": entry["final_100_step_mean"],
                    "change": entry["absolute_change"],
                }
            )
    component_changes.sort(key=lambda item: item["change"], reverse=True)

    reward_change_decomposition = []
    for key, label in POSITIVE_COMPONENTS:
        entry = metrics.get(key)
        if entry:
            reward_change_decomposition.append(
                {
                    "metric": key,
                    "label": label,
                    "raw_component_change": entry["absolute_change"],
                    "multiplier": POSITIVE_WEIGHTS[key],
                    "impact_on_total_reward_change": entry["absolute_change"] * POSITIVE_WEIGHTS[key],
                }
            )
    for key, label in PENALTY_COMPONENTS:
        entry = metrics.get(key)
        if entry:
            reward_change_decomposition.append(
                {
                    "metric": key,
                    "label": f"Penalty: {label}",
                    "raw_component_change": entry["absolute_change"],
                    "multiplier": -1.0,
                    "impact_on_total_reward_change": -entry["absolute_change"],
                }
            )
    reward_change_decomposition.sort(
        key=lambda item: abs(item["impact_on_total_reward_change"]), reverse=True
    )
    accounted_change = sum(
        item["impact_on_total_reward_change"] for item in reward_change_decomposition
    )
    observed_change = metrics.get(TOTAL_REWARD, {}).get("absolute_change")

    return {
        "run": metadata,
        "analysis": {
            "min_step": min_step,
            "max_step": int(frame[STEP].max()),
            "rows": int(len(frame)),
            "bin_size": bin_size,
            "uncertainty_band": "± one standard error across logged trainer steps within each bin",
            "initial_window": [min_step, initial_end],
            "final_window": [final_start, max_step],
            "reward_peak_bin": reward_peak,
        },
        "metrics": metrics,
        "reward_component_changes": component_changes,
        "reward_change_decomposition": reward_change_decomposition,
        "reward_change_accounting": {
            "mean_component_accounted_change": accounted_change,
            "observed_total_reward_change": observed_change,
            "difference": (accounted_change - observed_change) if observed_change is not None else None,
            "note": "Mean-component accounting is approximate because the reward applies per-sample clipping and a fixed-object failure cap before averaging.",
        },
    }


def write_analysis_markdown(analysis: dict, path: Path) -> None:
    metrics = analysis["metrics"]
    info = analysis["analysis"]
    max_step = info["max_step"]
    rows = [
        f"# W&B run analysis through step {max_step}",
        "",
        f"Run: `{analysis.get('run', {}).get('run_path', 'unknown')}`",
        "",
        (
            f"Figures use arithmetic means in non-overlapping {info['bin_size']}-step bins. "
            "Ribbons show ± one standard error across the logged trainer steps in each bin; "
            "faint traces are the exact unsmoothed values. Bin means are plotted at each bin's ending step."
        ),
        "",
        "## Initial versus final 100-step means",
        "",
        "| Metric | Initial | Final | Change |",
        "|---|---:|---:|---:|",
    ]
    display_metrics = [
        (TOTAL_REWARD, "Total reward"),
        ("train/completions/mean_length", "Mean completion length"),
        ("train/reward_std", "Within-group reward SD"),
        ("train/frac_reward_zero_std", "Zero-variance group fraction"),
        ("train/graph_completion_metric/node_f1", "Node F1"),
        ("train/graph_completion_metric/edge_f1", "Edge F1"),
        ("train/graph_completion_metric/relation_accuracy", "Relation accuracy"),
        ("train/graph_completion_metric/exact_canonical_match", "Exact canonical match"),
        ("train/graph_completion_metric/fixed_object_exact", "Fixed objects exact"),
    ]
    for key, label in display_metrics:
        metric = metrics.get(key)
        if metric:
            rows.append(
                f"| {label} | {metric['initial_100_step_mean']:.4g} | "
                f"{metric['final_100_step_mean']:.4g} | {metric['absolute_change']:+.4g} |"
            )
    peak = info.get("reward_peak_bin", {})
    if peak:
        rows.extend(
            [
                "",
                "## Reward peak",
                "",
                (
                    f"The highest {info['bin_size']}-step mean reward through step {max_step} is "
                    f"{peak['mean']:.4f} ± {peak['sem']:.4f} in steps "
                    f"{peak['bin_start']}-{peak['bin_end']}."
                ),
            ]
        )
    decomposition = analysis.get("reward_change_decomposition", [])
    accounting = analysis.get("reward_change_accounting", {})
    total = metrics.get(TOTAL_REWARD, {})
    length = metrics.get("train/completions/mean_length", {})
    reward_std = metrics.get("train/reward_std", {})
    zero_std = metrics.get("train/frac_reward_zero_std", {})
    ratio_min = metrics.get("train/sampling/importance_sampling_ratio/min", {})
    ratio_mean = metrics.get("train/sampling/importance_sampling_ratio/mean", {})
    ratio_max = metrics.get("train/sampling/importance_sampling_ratio/max", {})
    clipped = metrics.get("train/completions/clipped_ratio", {})
    terminated = metrics.get("train/completions/mean_terminated_length", {})
    node_f1 = metrics.get("train/graph_completion_metric/node_f1", {})
    edge_f1 = metrics.get("train/graph_completion_metric/edge_f1", {})
    relation = metrics.get("train/graph_completion_metric/relation_accuracy", {})
    exact = metrics.get("train/graph_completion_metric/exact_canonical_match", {})
    rows.extend(["", "## Interpretation", ""])
    if total and length:
        rows.append(
            f"- Mean total reward increased by {total['absolute_change']:+.3f} "
            f"({total['relative_change_percent']:+.1f}%), while mean completion length changed "
            f"from {length['initial_100_step_mean']:.0f} to {length['final_100_step_mean']:.0f} tokens."
        )
    if decomposition:
        top = decomposition[:5]
        formatted = ", ".join(
            f"{item['label']} ({item['impact_on_total_reward_change']:+.3f})" for item in top
        )
        rows.append(f"- The largest mean-component contributions to the reward change were: {formatted}.")
    if accounting.get("observed_total_reward_change") is not None:
        rows.append(
            f"- Weighted mean-component accounting gives {accounting['mean_component_accounted_change']:+.3f}, "
            f"versus the observed total-reward change of {accounting['observed_total_reward_change']:+.3f}; "
            "the small difference is expected from per-sample clipping and the fixed-object failure cap."
        )
    if node_f1 and edge_f1 and relation and exact:
        rows.append(
            "- Conditional on a valid parsed graph, exact canonical match changed from "
            f"{exact['initial_100_step_mean']:.3f} to {exact['final_100_step_mean']:.3f}, edge F1 from "
            f"{edge_f1['initial_100_step_mean']:.3f} to {edge_f1['final_100_step_mean']:.3f}, node F1 from "
            f"{node_f1['initial_100_step_mean']:.3f} to {node_f1['final_100_step_mean']:.3f}, and relation "
            f"accuracy from {relation['initial_100_step_mean']:.3f} to {relation['final_100_step_mean']:.3f}."
        )
    if reward_std and zero_std:
        rows.append(
            f"- Within-group reward SD fell from {reward_std['initial_100_step_mean']:.3f} to "
            f"{reward_std['final_100_step_mean']:.3f}, while zero-variance groups rose from "
            f"{zero_std['initial_100_step_mean']:.2%} to {zero_std['final_100_step_mean']:.2%}; "
            f"the GRPO learning signal is therefore becoming sparser by step {max_step}."
        )
    if ratio_min and ratio_mean and ratio_max:
        rows.append(
            "- The mean token importance-sampling ratio remained near one "
            f"({ratio_mean['initial_100_step_mean']:.3f} to {ratio_mean['final_100_step_mean']:.3f}), "
            f"but its average min/max widened from {ratio_min['initial_100_step_mean']:.3f}/"
            f"{ratio_max['initial_100_step_mean']:.3f} to {ratio_min['final_100_step_mean']:.3f}/"
            f"{ratio_max['final_100_step_mean']:.3f}, close to the configured upper clip of 3.0."
        )
    if clipped and terminated:
        rows.append(
            f"- EOS telemetry reports clipped ratio {clipped['final_100_step_mean']:.1f} and mean terminated "
            f"length {terminated['final_100_step_mean']:.1f} throughout the final window. This should be "
            "treated as a tokenizer/stop-accounting warning, not as evidence that every generation used the "
            "full 4096-token budget, because the observed mean completion length is much shorter."
        )
    rows.extend(
        [
            "",
            "## Files",
            "",
            f"- `training_overview_step{max_step}.{{svg,png}}`",
            f"- `graph_quality_step{max_step}.{{svg,png}}`",
            f"- `optimization_diagnostics_step{max_step}.{{svg,png}}`",
            f"- `reward_components_all_step{max_step}.{{svg,png}}`",
            f"- `reward_change_decomposition_step{max_step}.{{svg,png}}`",
            f"- `history_raw_step{max_step}.csv` and `history_binned_step{max_step}.csv`",
            f"- `analysis_step{max_step}.json` and `run_metadata.json`",
            "",
        ]
    )
    path.write_text("\n".join(rows))
    print(f"Wrote {path}")


def print_key_findings(analysis: dict) -> None:
    metrics = analysis["metrics"]
    print("\nKey changes: first 100 logged steps -> final 100 steps")
    for key, label in [
        (TOTAL_REWARD, "reward"),
        ("train/completions/mean_length", "mean completion length"),
        ("train/graph_completion_metric/node_f1", "node F1"),
        ("train/graph_completion_metric/edge_f1", "edge F1"),
        ("train/graph_completion_metric/relation_accuracy", "relation accuracy"),
        ("train/graph_completion_metric/exact_canonical_match", "exact graph"),
        ("train/frac_reward_zero_std", "zero-variance groups"),
    ]:
        metric = metrics.get(key)
        if metric:
            print(
                f"  {label}: {metric['initial_100_step_mean']:.4g} -> "
                f"{metric['final_100_step_mean']:.4g} "
                f"({metric['absolute_change']:+.4g})"
            )
    peak = analysis["analysis"].get("reward_peak_bin", {})
    if peak:
        print(
            f"  peak binned reward: {peak['mean']:.4f} ± {peak['sem']:.4f} "
            f"at steps {peak['bin_start']}-{peak['bin_end']}"
        )


def main() -> None:
    args = parse_args()
    if args.max_step < 1:
        raise SystemExit("--max-step must be positive")
    if args.bin_size < 2:
        raise SystemExit("--bin-size must be at least 2")
    configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / f"history_raw_step{args.max_step}.csv"
    frame, metadata = download_exact_history(
        args.run,
        args.max_step,
        raw_path,
        args.refresh,
    )
    actual_max = int(frame[STEP].max())
    if actual_max < args.max_step:
        print(f"Warning: requested step {args.max_step}, but history ends at {actual_max}")
    frame = frame[frame[STEP] <= args.max_step].copy()

    save_binned_table(frame, args.bin_size, args.out_dir / f"history_binned_step{args.max_step}.csv")
    plot_training_overview(frame, args.max_step, args.bin_size, args.out_dir, args.png_dpi)
    plot_graph_quality(frame, args.max_step, args.bin_size, args.out_dir, args.png_dpi)
    plot_optimization_diagnostics(frame, args.max_step, args.bin_size, args.out_dir, args.png_dpi)
    plot_reward_components(frame, args.max_step, args.bin_size, args.out_dir, args.png_dpi)

    analysis = build_analysis(frame, args.max_step, args.bin_size, metadata)
    analysis_path = args.out_dir / f"analysis_step{args.max_step}.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {analysis_path}")
    plot_reward_change_decomposition(analysis, args.out_dir, args.png_dpi)
    write_analysis_markdown(analysis, args.out_dir / "README.md")
    print_key_findings(analysis)


if __name__ == "__main__":
    main()
