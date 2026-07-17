#!/usr/bin/env python
"""Score saved graph-completion predictions and make publication figures.

This CLI is deliberately separate from ``sample_graph_completion.py``:
sampling writes raw model rollouts, while this program performs deterministic
reference scoring, coverage checks, aggregation, confidence intervals, tables,
and plots without loading the model.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graph_completion_data import (
    DEFAULT_DATASET,
    graph_completion_task_key,
    prepare_graph_completion_reference_split,
)
from graph_completion_parsing import GRAPH_COMPLETION_MODES
from graph_completion_rewards import (
    add_reward_arguments,
    reward_config_from_args,
)
from validate_graph_completion import score_saved_predictions


MetricExtractor = Callable[[Mapping[str, Any]], Optional[float]]
MetricSpec = tuple[str, str, MetricExtractor]

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#333333",
}

MODE_LABELS = {
    "prior_empty": "Empty prior",
    "fixed_nodes_only": "Fixed nodes",
    "missing_edges": "Missing edges",
    "partial_subgraph": "Partial subgraph",
    "wrong_relations": "Wrong relations",
    "extra_edges": "Extra edges",
}

COMPONENT_LABELS = {
    "format_parse": "Format / parse",
    "schema_structure": "Schema / structure",
    "fixed_contract": "Fixed-object contract",
    "node": "Node score",
    "edge": "Edge score",
    "mode_primary": "Mode-specific score",
    "improvement": "Improvement",
    "exact_match": "Exact graph",
    "penalty_dangling": "Dangling objects",
    "penalty_duplicate": "Duplicates",
    "penalty_excessive_length": "Excessive length",
    "penalty_forbidden_change": "Forbidden changes",
    "penalty_no_op": "No-op",
    "penalty_spurious_content": "Spurious content",
    "penalty_truncation": "Truncation",
}

POSITIVE_COMPONENTS = (
    "format_parse",
    "schema_structure",
    "fixed_contract",
    "node",
    "edge",
    "mode_primary",
    "improvement",
    "exact_match",
)

PENALTY_COMPONENTS = (
    "penalty_dangling",
    "penalty_duplicate",
    "penalty_excessive_length",
    "penalty_forbidden_change",
    "penalty_no_op",
    "penalty_spurious_content",
    "penalty_truncation",
)

CONDITIONAL_METRICS = (
    ("node_precision", "Node precision"),
    ("node_recall", "Node recall"),
    ("node_f1", "Node F1"),
    ("edge_precision", "Edge precision"),
    ("edge_recall", "Edge recall"),
    ("edge_f1", "Edge F1"),
    ("relation_accuracy", "Relation accuracy"),
    ("relation_recall", "Relation recall"),
    ("fixed_object_exact", "Fixed objects exact"),
    ("payload_precision", "Payload precision"),
    ("exact_canonical_match", "Exact canonical graph"),
    ("structural_validity", "Structural validity"),
)


def _component(name: str) -> MetricExtractor:
    return lambda record: float(record.get("components", {}).get(name, 0.0))


def _conditional_metric(name: str) -> MetricExtractor:
    def extract(record: Mapping[str, Any]) -> Optional[float]:
        metrics = record.get("metrics", {})
        return float(metrics[name]) if name in metrics else None

    return extract


END_TO_END_METRICS: tuple[MetricSpec, ...] = (
    ("reward", "Shaped reward", lambda record: float(record["reward"])),
    (
        "valid_completion",
        "Valid completion",
        lambda record: float(not record.get("errors")),
    ),
    ("exact_match", "Exact graph", _component("exact_match")),
    ("fixed_contract", "Fixed-object contract", _component("fixed_contract")),
    ("node", "Node score", _component("node")),
    ("edge", "Edge score", _component("edge")),
    ("mode_primary", "Mode-specific score", _component("mode_primary")),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
    )
    parser.add_argument(
        "--validation_manifest",
        default="outputs/graph_completion/validation_source_indices.json",
    )
    parser.add_argument("--validation_source_count", type=int, default=512)
    parser.add_argument(
        "--invalid_pair_policy",
        choices=["filter", "error"],
        default="filter",
    )
    parser.add_argument("--modes", default=None)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Output directory. Default: a '<prediction-stem>-analysis' "
            "directory beside the prediction file."
        ),
    )
    parser.add_argument("--label", default=None, help="Model label used in figure titles.")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--aggregation_unit",
        choices=["source", "task"],
        default="source",
        help=(
            "Unit used for macro averaging and bootstrap resampling. 'source' "
            "(default) first averages correlated corruption variants belonging "
            "to the same source graph; 'task' treats every prediction row as "
            "independent."
        ),
    )
    parser.add_argument("--png_dpi", type=int, default=450)
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Analyze a partial prediction file instead of requiring full split coverage.",
    )
    parser.add_argument(
        "--no_scored_jsonl",
        action="store_true",
        help="Do not write the row-level scored JSONL.",
    )
    add_reward_arguments(parser)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"prediction file is empty: {path}")
    return records


def _prediction_integrity(
    predictions: list[Mapping[str, Any]],
    reference_rows: Any,
) -> dict[str, Any]:
    expected_keys = {graph_completion_task_key(row) for row in reference_rows}
    predicted_keys: set[tuple[str, str, str]] = set()
    missing_mode = 0
    record_ids: list[tuple[str, str, str, str]] = []
    for prediction in predictions:
        if prediction.get("mode") is None or not str(prediction.get("mode")).strip():
            missing_mode += 1
            continue
        key = graph_completion_task_key(prediction)
        predicted_keys.add(key)
        generation = str(prediction.get("generation_index", 1))
        record_ids.append((*key, generation))
    duplicate_records = len(record_ids) - len(set(record_ids))
    missing_expected = expected_keys - predicted_keys
    unexpected = predicted_keys - expected_keys
    return {
        "input_prediction_rows": len(predictions),
        "expected_unique_tasks": len(expected_keys),
        "predicted_unique_tasks": len(predicted_keys),
        "missing_mode_rows": missing_mode,
        "duplicate_task_generation_rows": duplicate_records,
        "missing_expected_tasks": len(missing_expected),
        "unexpected_tasks": len(unexpected),
        "prediction_modes": dict(
            sorted(Counter(str(row.get("mode")) for row in predictions).items())
        ),
        "missing_expected_examples": [":".join(key) for key in sorted(missing_expected)[:10]],
        "unexpected_examples": [":".join(key) for key in sorted(unexpected)[:10]],
    }


def _bootstrap_mean(
    values: Iterable[float],
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, math.nan, 0
    point = float(array.mean())
    if samples <= 0 or array.size == 1 or np.all(array == array[0]):
        return point, point, point, int(array.size)
    boot = np.empty(samples, dtype=float)
    chunk_size = max(1, min(256, int(2_000_000 / array.size)))
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        boot[start:stop] = array[indices].mean(axis=1)
    alpha = 0.5 * (1.0 - confidence)
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return point, float(low), float(high), int(array.size)


def _summarize_specs(
    records: list[dict[str, Any]],
    specs: Iterable[MetricSpec],
    *,
    scope: str,
    mode: str,
    conditioning: str,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
    aggregation_unit: str,
) -> list[dict[str, Any]]:
    rows = []
    for key, label, extractor in specs:
        values: list[float] = []
        values_by_source: dict[str, list[float]] = defaultdict(list)
        for record in records:
            value = extractor(record)
            if value is not None:
                values.append(value)
                values_by_source[str(record["source_index"])].append(value)
        unit_values = (
            [float(np.mean(source_values)) for source_values in values_by_source.values()]
            if aggregation_unit == "source"
            else values
        )
        point, low, high, count = _bootstrap_mean(
            unit_values,
            samples=samples,
            confidence=confidence,
            rng=rng,
        )
        rows.append(
            {
                "scope": scope,
                "mode": mode,
                "metric": key,
                "label": label,
                "conditioning": conditioning,
                "aggregation_unit": aggregation_unit,
                "mean": point,
                "ci_low": low,
                "ci_high": high,
                "n": count,
                "prediction_rows": len(values),
            }
        )
    return rows


def _group_records(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups = {"overall": records}
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_mode[str(record["mode"])].append(record)
    for mode in GRAPH_COMPLETION_MODES:
        if mode in by_mode:
            groups[mode] = by_mode[mode]
    for mode in sorted(set(by_mode) - set(GRAPH_COMPLETION_MODES)):
        groups[mode] = by_mode[mode]
    return groups


def _component_specs(names: Iterable[str]) -> tuple[MetricSpec, ...]:
    return tuple(
        (name, COMPONENT_LABELS.get(name, name), _component(name))
        for name in names
    )


def _summarize_generation(
    groups: Mapping[str, list[dict[str, Any]]],
    *,
    max_completion_length: int,
) -> pd.DataFrame:
    rows = []
    for mode, records in groups.items():
        counts = np.asarray(
            [
                int(record["completion_token_count"])
                for record in records
                if record.get("completion_token_count") is not None
            ],
            dtype=float,
        )
        valid = np.asarray([float(not record.get("errors")) for record in records])
        finish = Counter(str(record.get("finish_reason")) for record in records)
        rows.append(
            {
                "scope": "overall" if mode == "overall" else "mode",
                "mode": mode,
                "n": len(records),
                "rows_with_token_counts": int(counts.size),
                "mean_tokens": float(counts.mean()) if counts.size else math.nan,
                "p05_tokens": float(np.quantile(counts, 0.05)) if counts.size else math.nan,
                "p25_tokens": float(np.quantile(counts, 0.25)) if counts.size else math.nan,
                "median_tokens": float(np.quantile(counts, 0.50)) if counts.size else math.nan,
                "p75_tokens": float(np.quantile(counts, 0.75)) if counts.size else math.nan,
                "p95_tokens": float(np.quantile(counts, 0.95)) if counts.size else math.nan,
                "max_tokens": float(counts.max()) if counts.size else math.nan,
                "at_max_length_count": (
                    int((counts >= max_completion_length).sum()) if counts.size else 0
                ),
                "at_max_length_rate": (
                    float((counts >= max_completion_length).mean())
                    if counts.size
                    else math.nan
                ),
                "valid_completion_rate": float(valid.mean()) if valid.size else math.nan,
                "finish_stop_count": int(finish.get("stop", 0)),
                "finish_length_count": int(finish.get("length", 0)),
            }
        )
    return pd.DataFrame(rows)


def _frame_to_nested(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for _, row in frame.iterrows():
        mode = str(row["mode"])
        metric = str(row["metric"])
        result.setdefault(mode, {})[metric] = {
            "mean": None if pd.isna(row["mean"]) else float(row["mean"]),
            "ci_low": None if pd.isna(row["ci_low"]) else float(row["ci_low"]),
            "ci_high": None if pd.isna(row["ci_high"]) else float(row["ci_high"]),
            "n": int(row["n"]),
            "prediction_rows": int(row["prediction_rows"]),
            "conditioning": str(row["conditioning"]),
            "aggregation_unit": str(row["aggregation_unit"]),
        }
    return result


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.85,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def _save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    *,
    dpi: int,
) -> None:
    for extension in ("svg", "png"):
        path = out_dir / f"{stem}.{extension}"
        kwargs = {"dpi": dpi} if extension == "png" else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        print(f"[graph-completion benchmark] wrote {path}", flush=True)
    plt.close(fig)


def _errorbar_values(rows: pd.DataFrame) -> np.ndarray:
    return np.maximum(
        0.0,
        np.vstack(
        [
            rows["mean"].to_numpy() - rows["ci_low"].to_numpy(),
            rows["ci_high"].to_numpy() - rows["mean"].to_numpy(),
        ]
        ),
    )


def _plot_overall(
    end_to_end: pd.DataFrame,
    *,
    label: str,
    confidence: float,
    out_dir: Path,
    dpi: int,
) -> None:
    order = [key for key, _, _ in END_TO_END_METRICS]
    rows = (
        end_to_end[end_to_end["mode"] == "overall"]
        .set_index("metric")
        .loc[order]
        .reset_index()
    )
    rows = rows.iloc[::-1].reset_index(drop=True)
    aggregation_label = str(rows["aggregation_unit"].iloc[0]).capitalize()
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 4.25), constrained_layout=True)
    ax.barh(
        y,
        rows["mean"],
        xerr=_errorbar_values(rows),
        color=COLORS["blue"],
        alpha=0.9,
        height=0.68,
        error_kw={"elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8},
    )
    ax.set_yticks(y, rows["label"])
    ax.set_xlabel(
        f"{aggregation_label}-macro mean with {confidence:.0%} bootstrap CI"
    )
    ax.set_xlim(min(0.0, float(rows["ci_low"].min()) - 0.03), 1.03)
    ax.set_title(f"{label}: official benchmark overview", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    for yi, value in zip(y, rows["mean"]):
        ax.text(
            min(1.0, float(value) + 0.018),
            yi,
            f"{value:.3f}",
            va="center",
            fontsize=7.5,
        )
    _save_figure(fig, out_dir, "benchmark_overall", dpi=dpi)


def _plot_mode_heatmap(
    end_to_end: pd.DataFrame,
    *,
    label: str,
    out_dir: Path,
    dpi: int,
) -> None:
    metric_order = [key for key, _, _ in END_TO_END_METRICS]
    metric_labels = {key: text for key, text, _ in END_TO_END_METRICS}
    modes = [
        mode
        for mode in GRAPH_COMPLETION_MODES
        if mode in set(end_to_end["mode"])
    ]
    pivot = (
        end_to_end[end_to_end["mode"].isin(modes)]
        .pivot(index="mode", columns="metric", values="mean")
        .reindex(index=modes, columns=metric_order)
    )
    values = pivot.to_numpy(dtype=float)
    vmin = min(0.0, float(np.nanmin(values)))
    aggregation_label = str(
        end_to_end["aggregation_unit"].iloc[0]
    ).capitalize()
    fig, ax = plt.subplots(figsize=(8.2, 4.35), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=vmin, vmax=1.0)
    ax.set_xticks(
        np.arange(len(metric_order)),
        [metric_labels[key] for key in metric_order],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(
        np.arange(len(modes)),
        [MODE_LABELS.get(mode, mode) for mode in modes],
    )
    ax.set_title(f"{label}: end-to-end metrics by corruption mode", loc="left", fontweight="bold")
    ax.grid(False)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if not np.isfinite(value):
                continue
            color = "white" if value > 0.58 else "#222222"
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7.4,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(f"{aggregation_label}-macro mean")
    _save_figure(fig, out_dir, "benchmark_by_mode", dpi=dpi)


def _plot_conditional_prf(
    conditional: pd.DataFrame,
    *,
    label: str,
    confidence: float,
    out_dir: Path,
    dpi: int,
) -> None:
    modes = [
        mode
        for mode in GRAPH_COMPLETION_MODES
        if mode in set(conditional["mode"])
    ]
    aggregation_label = str(
        conditional["aggregation_unit"].iloc[0]
    ).capitalize()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    panels = [
        (
            "Node reconstruction",
            ("node_precision", "node_recall", "node_f1"),
        ),
        (
            "Edge reconstruction",
            ("edge_precision", "edge_recall", "edge_f1"),
        ),
    ]
    series_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    series_labels = ["Precision", "Recall", "F1"]
    x = np.arange(len(modes))
    offsets = np.linspace(-0.18, 0.18, 3)
    for panel_index, (ax, (title, metrics)) in enumerate(zip(axes, panels)):
        for metric, color, series_label, offset in zip(
            metrics,
            series_colors,
            series_labels,
            offsets,
        ):
            rows = (
                conditional[
                    (conditional["mode"].isin(modes))
                    & (conditional["metric"] == metric)
                ]
                .set_index("mode")
                .reindex(modes)
            )
            ax.errorbar(
                x + offset,
                rows["mean"],
                yerr=_errorbar_values(rows),
                fmt="o",
                markersize=4.3,
                color=color,
                linewidth=1.0,
                capsize=2.3,
                label=series_label,
            )
        ax.set_xticks(
            x,
            [MODE_LABELS.get(mode, mode) for mode in modes],
            rotation=25,
            ha="right",
        )
        ax.set_ylim(0, 1.03)
        ax.set_ylabel(f"Conditional {aggregation_label.lower()}-macro score")
        ax.set_title(f"({'ab'[panel_index]}) {title}", loc="left", fontweight="bold")
        ax.legend(frameon=False, ncol=3, loc="lower center")
    fig.suptitle(
        f"{label}: parsed-graph quality ({confidence:.0%} bootstrap CI)",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        -0.03,
        "These metrics condition on predictions that produced a parsed, schema-valid graph.",
        ha="center",
        fontsize=7.4,
        color="0.25",
    )
    _save_figure(fig, out_dir, "benchmark_conditional_precision_recall", dpi=dpi)


def _plot_reward_components(
    component_frame: pd.DataFrame,
    *,
    label: str,
    confidence: float,
    out_dir: Path,
    dpi: int,
) -> None:
    overall = component_frame[component_frame["mode"] == "overall"].set_index("metric")
    aggregation_label = str(
        component_frame["aggregation_unit"].iloc[0]
    ).capitalize()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
    panels = [
        (
            axes[0],
            POSITIVE_COMPONENTS,
            "Positive component scores",
            COLORS["blue"],
            "Unweighted score",
        ),
        (
            axes[1],
            PENALTY_COMPONENTS,
            "Penalty magnitudes",
            COLORS["vermillion"],
            "Already-scaled magnitude",
        ),
    ]
    for panel_index, (ax, names, title, color, xlabel) in enumerate(panels):
        rows = overall.loc[list(names)].reset_index().iloc[::-1].reset_index(drop=True)
        y = np.arange(len(rows))
        ax.barh(
            y,
            rows["mean"],
            xerr=_errorbar_values(rows),
            color=color,
            alpha=0.9,
            height=0.68,
            error_kw={"elinewidth": 0.8, "capsize": 2.2, "capthick": 0.8},
        )
        ax.set_yticks(y, rows["label"])
        ax.set_xlabel(
            f"{xlabel}; {aggregation_label.lower()}-macro; "
            f"{confidence:.0%} bootstrap CI"
        )
        if panel_index == 0:
            left = min(0.0, float(rows["ci_low"].min()) - 0.03)
            ax.set_xlim(left=left)
            ax.axvline(0.0, color="0.35", linewidth=0.7)
        else:
            ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"({'ab'[panel_index]}) {title}", loc="left", fontweight="bold")
        for yi, value in zip(y, rows["mean"]):
            offset = max(0.002, 0.015 * max(abs(rows["mean"]).max(), 0.01))
            ax.text(
                float(value) + offset,
                yi,
                f"{value:.3f}",
                va="center",
                fontsize=7.1,
            )
    fig.suptitle(f"{label}: shaped-reward decomposition", fontsize=10.5)
    _save_figure(fig, out_dir, "benchmark_reward_components", dpi=dpi)


def _plot_generation(
    generation: pd.DataFrame,
    end_to_end: pd.DataFrame,
    *,
    label: str,
    max_completion_length: int,
    confidence: float,
    out_dir: Path,
    dpi: int,
) -> None:
    modes = [
        mode
        for mode in GRAPH_COMPLETION_MODES
        if mode in set(generation["mode"])
    ]
    rows = generation.set_index("mode").loc[modes]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.2, 4.2),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
        constrained_layout=True,
    )

    stats = []
    for mode in modes:
        row = rows.loc[mode]
        stats.append(
            {
                "label": MODE_LABELS.get(mode, mode),
                "whislo": row["p05_tokens"],
                "q1": row["p25_tokens"],
                "med": row["median_tokens"],
                "q3": row["p75_tokens"],
                "whishi": row["p95_tokens"],
                "fliers": [],
            }
        )
    supports_orientation = "orientation" in inspect.signature(
        axes[0].bxp
    ).parameters
    bxp_kwargs: dict[str, Any] = {
        "orientation" if supports_orientation else "vert": (
            "horizontal" if supports_orientation else False
        )
    }
    axes[0].bxp(
        stats,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": COLORS["sky"], "alpha": 0.55, "linewidth": 0.8},
        medianprops={"color": COLORS["blue"], "linewidth": 1.4},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        **bxp_kwargs,
    )
    axes[0].axvline(
        max_completion_length,
        color=COLORS["vermillion"],
        linestyle="--",
        linewidth=1.0,
        label=f"Maximum ({max_completion_length})",
    )
    axes[0].set_xlabel("Completion tokens (5th-95th percentile)")
    axes[0].set_title("(a) Length distribution", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="lower right")

    valid_rows = (
        end_to_end[
            (end_to_end["mode"].isin(modes))
            & (end_to_end["metric"] == "valid_completion")
        ]
        .set_index("mode")
        .loc[modes]
    )
    x = np.arange(len(modes))
    axes[1].bar(
        x,
        valid_rows["mean"],
        yerr=_errorbar_values(valid_rows),
        color=COLORS["green"],
        alpha=0.9,
        error_kw={"elinewidth": 0.8, "capsize": 2.2},
    )
    axes[1].set_ylim(0, 1.03)
    axes[1].set_ylabel(f"Rate with {confidence:.0%} bootstrap CI")
    axes[1].set_title("(b) Valid completions", loc="left", fontweight="bold")

    at_limit = rows["at_max_length_rate"].to_numpy(dtype=float)
    axes[2].bar(x, at_limit, color=COLORS["vermillion"], alpha=0.9)
    axes[2].set_ylim(0, max(0.02, float(np.nanmax(at_limit)) * 1.22))
    axes[2].set_ylabel("Fraction of responses")
    axes[2].set_title("(c) Maximum-length stops", loc="left", fontweight="bold")
    for ax in axes[1:]:
        ax.set_xticks(
            x,
            [MODE_LABELS.get(mode, mode) for mode in modes],
            rotation=35,
            ha="right",
        )
    fig.suptitle(f"{label}: generation diagnostics", fontsize=10.5)
    _save_figure(fig, out_dir, "benchmark_generation_diagnostics", dpi=dpi)


def _write_markdown_report(
    path: Path,
    *,
    label: str,
    predictions: Path,
    split: str,
    confidence: float,
    bootstrap_samples: int,
    aggregation_unit: str,
    integrity: Mapping[str, Any],
    end_to_end: pd.DataFrame,
    generation: pd.DataFrame,
) -> None:
    overall = end_to_end[end_to_end["mode"] == "overall"].set_index("metric")
    overall_generation = generation[generation["mode"] == "overall"].iloc[0]
    lines = [
        f"# {label}: graph-completion benchmark",
        "",
        f"- Predictions: `{predictions}`",
        f"- Split: `{split}`",
        f"- Prediction rows: {integrity['input_prediction_rows']}",
        f"- Unique expected tasks: {integrity['expected_unique_tasks']}",
        f"- Missing expected tasks: {integrity['missing_expected_tasks']}",
        f"- Missing/ambiguous references: {integrity['predictions_without_reference']}/{integrity['predictions_with_ambiguous_reference']}",
        f"- Aggregation unit: `{aggregation_unit}`",
        (
            f"- Aggregation: macro mean with {confidence:.0%} percentile "
            f"bootstrap CIs ({bootstrap_samples} resamples)"
        ),
        "",
        "## End-to-end results",
        "",
        "| Metric | Mean | CI low | CI high | Units | Prediction rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, _, _ in END_TO_END_METRICS:
        row = overall.loc[key]
        lines.append(
            f"| {row['label']} | {row['mean']:.4f} | {row['ci_low']:.4f} | "
            f"{row['ci_high']:.4f} | {int(row['n'])} | "
            f"{int(row['prediction_rows'])} |"
        )
    lines.extend(
        [
            "",
            "## Generation diagnostics",
            "",
            f"- Mean completion length: {overall_generation['mean_tokens']:.1f} tokens",
            f"- Median completion length: {overall_generation['median_tokens']:.1f} tokens",
            f"- 95th percentile completion length: {overall_generation['p95_tokens']:.1f} tokens",
            f"- Maximum-length responses: {int(overall_generation['at_max_length_count'])} "
            f"({overall_generation['at_max_length_rate']:.2%})",
            "",
            "## Interpretation note",
            "",
            (
                "End-to-end component means include every completion and assign zero semantic "
                "component credit to invalid outputs. Conditional precision/recall metrics use "
                "only outputs that produced a parsed, schema-valid graph."
            ),
            (
                "With source aggregation, corruption modes and variants from the same original "
                "graph are averaged before sources are weighted equally."
            ),
            "",
            "## Figures",
            "",
            "- `benchmark_overall.{svg,png}`",
            "- `benchmark_by_mode.{svg,png}`",
            "- `benchmark_conditional_precision_recall.{svg,png}`",
            "- `benchmark_reward_components.{svg,png}`",
            "- `benchmark_generation_diagnostics.{svg,png}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap_samples must be non-negative")
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("--confidence must be between zero and one")
    predictions_path = Path(args.predictions).expanduser()
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else predictions_path.with_name(f"{predictions_path.stem}-analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or predictions_path.stem
    modes = (
        [item.strip() for item in args.modes.split(",") if item.strip()]
        if args.modes
        else None
    )

    print(f"[graph-completion benchmark] reading {predictions_path}", flush=True)
    predictions = _read_jsonl(predictions_path)
    selected = prepare_graph_completion_reference_split(
        args.dataset,
        split=args.split,
        validation_manifest=args.validation_manifest,
        invalid_pair_policy=args.invalid_pair_policy,
        seed=args.seed,
        validation_source_count=args.validation_source_count,
        modes=modes,
        max_rows=args.max_rows if args.max_rows > 0 else None,
    )
    integrity = _prediction_integrity(predictions, selected)
    blocking_integrity = {
        key: integrity[key]
        for key in (
            "missing_mode_rows",
            "duplicate_task_generation_rows",
            "missing_expected_tasks",
            "unexpected_tasks",
        )
        if integrity[key]
    }
    if blocking_integrity and not args.allow_incomplete:
        raise SystemExit(
            "prediction coverage is not a complete, unique benchmark; "
            f"use --allow_incomplete only for a diagnostic partial run: {blocking_integrity}"
        )

    print(
        "[graph-completion benchmark] scoring "
        f"{len(predictions):,} response(s) against {len(selected):,} reference task(s)",
        flush=True,
    )
    scored, missing, ambiguous = score_saved_predictions(
        selected,
        predictions,
        reward_config_from_args(args),
    )
    integrity["predictions_scored"] = len(scored)
    integrity["predictions_without_reference"] = missing
    integrity["predictions_with_ambiguous_reference"] = ambiguous
    if (missing or ambiguous) and not args.allow_incomplete:
        raise SystemExit(
            f"reference matching failed: missing={missing}, ambiguous={ambiguous}"
        )
    if not scored:
        raise SystemExit("no predictions could be scored")

    if not args.no_scored_jsonl:
        scored_path = output_dir / "scored_predictions.jsonl"
        with scored_path.open("w", encoding="utf-8") as handle:
            for record in scored:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[graph-completion benchmark] wrote {scored_path}", flush=True)

    rng = np.random.default_rng(args.bootstrap_seed)
    groups = _group_records(scored)
    end_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    conditional_specs = tuple(
        (key, label_text, _conditional_metric(key))
        for key, label_text in CONDITIONAL_METRICS
    )
    component_specs = _component_specs(POSITIVE_COMPONENTS + PENALTY_COMPONENTS)
    for mode, records in groups.items():
        scope = "overall" if mode == "overall" else "mode"
        end_rows.extend(
            _summarize_specs(
                records,
                END_TO_END_METRICS,
                scope=scope,
                mode=mode,
                conditioning="all_completions",
                samples=args.bootstrap_samples,
                confidence=args.confidence,
                rng=rng,
                aggregation_unit=args.aggregation_unit,
            )
        )
        conditional_rows.extend(
            _summarize_specs(
                records,
                conditional_specs,
                scope=scope,
                mode=mode,
                conditioning="parsed_schema_valid_graph",
                samples=args.bootstrap_samples,
                confidence=args.confidence,
                rng=rng,
                aggregation_unit=args.aggregation_unit,
            )
        )
        component_rows.extend(
            _summarize_specs(
                records,
                component_specs,
                scope=scope,
                mode=mode,
                conditioning="all_completions",
                samples=args.bootstrap_samples,
                confidence=args.confidence,
                rng=rng,
                aggregation_unit=args.aggregation_unit,
            )
        )

    end_frame = pd.DataFrame(end_rows)
    conditional_frame = pd.DataFrame(conditional_rows)
    component_frame = pd.DataFrame(component_rows)
    generation_frame = _summarize_generation(
        groups,
        max_completion_length=args.max_completion_length,
    )
    table_outputs = (
        (end_frame, output_dir / "end_to_end_metrics.csv"),
        (conditional_frame, output_dir / "conditional_graph_metrics.csv"),
        (component_frame, output_dir / "reward_components.csv"),
        (generation_frame, output_dir / "generation_metrics.csv"),
    )
    for frame, path in table_outputs:
        frame.to_csv(path, index=False)
        print(f"[graph-completion benchmark] wrote {path}", flush=True)

    summary = {
        "label": label,
        "predictions": str(predictions_path),
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "max_completion_length": args.max_completion_length,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "confidence": args.confidence,
            "unit": args.aggregation_unit,
            "estimator": f"{args.aggregation_unit}-macro mean",
        },
        "integrity": integrity,
        "end_to_end": _frame_to_nested(end_frame),
        "conditional_graph_metrics": _frame_to_nested(conditional_frame),
        "reward_components": _frame_to_nested(component_frame),
        "generation": {
            str(row["mode"]): {
                key: (
                    None
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                )
                for key, value in row.items()
                if key not in {"scope", "mode"}
            }
            for row in generation_frame.to_dict(orient="records")
        },
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[graph-completion benchmark] wrote {summary_path}", flush=True)

    _configure_matplotlib()
    _plot_overall(
        end_frame,
        label=label,
        confidence=args.confidence,
        out_dir=output_dir,
        dpi=args.png_dpi,
    )
    _plot_mode_heatmap(
        end_frame,
        label=label,
        out_dir=output_dir,
        dpi=args.png_dpi,
    )
    _plot_conditional_prf(
        conditional_frame,
        label=label,
        confidence=args.confidence,
        out_dir=output_dir,
        dpi=args.png_dpi,
    )
    _plot_reward_components(
        component_frame,
        label=label,
        confidence=args.confidence,
        out_dir=output_dir,
        dpi=args.png_dpi,
    )
    _plot_generation(
        generation_frame,
        end_frame,
        label=label,
        max_completion_length=args.max_completion_length,
        confidence=args.confidence,
        out_dir=output_dir,
        dpi=args.png_dpi,
    )
    report_path = output_dir / "README.md"
    _write_markdown_report(
        report_path,
        label=label,
        predictions=predictions_path,
        split=args.split,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        aggregation_unit=args.aggregation_unit,
        integrity=integrity,
        end_to_end=end_frame,
        generation=generation_frame,
    )
    print(f"[graph-completion benchmark] wrote {report_path}", flush=True)

    overall = end_frame[end_frame["mode"] == "overall"].set_index("metric")
    print(
        "[graph-completion benchmark] complete: "
        f"reward={overall.loc['reward', 'mean']:.4f}, "
        f"valid={overall.loc['valid_completion', 'mean']:.2%}, "
        f"exact={overall.loc['exact_match', 'mean']:.2%}, "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
