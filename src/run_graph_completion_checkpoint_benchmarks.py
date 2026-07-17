#!/usr/bin/env python
"""Benchmark a labeled series of graph-completion checkpoints.

The runner composes ``sample_graph_completion.py`` and
``analyze_graph_completion_benchmark.py`` sequentially for each inference-ready
model in a JSON manifest. It then combines their benchmark summaries into
tables and publication-ready checkpoint-trend figures.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATASET = "lamm-mit/graph-canvas-inpainting-121k"
DEFAULT_METRICS = (
    "reward",
    "valid_completion",
    "exact_match",
    "fixed_contract",
    "node",
    "edge",
    "mode_primary",
)
METRIC_LABELS = {
    "reward": "Shaped reward",
    "valid_completion": "Valid completion",
    "exact_match": "Exact graph",
    "fixed_contract": "Fixed-object contract",
    "node": "Node score",
    "edge": "Edge score",
    "mode_primary": "Mode-specific score",
}
COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#333333",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


@dataclass(frozen=True)
class CheckpointSpec:
    step: int
    label: str
    model: str
    identifier: str
    revision: Optional[str] = None
    tokenizer_model: Optional[str] = None
    predictions: Optional[Path] = None
    analysis_dir: Optional[Path] = None


@dataclass(frozen=True)
class CheckpointPaths:
    root: Path
    predictions: Path
    generation_log: Path
    analysis_dir: Path
    analysis_log: Path


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "checkpoint"


def _optional_path(value: Any) -> Optional[Path]:
    return Path(str(value)).expanduser() if value else None


def load_checkpoint_manifest(path: Path) -> list[CheckpointSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("checkpoints") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must be a non-empty list or contain 'checkpoints'")
    specs: list[CheckpointSpec] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"manifest checkpoint {index} is not an object")
        missing = [
            name
            for name in ("step", "label", "model")
            if record.get(name) is None or not str(record.get(name)).strip()
        ]
        if missing:
            raise ValueError(
                f"manifest checkpoint {index} is missing fields: {missing}"
            )
        try:
            step = int(record["step"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"manifest checkpoint {index} has non-integer step"
            ) from exc
        label = str(record["label"]).strip()
        identifier = str(
            record.get("id") or f"step-{step}-{_slug(label)}"
        ).strip()
        specs.append(
            CheckpointSpec(
                step=step,
                label=label,
                model=str(record["model"]).strip(),
                identifier=_slug(identifier),
                revision=(
                    str(record["revision"]).strip()
                    if record.get("revision")
                    else None
                ),
                tokenizer_model=(
                    str(record["tokenizer_model"]).strip()
                    if record.get("tokenizer_model")
                    else None
                ),
                predictions=_optional_path(record.get("predictions")),
                analysis_dir=_optional_path(record.get("analysis_dir")),
            )
        )
    duplicate_steps = [
        step
        for step in sorted({spec.step for spec in specs})
        if sum(candidate.step == step for candidate in specs) > 1
    ]
    duplicate_ids = [
        identifier
        for identifier in sorted({spec.identifier for spec in specs})
        if sum(candidate.identifier == identifier for candidate in specs) > 1
    ]
    if duplicate_steps:
        raise ValueError(f"manifest checkpoint steps must be unique: {duplicate_steps}")
    if duplicate_ids:
        raise ValueError(f"manifest checkpoint ids must be unique: {duplicate_ids}")
    if len(specs) < 2:
        raise ValueError("checkpoint trend analysis requires at least two checkpoints")
    return sorted(specs, key=lambda spec: spec.step)


def checkpoint_paths(spec: CheckpointSpec, output_dir: Path) -> CheckpointPaths:
    root = output_dir / "checkpoints" / spec.identifier
    return CheckpointPaths(
        root=root,
        predictions=spec.predictions or root / "predictions.jsonl",
        generation_log=root / "generation.log",
        analysis_dir=spec.analysis_dir or root / "analysis",
        analysis_log=root / "analysis.log",
    )


def parse_metrics(value: str) -> list[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(metrics) - set(METRIC_LABELS)
    if unknown:
        raise ValueError(f"unknown checkpoint comparison metrics: {sorted(unknown)}")
    if not metrics:
        raise ValueError("--metrics must contain at least one metric")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output_dir",
        default="outputs/graph_completion/checkpoint-comparison",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help=(
            "Use validation for checkpoint selection. Test sweeps require "
            "--allow_test_checkpoint_sweep."
        ),
    )
    parser.add_argument(
        "--validation_manifest",
        default="outputs/graph_completion/full-validation-sources.json",
    )
    parser.add_argument("--validation_source_count", type=int, default=512)
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=0,
        help="Default: 512 for validation and 3641 for test.",
    )
    parser.add_argument("--modes", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--generation_batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.45)
    parser.add_argument("--chat_template_enable_thinking", default="true")
    parser.add_argument("--reward_stage", choices=["format", "shaped", "exact"], default="shaped")
    parser.add_argument(
        "--aggregation_unit",
        choices=["source", "task"],
        default="source",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument(
        "--primary_metric",
        choices=sorted(METRIC_LABELS),
        default="reward",
        help="Metric used for the focused checkpoint curve. Default: reward.",
    )
    parser.add_argument("--png_dpi", type=int, default=450)
    parser.add_argument(
        "--analysis_only",
        action="store_true",
        help="Skip model generation and analyze each existing prediction JSONL.",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip generation and analysis; rebuild comparison outputs from summaries.",
    )
    parser.add_argument(
        "--force_generation",
        action="store_true",
        help="Regenerate and truncate each managed prediction JSONL.",
    )
    parser.add_argument(
        "--allow_test_checkpoint_sweep",
        action="store_true",
        help="Explicitly acknowledge that comparing checkpoints on test consumes the test split.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print generation and analysis commands without executing them.",
    )
    return parser


def _resolved_num_tasks(args: argparse.Namespace) -> int:
    if args.num_tasks > 0:
        return args.num_tasks
    return 512 if args.split == "validation" else 3641


def generation_command(
    spec: CheckpointSpec,
    paths: CheckpointPaths,
    args: argparse.Namespace,
    *,
    resume: bool,
) -> list[str]:
    src = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-u",
        str(src / "sample_graph_completion.py"),
        "--model",
        spec.model,
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--validation_manifest",
        args.validation_manifest,
        "--validation_source_count",
        str(args.validation_source_count),
        "--invalid_pair_policy",
        "filter",
        "--num_tasks",
        str(_resolved_num_tasks(args)),
        "--num_generations",
        str(args.num_generations),
        "--generation_batch_size",
        str(args.generation_batch_size),
        "--stream_output_jsonl",
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--max_prompt_length",
        str(args.max_prompt_length),
        "--max_completion_length",
        str(args.max_completion_length),
        "--dtype",
        args.dtype,
        "--vllm_gpu_memory_utilization",
        str(args.vllm_gpu_memory_utilization),
        "--chat_template_enable_thinking",
        args.chat_template_enable_thinking,
        "--view",
        "raw",
        "--output_jsonl",
        str(paths.predictions),
    ]
    if spec.tokenizer_model:
        command.extend(["--tokenizer_model", spec.tokenizer_model])
    if spec.revision:
        command.extend(["--revision", spec.revision])
    if args.modes:
        command.extend(["--modes", args.modes])
    if resume:
        command.append("--resume_output_jsonl")
    return command


def analysis_command(
    spec: CheckpointSpec,
    paths: CheckpointPaths,
    args: argparse.Namespace,
) -> list[str]:
    src = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-u",
        str(src / "analyze_graph_completion_benchmark.py"),
        "--predictions",
        str(paths.predictions),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--validation_manifest",
        args.validation_manifest,
        "--validation_source_count",
        str(args.validation_source_count),
        "--invalid_pair_policy",
        "filter",
        "--seed",
        str(args.seed),
        "--max_completion_length",
        str(args.max_completion_length),
        "--reward_stage",
        args.reward_stage,
        "--aggregation_unit",
        args.aggregation_unit,
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--bootstrap_seed",
        str(args.bootstrap_seed),
        "--confidence",
        str(args.confidence),
        "--png_dpi",
        str(args.png_dpi),
        "--label",
        spec.label,
        "--output_dir",
        str(paths.analysis_dir),
    ]
    if args.split == "validation":
        command.extend(["--max_rows", str(_resolved_num_tasks(args))])
    if args.modes:
        command.extend(["--modes", args.modes])
    return command


def run_logged_command(
    command: list[str],
    log_path: Path,
    *,
    append: bool,
    dry_run: bool,
) -> None:
    print(f"[checkpoint benchmark] $ {shlex.join(command)}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a" if append else "w", encoding="utf-8") as handle:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"command failed with exit code {exc.returncode}; see {log_path}"
            ) from exc


def _metric_record(
    summary: Mapping[str, Any],
    metric: str,
    *,
    mode: str = "overall",
) -> Mapping[str, Any]:
    try:
        return summary["end_to_end"][mode][metric]
    except KeyError as exc:
        raise ValueError(
            f"benchmark summary is missing end_to_end/{mode}/{metric}"
        ) from exc


def collect_results(
    specs: list[CheckpointSpec],
    output_dir: Path,
    metrics: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    combined = []
    for spec in specs:
        paths = checkpoint_paths(spec, output_dir)
        summary_path = paths.analysis_dir / "benchmark_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"missing checkpoint summary for {spec.label}: {summary_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "step": spec.step,
            "label": spec.label,
            "model": spec.model,
            "identifier": spec.identifier,
            "predictions": str(paths.predictions),
            "analysis_dir": str(paths.analysis_dir),
        }
        for metric in metrics:
            record = _metric_record(summary, metric)
            row[metric] = float(record["mean"])
            row[f"{metric}_ci_low"] = float(record["ci_low"])
            row[f"{metric}_ci_high"] = float(record["ci_high"])
            row[f"{metric}_n"] = int(record["n"])
        generation = summary.get("generation", {}).get("overall", {})
        for name in (
            "mean_tokens",
            "median_tokens",
            "p95_tokens",
            "at_max_length_rate",
        ):
            value = generation.get(name)
            row[name] = math.nan if value is None else float(value)
        rows.append(row)
        combined.append(
            {
                "checkpoint": {
                    "step": spec.step,
                    "label": spec.label,
                    "model": spec.model,
                    "identifier": spec.identifier,
                },
                "summary_path": str(summary_path),
                "summary": summary,
            }
        )
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True), combined


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


def _checkpoint_tick_labels(frame: pd.DataFrame) -> list[str]:
    return [
        f"{int(row.step)}\n{textwrap.fill(str(row.label), width=16)}"
        for row in frame.itertuples()
    ]


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    for extension in ("svg", "png"):
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"dpi": dpi} if extension == "png" else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        print(f"[checkpoint benchmark] wrote {path}", flush=True)
    plt.close(fig)


def _ci_errors(frame: pd.DataFrame, metric: str) -> np.ndarray:
    mean = frame[metric].to_numpy(dtype=float)
    low = frame[f"{metric}_ci_low"].to_numpy(dtype=float)
    high = frame[f"{metric}_ci_high"].to_numpy(dtype=float)
    return np.maximum(0.0, np.vstack([mean - low, high - mean]))


def plot_primary_metric_curve(
    frame: pd.DataFrame,
    metric: str,
    output_dir: Path,
    dpi: int,
    *,
    aggregation_unit: str,
    split: str,
) -> None:
    x = frame["step"].to_numpy(dtype=float)
    y = frame[metric].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.3), constrained_layout=True)
    ax.errorbar(
        x,
        y,
        yerr=_ci_errors(frame, metric),
        color=COLORS[0],
        marker="o",
        markersize=5,
        linewidth=1.7,
        capsize=3,
        label=METRIC_LABELS[metric],
    )
    for step, value in zip(x, y):
        ax.annotate(
            f"{value:.3f}",
            (step, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
    ax.set_xticks(x, _checkpoint_tick_labels(frame))
    ax.set_xlabel("Training checkpoint step")
    ax.set_ylabel(
        f"{aggregation_unit.capitalize()}-macro {METRIC_LABELS[metric].lower()}"
    )
    ax.set_ylim(
        max(-1.03, float(frame[f"{metric}_ci_low"].min()) - 0.05),
        min(1.03, float(frame[f"{metric}_ci_high"].max()) + 0.08),
    )
    ax.set_title(
        f"Graph-completion checkpoint selection: {split} "
        f"{METRIC_LABELS[metric].lower()}",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", visible=False)
    _save_figure(fig, output_dir, f"checkpoint_{metric}_over_steps", dpi)


def plot_metric_curves(
    frame: pd.DataFrame,
    metrics: list[str],
    output_dir: Path,
    dpi: int,
    *,
    aggregation_unit: str,
) -> None:
    x = frame["step"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    for index, metric in enumerate(metrics):
        ax.errorbar(
            x,
            frame[metric],
            yerr=_ci_errors(frame, metric),
            color=COLORS[index % len(COLORS)],
            marker=MARKERS[index % len(MARKERS)],
            markersize=4.5,
            linewidth=1.35,
            capsize=2.3,
            label=METRIC_LABELS[metric],
        )
    ax.set_xticks(x, _checkpoint_tick_labels(frame))
    ax.set_xlabel("Training checkpoint step")
    ax.set_ylabel(f"{aggregation_unit.capitalize()}-macro metric")
    lowest = min(
        float(frame[f"{metric}_ci_low"].min())
        for metric in metrics
    )
    ax.set_ylim(min(0.0, lowest - 0.03), 1.03)
    ax.set_title(
        "Graph-completion quality over training checkpoints",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=min(3, len(metrics)), loc="lower right")
    ax.grid(axis="x", visible=False)
    _save_figure(fig, output_dir, "checkpoint_metrics_over_steps", dpi)


def write_report(
    path: Path,
    frame: pd.DataFrame,
    metrics: list[str],
    *,
    manifest: Path,
    split: str,
    primary_metric: str,
) -> None:
    lines = [
        "# Graph-completion checkpoint comparison",
        "",
        f"- Manifest: `{manifest}`",
        f"- Split: `{split}`",
        f"- Checkpoints: {len(frame)}",
        "",
        "| Step | Label | "
        + " | ".join(METRIC_LABELS[metric] for metric in metrics)
        + " |",
        "|---:|---|" + "|".join("---:" for _ in metrics) + "|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {int(row.step)} | {row.label} | "
            + " | ".join(f"{getattr(row, metric):.4f}" for metric in metrics)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `checkpoint_metrics.csv`",
            "- `checkpoint_comparison.json`",
            f"- `checkpoint_{primary_metric}_over_steps.{{svg,png}}`",
            "- `checkpoint_metrics_over_steps.{svg,png}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    manifest = Path(args.manifest).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    metrics = parse_metrics(args.metrics)
    if args.primary_metric not in metrics:
        metrics.insert(0, args.primary_metric)
    specs = load_checkpoint_manifest(manifest)
    num_tasks = _resolved_num_tasks(args)
    if args.num_generations != 1:
        raise SystemExit(
            "multi-checkpoint streaming currently requires --num_generations 1"
        )
    if args.split == "test" and not args.allow_test_checkpoint_sweep:
        raise SystemExit(
            "refusing a multi-checkpoint test sweep: use --split validation for "
            "checkpoint selection, or pass --allow_test_checkpoint_sweep to "
            "acknowledge test-set consumption"
        )
    if args.split == "test" and num_tasks != 3641:
        raise SystemExit(
            "the current official test sweep requires --num_tasks 3641"
        )
    if args.force_generation and (args.analysis_only or args.plot_only):
        raise SystemExit("--force_generation cannot be combined with skip modes")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_manifest = {
        "source_manifest": str(manifest),
        "dataset": args.dataset,
        "split": args.split,
        "num_tasks": num_tasks,
        "checkpoints": [
            {
                "step": spec.step,
                "label": spec.label,
                "model": spec.model,
                "identifier": spec.identifier,
                "revision": spec.revision,
                "predictions": str(checkpoint_paths(spec, output_dir).predictions),
                "analysis_dir": str(checkpoint_paths(spec, output_dir).analysis_dir),
            }
            for spec in specs
        ],
    }
    (output_dir / "resolved_manifest.json").write_text(
        json.dumps(resolved_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for index, spec in enumerate(specs, start=1):
        paths = checkpoint_paths(spec, output_dir)
        paths.root.mkdir(parents=True, exist_ok=True)
        print(
            f"[checkpoint benchmark] checkpoint {index}/{len(specs)}: "
            f"step={spec.step}, label={spec.label!r}",
            flush=True,
        )
        if not args.analysis_only and not args.plot_only:
            resume = paths.predictions.exists() and not args.force_generation
            run_logged_command(
                generation_command(spec, paths, args, resume=resume),
                paths.generation_log,
                append=resume,
                dry_run=args.dry_run,
            )
        if not args.plot_only:
            if not args.dry_run and not paths.predictions.exists():
                raise FileNotFoundError(
                    f"prediction JSONL does not exist: {paths.predictions}"
                )
            run_logged_command(
                analysis_command(spec, paths, args),
                paths.analysis_log,
                append=False,
                dry_run=args.dry_run,
            )
    if args.dry_run:
        print("[checkpoint benchmark] dry run complete", flush=True)
        return

    frame, combined = collect_results(specs, output_dir, metrics)
    frame.to_csv(output_dir / "checkpoint_metrics.csv", index=False)
    (output_dir / "checkpoint_comparison.json").write_text(
        json.dumps(
            {
                "manifest": resolved_manifest,
                "checkpoints": combined,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _configure_matplotlib()
    plot_primary_metric_curve(
        frame,
        args.primary_metric,
        output_dir,
        args.png_dpi,
        aggregation_unit=args.aggregation_unit,
        split=args.split,
    )
    plot_metric_curves(
        frame,
        metrics,
        output_dir,
        args.png_dpi,
        aggregation_unit=args.aggregation_unit,
    )
    write_report(
        output_dir / "README.md",
        frame,
        metrics,
        manifest=manifest,
        split=args.split,
        primary_metric=args.primary_metric,
    )
    print(
        f"[checkpoint benchmark] complete: checkpoints={len(frame)}, "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
