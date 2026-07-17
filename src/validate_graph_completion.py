#!/usr/bin/env python
"""Audit graph-completion data and optionally score saved completion JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Optional

from graph_completion_data import (
    DEFAULT_DATASET,
    graph_completion_task_key,
    prepare_graph_completion_datasets,
)
from graph_completion_rewards import (
    RewardConfig,
    add_reward_arguments,
    reward_config_from_args,
    score_graph_completion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--validation_manifest", default="outputs/graph_completion/validation_source_indices.json")
    parser.add_argument("--validation_source_count", type=int, default=512)
    parser.add_argument("--invalid_pair_policy", choices=["filter", "error"], default="filter")
    parser.add_argument("--modes", default=None)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predictions",
        default=None,
        help=(
            "Optional JSONL with source_index, variant_index, mode, and "
            "raw_completion/completion. Mode may be omitted only when the "
            "source/variant pair has one unambiguous reference row."
        ),
    )
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    add_reward_arguments(parser)
    return parser


def _averages(records: list[dict[str, Any]]) -> dict[str, float]:
    names = sorted(set().union(*(record.get("metrics", {}) for record in records)))
    result = {"reward": mean(record["reward"] for record in records)} if records else {}
    for name in names:
        values = [record["metrics"][name] for record in records if name in record["metrics"]]
        if values:
            result[name] = mean(values)
    return result


def _component_averages(records: list[dict[str, Any]]) -> dict[str, float]:
    names = sorted(set().union(*(record.get("components", {}) for record in records)))
    if not records:
        return {}
    return {
        name: mean(
            float(record.get("components", {}).get(name, 0.0))
            for record in records
        )
        for name in names
    }


def _prediction_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    conditional = _averages(records)
    conditional.pop("reward", None)
    return {
        "rows": len(records),
        "reward": mean(record["reward"] for record in records) if records else None,
        "valid_completion_rate": (
            mean(not record.get("errors") for record in records) if records else None
        ),
        "graph_metric_coverage": (
            mean(bool(record.get("metrics")) for record in records) if records else None
        ),
        "component_means_all_completions": _component_averages(records),
        "metrics_conditional_on_parsed_graph": conditional,
    }


def _generation_summary(
    records: list[dict[str, Any]],
    *,
    max_completion_length: int,
) -> dict[str, Any]:
    token_counts = [
        int(record["completion_token_count"])
        for record in records
        if record.get("completion_token_count") is not None
    ]
    finish_reasons = Counter(
        str(record.get("finish_reason"))
        for record in records
        if "finish_reason" in record
    )
    stop_reasons = Counter(
        str(record.get("stop_reason"))
        for record in records
        if "stop_reason" in record
    )
    return {
        "rows_with_token_counts": len(token_counts),
        "mean_completion_tokens": mean(token_counts) if token_counts else None,
        "max_completion_tokens": max(token_counts) if token_counts else None,
        "at_max_completion_length_count": (
            sum(value >= max_completion_length for value in token_counts)
            if token_counts
            else 0
        ),
        "at_max_completion_length_rate": (
            mean(value >= max_completion_length for value in token_counts)
            if token_counts
            else None
        ),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "stop_reasons": dict(sorted(stop_reasons.items())),
    }


def _build_reference_lookups(
    rows: Any,
) -> tuple[
    dict[tuple[str, str, str], Mapping[str, Any]],
    dict[tuple[str, str], list[Mapping[str, Any]]],
]:
    full: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        full[graph_completion_task_key(row)] = row
        pair = (str(row["source_index"]), str(row["variant_index"]))
        by_pair[pair].append(row)
    return full, by_pair


def _resolve_reference(
    prediction: Mapping[str, Any],
    full: Mapping[tuple[str, str, str], Mapping[str, Any]],
    by_pair: Mapping[tuple[str, str], list[Mapping[str, Any]]],
) -> tuple[Optional[Mapping[str, Any]], bool]:
    """Return (row, ambiguous) with a unique-pair fallback for older JSONL."""

    if prediction.get("mode") is not None and str(prediction.get("mode")).strip():
        return full.get(graph_completion_task_key(prediction)), False
    pair = (
        str(prediction.get("source_index")),
        str(prediction.get("variant_index")),
    )
    candidates = by_pair.get(pair, [])
    if len(candidates) == 1:
        return candidates[0], False
    return None, len(candidates) > 1


def _completion_token_count(prediction: Mapping[str, Any]) -> Optional[int]:
    value = prediction.get("completion_token_count")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "completion_token_count must be an integer for "
            f"{prediction.get('source_index')}:{prediction.get('variant_index')}:"
            f"{prediction.get('mode')}"
        ) from exc


def score_saved_predictions(
    reference_rows: Any,
    predictions: list[Mapping[str, Any]],
    reward_config: RewardConfig,
) -> tuple[list[dict[str, Any]], int, int]:
    """Score saved predictions with mode-safe reference matching.

    Returns ``(scored_rows, missing_reference_count,
    ambiguous_reference_count)``.
    """

    lookup, lookup_by_pair = _build_reference_lookups(reference_rows)
    output_rows: list[dict[str, Any]] = []
    missing = 0
    ambiguous = 0
    for prediction in predictions:
        row, is_ambiguous = _resolve_reference(
            prediction,
            lookup,
            lookup_by_pair,
        )
        if row is None:
            ambiguous += int(is_ambiguous)
            missing += int(not is_ambiguous)
            continue
        completion = prediction.get(
            "raw_completion",
            prediction.get("completion", ""),
        )
        result = score_graph_completion(
            completion,
            row,
            reward_config,
            completion_token_count=_completion_token_count(prediction),
        )
        record = {
            "source_index": row["source_index"],
            "variant_index": row["variant_index"],
            "mode": row["mode"],
            "reward": result.total,
            "components": result.components,
            "metrics": result.metrics,
            "errors": result.errors,
            "raw_completion": completion,
        }
        for name in (
            "task_index",
            "generation_index",
            "completion_token_count",
            "finish_reason",
            "stop_reason",
        ):
            if name in prediction:
                record[name] = prediction[name]
        output_rows.append(record)
    return output_rows, missing, ambiguous


def main() -> None:
    args = build_parser().parse_args()
    modes = [item.strip() for item in args.modes.split(",") if item.strip()] if args.modes else None
    prepared = prepare_graph_completion_datasets(
        args.dataset,
        validation_manifest=args.validation_manifest,
        invalid_pair_policy=args.invalid_pair_policy,
        seed=args.seed,
        validation_source_count=args.validation_source_count,
        modes=modes,
        max_eval_rows=args.max_rows if args.max_rows > 0 else None,
        max_train_rows=args.max_rows if args.max_rows > 0 else None,
    )
    selected = getattr(prepared, args.split)
    summary: dict[str, Any] = {
        "split": args.split,
        "rows": len(selected),
        "sources": len({str(value) for value in selected["source_index"]}),
        "modes": dict(sorted(Counter(map(str, selected["mode"])).items())),
        "train_invalid_by_mode": prepared.train_audit.invalid_by_mode,
        "test_invalid_by_mode": prepared.test_audit.invalid_by_mode,
        "validation_source_count": len(prepared.validation_source_indices),
        "mean_no_edit_primary": mean(selected["no_edit_primary"]) if len(selected) else None,
        "mean_no_edit_exact": mean(selected["no_edit_exact"]) if len(selected) else None,
    }
    output_rows: list[dict[str, Any]] = []
    if args.predictions:
        reward_config = reward_config_from_args(args)
        with Path(args.predictions).open(encoding="utf-8") as handle:
            predictions = [json.loads(line) for line in handle if line.strip()]
        output_rows, missing, ambiguous = score_saved_predictions(
            selected,
            predictions,
            reward_config,
        )
        per_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in output_rows:
            per_mode[str(record["mode"])].append(record)
        summary["predictions_scored"] = len(output_rows)
        summary["predictions_without_reference"] = missing
        summary["predictions_with_ambiguous_reference"] = ambiguous
        summary["prediction_metrics"] = _averages(output_rows)
        summary["prediction_metrics_by_mode"] = {
            mode: _averages(records) for mode, records in sorted(per_mode.items())
        }
        summary["prediction_analysis"] = _prediction_analysis(output_rows)
        summary["prediction_analysis_by_mode"] = {
            mode: _prediction_analysis(records)
            for mode, records in sorted(per_mode.items())
        }
        summary["prediction_generation"] = _generation_summary(
            output_rows,
            max_completion_length=args.max_completion_length,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_file:
        path = Path(args.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in output_rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary_path = path.with_suffix(path.suffix + ".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
