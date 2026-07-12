#!/usr/bin/env python
"""Audit graph-completion data and optionally score saved completion JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from graph_completion_data import DEFAULT_DATASET, prepare_graph_completion_datasets
from graph_completion_rewards import (
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
        help="Optional JSONL with source_index, variant_index, and raw_completion/completion.",
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
        lookup = {
            (str(row["source_index"]), str(row["variant_index"])): row
            for row in selected
        }
        reward_config = reward_config_from_args(args)
        with Path(args.predictions).open(encoding="utf-8") as handle:
            predictions = [json.loads(line) for line in handle if line.strip()]
        missing = 0
        per_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions:
            key = (str(prediction.get("source_index")), str(prediction.get("variant_index")))
            row = lookup.get(key)
            if row is None:
                missing += 1
                continue
            completion = prediction.get("raw_completion", prediction.get("completion", ""))
            result = score_graph_completion(completion, row, reward_config)
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
            output_rows.append(record)
            per_mode[str(row["mode"])].append(record)
        summary["predictions_scored"] = len(output_rows)
        summary["predictions_without_reference"] = missing
        summary["prediction_metrics"] = _averages(output_rows)
        summary["prediction_metrics_by_mode"] = {
            mode: _averages(records) for mode, records in sorted(per_mode.items())
        }
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

