import json
import sys

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

import run_graph_completion_checkpoint_benchmarks as checkpoint_benchmarks
from graph_completion_data import prepare_graph_completion_reference_split
from graph_completion_parsing import render_graph_canvas


def _write_manifest(path):
    payload = {
        "checkpoints": [
            {
                "step": 100,
                "label": "Early checkpoint",
                "model": "example/model-step-100",
            },
            {
                "step": 200,
                "label": "Later checkpoint",
                "model": "example/model-step-200",
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_summary(path, value):
    end_to_end = {}
    for metric in checkpoint_benchmarks.DEFAULT_METRICS:
        end_to_end[metric] = {
            "mean": value,
            "ci_low": max(0.0, value - 0.05),
            "ci_high": min(1.0, value + 0.05),
            "n": 4,
            "prediction_rows": 12,
            "conditioning": "all_completions",
            "aggregation_unit": "source",
        }
    payload = {
        "end_to_end": {"overall": end_to_end},
        "generation": {
            "overall": {
                "mean_tokens": 1000 + value * 100,
                "median_tokens": 900 + value * 100,
                "p95_tokens": 1800 + value * 100,
                "at_max_length_rate": 0.01,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _dataset_row(source):
    empty = {"nodes": [], "edges": []}
    target = {"nodes": [{"id": f"N{source}"}], "edges": []}
    return {
        "source_index": source,
        "variant_index": 0,
        "mode": "prior_empty",
        "condition": f"Complete graph {source}",
        "x0": render_graph_canvas(empty),
        "fixed_node_ids": [],
        "fixed_edge_keys": [],
        "x0_graph_json": json.dumps(empty),
        "x1_graph_json": json.dumps(target),
    }


def test_manifest_and_commands_use_validation_and_safe_resume(tmp_path):
    manifest = tmp_path / "checkpoints.json"
    _write_manifest(manifest)
    specs = checkpoint_benchmarks.load_checkpoint_manifest(manifest)
    args = checkpoint_benchmarks.build_parser().parse_args(
        ["--manifest", str(manifest)]
    )
    paths = checkpoint_benchmarks.checkpoint_paths(specs[0], tmp_path / "out")

    generation = checkpoint_benchmarks.generation_command(
        specs[0],
        paths,
        args,
        resume=True,
    )
    analysis = checkpoint_benchmarks.analysis_command(specs[0], paths, args)

    assert specs[0].step == 100
    assert "--resume_output_jsonl" in generation
    assert generation[generation.index("--split") + 1] == "validation"
    assert generation[generation.index("--num_tasks") + 1] == "512"
    assert analysis[analysis.index("--max_rows") + 1] == "512"


def test_manifest_rejects_duplicate_checkpoint_steps(tmp_path):
    manifest = tmp_path / "duplicates.json"
    manifest.write_text(
        json.dumps(
            [
                {"step": 10, "label": "A", "model": "model-a"},
                {"step": 10, "label": "B", "model": "model-b"},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="steps must be unique"):
        checkpoint_benchmarks.load_checkpoint_manifest(manifest)


def test_runner_requires_explicit_acknowledgement_for_test_sweep(
    tmp_path,
    monkeypatch,
):
    manifest = tmp_path / "checkpoints.json"
    _write_manifest(manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_graph_completion_checkpoint_benchmarks.py",
            "--manifest",
            str(manifest),
            "--output_dir",
            str(tmp_path / "out"),
            "--split",
            "test",
            "--dry_run",
        ],
    )

    with pytest.raises(SystemExit, match="refusing a multi-checkpoint test sweep"):
        checkpoint_benchmarks.main()


def test_plot_only_combines_summaries_and_writes_checkpoint_figures(
    tmp_path,
    monkeypatch,
):
    manifest = tmp_path / "checkpoints.json"
    _write_manifest(manifest)
    output_dir = tmp_path / "comparison"
    specs = checkpoint_benchmarks.load_checkpoint_manifest(manifest)
    for spec, value in zip(specs, (0.35, 0.62)):
        paths = checkpoint_benchmarks.checkpoint_paths(spec, output_dir)
        _write_summary(paths.analysis_dir / "benchmark_summary.json", value)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_graph_completion_checkpoint_benchmarks.py",
            "--manifest",
            str(manifest),
            "--output_dir",
            str(output_dir),
            "--plot_only",
            "--metrics",
            "reward,exact_match,node",
            "--png_dpi",
            "72",
        ],
    )

    checkpoint_benchmarks.main()

    frame = pd.read_csv(output_dir / "checkpoint_metrics.csv")
    assert frame["step"].tolist() == [100, 200]
    assert frame["reward"].tolist() == [0.35, 0.62]
    for name in (
        "checkpoint_reward_over_steps.svg",
        "checkpoint_reward_over_steps.png",
        "checkpoint_metrics_over_steps.svg",
        "checkpoint_metrics_over_steps.png",
        "checkpoint_comparison.json",
        "resolved_manifest.json",
        "README.md",
    ):
        assert (output_dir / name).exists()


def test_analysis_only_runs_existing_analyzer_for_each_checkpoint(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "dataset"
    DatasetDict(
        {
            "train": Dataset.from_list(
                [_dataset_row(source) for source in range(6)]
            ),
            "test": Dataset.from_list(
                [_dataset_row(source) for source in range(20, 22)]
            ),
        }
    ).save_to_disk(str(dataset_path))
    validation_manifest = tmp_path / "validation.json"
    references = prepare_graph_completion_reference_split(
        str(dataset_path),
        split="validation",
        validation_manifest=str(validation_manifest),
        validation_source_count=2,
        seed=42,
        max_rows=2,
    )
    prediction_paths = []
    for index in range(2):
        prediction_path = tmp_path / f"predictions-{index}.jsonl"
        prediction_path.write_text(
            "".join(
                json.dumps(
                    {
                        "source_index": row["source_index"],
                        "variant_index": row["variant_index"],
                        "mode": row["mode"],
                        "generation_index": 1,
                        "raw_completion": (
                            f"<answer>{row['x1_graph_json']}</answer>"
                        ),
                        "completion_token_count": 20,
                        "finish_reason": "stop",
                    }
                )
                + "\n"
                for row in references
            ),
            encoding="utf-8",
        )
        prediction_paths.append(prediction_path)
    manifest = tmp_path / "analysis-only.json"
    manifest.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "step": 10,
                        "label": "Checkpoint 10",
                        "model": "unused/model-10",
                        "predictions": str(prediction_paths[0]),
                    },
                    {
                        "step": 20,
                        "label": "Checkpoint 20",
                        "model": "unused/model-20",
                        "predictions": str(prediction_paths[1]),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis-only-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_graph_completion_checkpoint_benchmarks.py",
            "--manifest",
            str(manifest),
            "--output_dir",
            str(output_dir),
            "--dataset",
            str(dataset_path),
            "--split",
            "validation",
            "--validation_manifest",
            str(validation_manifest),
            "--validation_source_count",
            "2",
            "--num_tasks",
            "2",
            "--analysis_only",
            "--bootstrap_samples",
            "20",
            "--png_dpi",
            "72",
        ],
    )

    checkpoint_benchmarks.main()

    assert (output_dir / "checkpoint_metrics.csv").exists()
    for spec in checkpoint_benchmarks.load_checkpoint_manifest(manifest):
        paths = checkpoint_benchmarks.checkpoint_paths(spec, output_dir)
        assert (paths.analysis_dir / "benchmark_summary.json").exists()
