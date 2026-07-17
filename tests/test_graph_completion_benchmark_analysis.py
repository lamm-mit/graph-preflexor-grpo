import json
import sys

import numpy as np
from datasets import Dataset

import analyze_graph_completion_benchmark as benchmark
from graph_completion_parsing import GRAPH_COMPLETION_MODES


TARGET = {
    "nodes": [{"id": "A"}, {"id": "B"}],
    "edges": [{"source": "A", "relation": "activates", "target": "B"}],
}
EMPTY = {"nodes": [], "edges": []}


def _reference_row(source, mode):
    return {
        "source_index": source,
        "variant_index": 0,
        "mode": mode,
        "x0_graph_json": json.dumps(EMPTY),
        "x1_graph_json": json.dumps(TARGET),
        "fixed_node_ids": [],
        "fixed_edge_keys": [],
        "no_edit_primary": 0.0,
    }


def test_source_macro_aggregation_does_not_treat_variants_as_independent():
    records = [
        {"source_index": 1, "value": 1.0},
        {"source_index": 1, "value": 1.0},
        {"source_index": 2, "value": 0.0},
    ]
    specs = (("value", "Value", lambda row: row["value"]),)

    source_rows = benchmark._summarize_specs(
        records,
        specs,
        scope="overall",
        mode="overall",
        conditioning="all",
        samples=0,
        confidence=0.95,
        rng=np.random.default_rng(1),
        aggregation_unit="source",
    )
    task_rows = benchmark._summarize_specs(
        records,
        specs,
        scope="overall",
        mode="overall",
        conditioning="all",
        samples=0,
        confidence=0.95,
        rng=np.random.default_rng(1),
        aggregation_unit="task",
    )

    assert source_rows[0]["mean"] == 0.5
    assert source_rows[0]["n"] == 2
    assert source_rows[0]["prediction_rows"] == 3
    assert task_rows[0]["mean"] == 2 / 3
    assert task_rows[0]["n"] == 3


def test_benchmark_cli_scores_all_modes_and_writes_tables_and_figures(
    tmp_path,
    monkeypatch,
):
    references = Dataset.from_list(
        [
            _reference_row(source, mode)
            for source in (10, 11)
            for mode in GRAPH_COMPLETION_MODES
        ]
    )
    predictions = []
    for row in references:
        is_valid = row["source_index"] == 10
        predictions.append(
            {
                "source_index": row["source_index"],
                "variant_index": row["variant_index"],
                "mode": row["mode"],
                "generation_index": 1,
                "raw_completion": (
                    f"<answer>{json.dumps(TARGET)}</answer>"
                    if is_valid
                    else "<answer>{bad}</answer>"
                ),
                "completion_token_count": 100 if is_valid else 4096,
                "finish_reason": "stop" if is_valid else "length",
                "stop_reason": "</answer>" if is_valid else None,
            }
        )

    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row) + "\n" for row in predictions),
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        benchmark,
        "prepare_graph_completion_reference_split",
        lambda *args, **kwargs: references,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_graph_completion_benchmark.py",
            "--predictions",
            str(prediction_path),
            "--output_dir",
            str(output_dir),
            "--bootstrap_samples",
            "20",
            "--png_dpi",
            "72",
            "--label",
            "Synthetic benchmark",
        ],
    )

    benchmark.main()

    expected_files = {
        "scored_predictions.jsonl",
        "end_to_end_metrics.csv",
        "conditional_graph_metrics.csv",
        "reward_components.csv",
        "generation_metrics.csv",
        "benchmark_summary.json",
        "README.md",
    }
    for stem in (
        "benchmark_overall",
        "benchmark_by_mode",
        "benchmark_conditional_precision_recall",
        "benchmark_reward_components",
        "benchmark_generation_diagnostics",
    ):
        expected_files.update({f"{stem}.svg", f"{stem}.png"})
    assert expected_files <= {path.name for path in output_dir.iterdir()}

    summary = json.loads((output_dir / "benchmark_summary.json").read_text())
    integrity = summary["integrity"]
    assert integrity["expected_unique_tasks"] == 12
    assert integrity["predicted_unique_tasks"] == 12
    assert integrity["missing_expected_tasks"] == 0
    assert integrity["predictions_without_reference"] == 0
    assert integrity["predictions_with_ambiguous_reference"] == 0
    exact = summary["end_to_end"]["overall"]["exact_match"]
    assert exact["mean"] == 0.5
    assert exact["n"] == 2
    assert exact["prediction_rows"] == 12
    assert summary["generation"]["overall"]["at_max_length_count"] == 6
    assert "<svg" in (output_dir / "benchmark_overall.svg").read_text()
