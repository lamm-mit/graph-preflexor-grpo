import io
import json
import sys
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from graph_completion_parsing import render_graph_canvas
from sample_graph_completion import (
    append_graph_completion_samples_jsonl,
    build_manual_graph_completion_task,
    build_graph_completion_sample_prompts,
    build_parser,
    load_completed_graph_completion_task_keys,
    parse_graph_completion_modes,
    print_graph_completion_samples,
    sample_graph_completion_tasks,
    score_graph_completion_samples,
    select_unfinished_graph_completion_tasks,
)
import sample_graph_completion


EMPTY = {"nodes": [], "edges": []}
TARGET = {"nodes": [{"id": "A"}], "edges": []}


def make_row(source, mode, variant=0):
    if mode == "prior_empty":
        x0_graph = EMPTY
        target_graph = TARGET
    else:
        nodes = [{"id": "A"}, {"id": "B"}]
        correct_edge = {"source": "A", "relation": "activates", "target": "B"}
        target_graph = {"nodes": nodes, "edges": [correct_edge]}
        if mode == "wrong_relations":
            x0_graph = {
                "nodes": nodes,
                "edges": [{"source": "A", "relation": "inhibits", "target": "B"}],
            }
        elif mode == "extra_edges":
            x0_graph = {
                "nodes": nodes,
                "edges": [
                    correct_edge,
                    {"source": "B", "relation": "spurious", "target": "A"},
                ],
            }
        else:
            x0_graph = {"nodes": nodes, "edges": []}
    return {
        "source_index": source,
        "variant_index": variant,
        "mode": mode,
        "condition": "A condition",
        "x0": render_graph_canvas(x0_graph),
        "fixed_node_ids": [],
        "fixed_edge_keys": [],
        "x0_graph_json": json.dumps(x0_graph),
        "x1_graph_json": json.dumps(target_graph),
    }


class MockTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\nMODEL:"


def test_sample_cli_defaults_match_training_rollouts():
    args = build_parser().parse_args([])
    assert args.model == "google/gemma-4-E4B-it"
    assert args.view == "raw"
    assert args.temperature == 0.8
    assert args.top_p == 0.95
    assert args.max_prompt_length == 4096
    assert args.max_completion_length == 4096
    assert args.dtype == "bfloat16"
    assert args.vllm_gpu_memory_utilization == 0.45
    assert not args.use_cuda_graphs
    assert args.chat_template_enable_thinking == "true"
    assert args.manual_mode == "partial_subgraph"
    assert args.manual_fixed_policy == "all"
    assert args.generation_batch_size == 0
    assert not args.stream_output_jsonl
    assert not args.resume_output_jsonl


def test_parse_modes_rejects_unknown_mode():
    assert parse_graph_completion_modes("missing_edges, extra_edges") == [
        "missing_edges",
        "extra_edges",
    ]
    with pytest.raises(ValueError, match="unknown graph-completion modes"):
        parse_graph_completion_modes("not_a_mode")


def test_task_sampling_is_mode_balanced_and_prompt_matches_training(tmp_path):
    rows = [
        make_row(1, "missing_edges"),
        make_row(2, "missing_edges"),
        make_row(3, "wrong_relations"),
        make_row(4, "extra_edges"),
    ]
    path = tmp_path / "dataset"
    DatasetDict(
        {
            "train": Dataset.from_list([make_row(10, "prior_empty")]),
            "test": Dataset.from_list(rows),
        }
    ).save_to_disk(str(path))
    sampled = sample_graph_completion_tasks(
        str(path),
        modes=["missing_edges", "wrong_relations", "extra_edges"],
        num_tasks=3,
        seed=9,
    )
    assert set(sampled["mode"]) == {"missing_edges", "wrong_relations", "extra_edges"}
    prompts = build_graph_completion_sample_prompts(sampled, MockTokenizer())
    assert len(prompts) == 3
    assert all("complete corrected graph" in prompt for prompt in prompts)
    assert all("Corruption mode:" in prompt for prompt in prompts)
    assert all("Condition:\nA condition" in prompt for prompt in prompts)


def test_manual_partial_graph_is_rendered_fixed_and_conditioned(tmp_path):
    graph = {
        "nodes": [{"id": "Humidity"}, {"id": "WaterUptake"}],
        "edges": [
            {
                "source": "Humidity",
                "relation": "increases",
                "target": "WaterUptake",
            }
        ],
    }
    row = build_manual_graph_completion_task(
        "Complete the humidity mechanism.",
        partial_graph_json=json.dumps(graph),
    )
    assert row["condition"] == "Complete the humidity mechanism."
    assert row["mode"] == "partial_subgraph"
    assert row["x0"].count("[FIXED]") == 3
    assert row["fixed_node_ids"] == ["Humidity", "WaterUptake"]
    assert len(row["fixed_edge_keys"]) == 1

    graph_file = tmp_path / "partial.json"
    graph_file.write_text(json.dumps(graph), encoding="utf-8")
    editable = build_manual_graph_completion_task(
        "Repair the humidity mechanism.",
        partial_graph_file=str(graph_file),
        mode="wrong_relations",
        fixed_policy="none",
    )
    assert "[FIXED]" not in editable["x0"]
    assert editable["fixed_node_ids"] == []
    assert editable["fixed_edge_keys"] == []


def test_manual_partial_graph_rejects_missing_condition_and_dangling_edges():
    with pytest.raises(ValueError, match="non-empty condition"):
        build_manual_graph_completion_task(
            "",
            partial_graph_json=json.dumps(EMPTY),
        )
    dangling = {
        "nodes": [{"id": "A"}],
        "edges": [{"source": "A", "relation": "points_to", "target": "B"}],
    }
    with pytest.raises(ValueError, match="dangling"):
        build_manual_graph_completion_task(
            "A condition",
            partial_graph_json=json.dumps(dangling),
        )


def test_raw_render_preserves_completion_and_scored_view_uses_same_record():
    rows = Dataset.from_list([make_row(1, "prior_empty")])
    raw = "thinking\n<answer>{\"nodes\":[],\"edges\":[]}</answer>\n"
    records = [
        {
            "task_index": 1,
            "generation_index": 1,
            "source_index": 1,
            "variant_index": 0,
            "mode": "prior_empty",
            "prompt": "PROMPT",
            "raw_completion": raw,
            "completion_token_count": 12,
            "finish_reason": "length",
            "stop_reason": None,
        }
    ]
    stream = io.StringIO()
    print_graph_completion_samples(records, stream=stream)
    assert raw in stream.getvalue()

    from graph_completion_rewards import RewardConfig

    scored = score_graph_completion_samples(records, rows, RewardConfig())
    assert scored[0]["raw_completion"] == raw
    assert "reward" in scored[0]
    assert "reference_graph_json" in scored[0]
    scored_stream = io.StringIO()
    print_graph_completion_samples(scored, view="scored", stream=scored_stream)
    assert "REFERENCE COMPLETE GRAPH" in scored_stream.getvalue()
    assert "TRAINING REWARD" in scored_stream.getvalue()


def test_scoring_uses_mode_in_task_identity_for_repeated_source_variant():
    rows = Dataset.from_list(
        [
            make_row(1, "prior_empty"),
            make_row(1, "wrong_relations"),
        ]
    )
    prior_target = json.dumps(TARGET, separators=(",", ":"))
    relation_target = json.dumps(
        {
            "nodes": [{"id": "A"}, {"id": "B"}],
            "edges": [{"source": "A", "relation": "activates", "target": "B"}],
        },
        separators=(",", ":"),
    )
    records = [
        {
            "source_index": 1,
            "variant_index": 0,
            "mode": "prior_empty",
            "raw_completion": f"<answer>{prior_target}</answer>",
            "completion_token_count": 12,
        },
        {
            "source_index": 1,
            "variant_index": 0,
            "mode": "wrong_relations",
            "raw_completion": f"<answer>{relation_target}</answer>",
            "completion_token_count": 24,
        },
    ]

    from graph_completion_rewards import RewardConfig

    scored = score_graph_completion_samples(records, rows, RewardConfig())

    assert [record["reward_components"]["exact_match"] for record in scored] == [1.0, 1.0]
    assert scored[0]["reference_graph_json"] != scored[1]["reference_graph_json"]


def test_vllm_023_prompt_truncation_is_not_a_sampling_parameter(monkeypatch):
    calls = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            calls["sampling"] = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            calls["engine"] = kwargs

        def generate(self, prompts, sampling, **kwargs):
            calls["generate"] = kwargs
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        SimpleNamespace(
                            text="RAW",
                            token_ids=[4, 5],
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ],
                )
            ]

    tokenizer = MockTokenizer()
    tokenizer.decode = lambda token_ids, **kwargs: "EFFECTIVE PROMPT"
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    monkeypatch.setattr(
        sample_graph_completion.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    rows = Dataset.from_list([make_row(1, "prior_empty")])
    records = sample_graph_completion.generate_graph_completion_samples(
        rows,
        max_prompt_length=123,
        max_completion_length=45,
    )

    assert "truncate_prompt_tokens" not in calls["sampling"]
    assert calls["generate"]["tokenization_kwargs"] == {
        "truncation": True,
        "max_length": 123,
    }
    assert calls["engine"]["max_model_len"] == 168
    assert records[0]["prompt"] == "EFFECTIVE PROMPT"
    assert records[0]["raw_completion"] == "RAW"


def test_generation_batches_preserve_task_indices_and_call_callback(monkeypatch):
    calls = {"batch_sizes": []}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, sampling, **kwargs):
            calls["batch_sizes"].append(len(prompts))
            return [
                SimpleNamespace(
                    prompt_token_ids=[index + 1],
                    outputs=[
                        SimpleNamespace(
                            text=f"RAW-{prompt}",
                            token_ids=[4, 5],
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ],
                )
                for index, prompt in enumerate(prompts)
            ]

    tokenizer = MockTokenizer()
    tokenizer.decode = lambda token_ids, **kwargs: f"PROMPT-{token_ids[0]}"
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    monkeypatch.setattr(
        sample_graph_completion.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    rows = Dataset.from_list(
        [make_row(1, "prior_empty"), make_row(2, "prior_empty"), make_row(3, "prior_empty")]
    )
    callback_sizes = []
    records = sample_graph_completion.generate_graph_completion_samples(
        rows,
        generation_batch_size=2,
        batch_callback=lambda batch: callback_sizes.append(len(batch)),
    )

    assert calls["batch_sizes"] == [2, 1]
    assert callback_sizes == [2, 1]
    assert [record["task_index"] for record in records] == [1, 2, 3]
    assert [record["source_index"] for record in records] == [1, 2, 3]


def test_append_jsonl_keeps_completed_batches(tmp_path):
    output = tmp_path / "intermediate.jsonl"
    first = [{"task_index": 1}, {"task_index": 2}]
    second = [{"task_index": 3}]

    assert append_graph_completion_samples_jsonl(first, str(output)) == 2
    assert append_graph_completion_samples_jsonl(second, str(output)) == 1
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task_index"] for row in rows] == [1, 2, 3]


def test_resume_selects_only_missing_task_identities(tmp_path):
    rows = Dataset.from_list(
        [
            make_row(1, "prior_empty"),
            make_row(1, "wrong_relations"),
            make_row(2, "extra_edges"),
        ]
    )
    output = tmp_path / "predictions.jsonl"
    append_graph_completion_samples_jsonl(
        [
            {
                "source_index": 1,
                "variant_index": 0,
                "mode": "prior_empty",
                "generation_index": 1,
                "raw_completion": "done",
            },
            {
                "source_index": 2,
                "variant_index": 0,
                "mode": "extra_edges",
                "generation_index": 1,
                "raw_completion": "done",
            },
        ],
        str(output),
    )

    completed, count = load_completed_graph_completion_task_keys(str(output))
    unfinished = select_unfinished_graph_completion_tasks(rows, completed)

    assert count == 2
    assert len(unfinished) == 1
    assert unfinished[0]["mode"] == "wrong_relations"


def test_resume_rejects_duplicate_existing_task_identity(tmp_path):
    output = tmp_path / "duplicates.jsonl"
    record = {
        "source_index": 1,
        "variant_index": 0,
        "mode": "prior_empty",
        "generation_index": 1,
    }
    append_graph_completion_samples_jsonl([record, record], str(output))

    with pytest.raises(ValueError, match="duplicate completed task identity"):
        load_completed_graph_completion_task_keys(str(output))


def test_resume_main_preserves_existing_jsonl_and_appends_missing(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "resume.jsonl"
    existing = {
        "source_index": 1,
        "variant_index": 0,
        "mode": "prior_empty",
        "generation_index": 1,
        "raw_completion": "existing",
    }
    append_graph_completion_samples_jsonl([existing], str(output))
    requested = Dataset.from_list(
        [
            make_row(1, "prior_empty"),
            make_row(2, "extra_edges"),
        ]
    )

    monkeypatch.setattr(
        sample_graph_completion,
        "sample_graph_completion_tasks",
        lambda *args, **kwargs: requested,
    )
    monkeypatch.setattr(
        sample_graph_completion,
        "print_graph_completion_samples",
        lambda *args, **kwargs: None,
    )

    def fake_generate(rows, *, batch_callback, **kwargs):
        assert len(rows) == 1
        record = {
            "source_index": 2,
            "variant_index": 0,
            "mode": "extra_edges",
            "generation_index": 1,
            "raw_completion": "new",
        }
        batch_callback([record])
        return [record]

    monkeypatch.setattr(
        sample_graph_completion,
        "generate_graph_completion_samples",
        fake_generate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample_graph_completion.py",
            "--num_tasks",
            "2",
            "--num_generations",
            "1",
            "--generation_batch_size",
            "1",
            "--stream_output_jsonl",
            "--resume_output_jsonl",
            "--view",
            "raw",
            "--output_jsonl",
            str(output),
        ],
    )

    sample_graph_completion.main()

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["raw_completion"] for record in records] == ["existing", "new"]
