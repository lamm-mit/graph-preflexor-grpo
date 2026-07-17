#!/usr/bin/env python
"""Sample raw or scored graph-completion rollouts with the training prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, TextIO

from transformers import AutoTokenizer

from chat_template_utils import add_chat_template_args, parse_chat_template_enable_thinking
from graph_completion_data import (
    DEFAULT_DATASET,
    add_no_edit_baselines,
    audit_and_filter_pairs,
    graph_completion_task_key,
    load_graph_completion_dataset,
)
from graph_completion_parsing import GRAPH_COMPLETION_MODES
from graph_completion_parsing import (
    audit_raw_graph,
    edge_key,
    parse_graph_json,
    render_graph_canvas,
    validate_graph_schema,
)
from graph_completion_prompting import apply_graph_completion_chat_template
from graph_completion_rewards import (
    RewardConfig,
    add_reward_arguments,
    reward_config_from_args,
    score_graph_completion,
)


DEFAULT_MODEL = "google/gemma-4-E4B-it"


def parse_graph_completion_modes(value: Optional[str]) -> Optional[list[str]]:
    """Parse and validate a comma-separated corruption-mode selection."""

    if not value:
        return None
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(modes) - set(GRAPH_COMPLETION_MODES)
    if unknown:
        raise ValueError(f"unknown graph-completion modes: {sorted(unknown)}")
    return modes


def sample_graph_completion_tasks(
    dataset_name: str = DEFAULT_DATASET,
    *,
    split: str = "test",
    modes: Optional[Iterable[str]] = None,
    num_tasks: int = 3,
    seed: int = 42,
    invalid_pair_policy: str = "filter",
) -> Any:
    """Select valid, deterministic, mode-balanced graph-completion rows."""

    if split not in {"train", "test"}:
        raise ValueError("sampling split must be the official train or test split")
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    selected_modes = list(modes) if modes else None
    if selected_modes:
        unknown = set(selected_modes) - set(GRAPH_COMPLETION_MODES)
        if unknown:
            raise ValueError(f"unknown graph-completion modes: {sorted(unknown)}")

    dataset = load_graph_completion_dataset(dataset_name)[split]
    dataset, _ = audit_and_filter_pairs(
        dataset,
        split_name=split,
        invalid_pair_policy=invalid_pair_policy,
    )
    if selected_modes:
        allowed = set(selected_modes)
        dataset = dataset.select(
            [index for index, mode in enumerate(dataset["mode"]) if str(mode) in allowed]
        )
    if not len(dataset):
        raise ValueError("no graph-completion rows match the requested split and modes")
    # Take one row per mode per round. This gives visual inspection precedence
    # to mode coverage even when a requested mode has only one available row.
    shuffled = dataset.shuffle(seed=seed)
    indices_by_mode: dict[str, list[int]] = {
        mode: [] for mode in (selected_modes or GRAPH_COMPLETION_MODES)
    }
    for index, mode in enumerate(shuffled["mode"]):
        indices_by_mode.setdefault(str(mode), []).append(index)
    chosen: list[int] = []
    while len(chosen) < min(num_tasks, len(shuffled)):
        progressed = False
        for mode in indices_by_mode:
            if indices_by_mode[mode] and len(chosen) < num_tasks:
                chosen.append(indices_by_mode[mode].pop())
                progressed = True
        if not progressed:
            break
    return shuffled.select(chosen)


def build_graph_completion_sample_prompts(
    rows: Any,
    tokenizer: Any,
    *,
    enable_thinking: Optional[bool] = True,
) -> list[str]:
    """Apply the exact graph-completion training prompt to sampled rows."""

    return [
        apply_graph_completion_chat_template(
            tokenizer,
            row["x0"],
            condition=row.get("condition"),
            mode=str(row["mode"]) if row.get("mode") else None,
            enable_thinking=enable_thinking,
        )
        for row in rows
    ]


def build_manual_graph_completion_task(
    condition: str,
    *,
    partial_graph_json: Optional[str] = None,
    partial_graph_file: Optional[str] = None,
    mode: Optional[str] = "partial_subgraph",
    fixed_policy: str = "all",
) -> dict[str, Any]:
    """Build one inference row from a user-provided partial graph.

    With the default ``fixed_policy='all'``, every supplied node and edge is
    rendered as ``[FIXED]`` so completion may add content but cannot rewrite
    the user's facts. Use ``fixed_policy='none'`` for a corrupted graph that the
    model is allowed to repair.
    """

    if not str(condition).strip():
        raise ValueError("manual graph completion requires a non-empty condition")
    if bool(partial_graph_json) == bool(partial_graph_file):
        raise ValueError(
            "provide exactly one of partial_graph_json or partial_graph_file"
        )
    if mode is not None and mode not in GRAPH_COMPLETION_MODES:
        raise ValueError(f"unknown graph-completion mode: {mode!r}")
    if fixed_policy not in {"all", "none"}:
        raise ValueError("fixed_policy must be all or none")

    source = (
        Path(str(partial_graph_file)).expanduser().read_text(encoding="utf-8")
        if partial_graph_file
        else str(partial_graph_json)
    )
    graph = parse_graph_json(source)
    schema_errors = validate_graph_schema(graph)
    if schema_errors:
        raise ValueError("invalid partial graph: " + "; ".join(schema_errors))
    audit = audit_raw_graph(graph)
    if not audit["structural_validity"]:
        raise ValueError(
            "partial graph must not contain duplicate nodes, duplicate edges, or dangling edges"
        )

    fixed_node_ids = (
        [str(node["id"]) for node in graph["nodes"]]
        if fixed_policy == "all"
        else []
    )
    fixed_edge_keys = (
        [edge_key(edge) for edge in graph["edges"]]
        if fixed_policy == "all"
        else []
    )
    return {
        "source_index": "manual",
        "variant_index": 0,
        "mode": mode,
        "condition": str(condition).strip(),
        "x0": render_graph_canvas(
            graph,
            fixed_node_ids=fixed_node_ids,
            fixed_edge_keys=fixed_edge_keys,
        ),
        "fixed_node_ids": fixed_node_ids,
        "fixed_edge_keys": fixed_edge_keys,
        "x0_graph_json": json.dumps(graph, ensure_ascii=False),
    }


def generate_graph_completion_samples(
    rows: Any,
    *,
    model: str = DEFAULT_MODEL,
    tokenizer_model: Optional[str] = None,
    revision: Optional[str] = None,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.45,
    max_prompt_length: int = 4096,
    max_completion_length: int = 4096,
    num_generations: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 42,
    enable_thinking: Optional[bool] = True,
    use_cuda_graphs: bool = False,
    enable_prefix_caching: bool = True,
    generation_batch_size: int = 0,
    show_progress: bool = False,
    batch_callback: Optional[Callable[[list[dict[str, Any]]], None]] = None,
) -> list[dict[str, Any]]:
    """Generate vLLM rollouts and retain each decoded completion unchanged.

    vLLM is imported lazily so dataset audits, validation, and unit tests do not
    acquire a vLLM dependency merely by importing this module.
    """

    if not 0.0 < gpu_memory_utilization < 1.0:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    if max_prompt_length <= 0 or max_completion_length <= 0:
        raise ValueError("prompt and completion lengths must be positive")
    if num_generations <= 0:
        raise ValueError("num_generations must be positive")
    if generation_batch_size < 0:
        raise ValueError("generation_batch_size must be non-negative")

    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise ImportError(
            "Graph-completion sampling requires vLLM. Install the TRL-compatible "
            "vLLM version used by the training environment."
        ) from error

    row_list = list(rows)
    if not row_list:
        return []

    tokenizer_source = tokenizer_model or model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=revision)
    prompts = build_graph_completion_sample_prompts(
        row_list,
        tokenizer,
        enable_thinking=enable_thinking,
    )

    engine_kwargs: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer_source,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_prompt_length + max_completion_length,
        "enforce_eager": not use_cuda_graphs,
        "enable_prefix_caching": enable_prefix_caching,
        "seed": seed,
    }
    if revision:
        engine_kwargs["revision"] = revision
        engine_kwargs["tokenizer_revision"] = revision
    # vLLM 0.23 removed truncate_prompt_tokens from SamplingParams. Construct
    # sampling parameters before loading model weights so an incompatible
    # generation API fails immediately instead of after a multi-minute load.
    sampling = SamplingParams(
        n=num_generations,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_completion_length,
        seed=seed,
        skip_special_tokens=True,
    )
    engine = LLM(**engine_kwargs)
    records: list[dict[str, Any]] = []
    batch_size = generation_batch_size or len(row_list)
    progress = None
    if show_progress:
        from tqdm.auto import tqdm

        progress = tqdm(
            total=len(row_list),
            desc="Graph-completion tasks",
            unit="task",
            dynamic_ncols=True,
        )

    try:
        for batch_start in range(0, len(row_list), batch_size):
            batch_end = min(len(row_list), batch_start + batch_size)
            batch_rows = row_list[batch_start:batch_end]
            batch_prompts = prompts[batch_start:batch_end]
            request_outputs = engine.generate(
                batch_prompts,
                sampling,
                use_tqdm=False,
                tokenization_kwargs={
                    "truncation": True,
                    "max_length": max_prompt_length,
                },
            )

            batch_records: list[dict[str, Any]] = []
            for local_index, (row, requested_prompt, request_output) in enumerate(
                zip(batch_rows, batch_prompts, request_outputs),
                start=1,
            ):
                task_index = batch_start + local_index
                prompt_token_ids = list(
                    getattr(request_output, "prompt_token_ids", []) or []
                )
                effective_prompt = (
                    tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
                    if prompt_token_ids
                    else requested_prompt
                )
                for generation_index, completion in enumerate(
                    request_output.outputs, start=1
                ):
                    token_ids = list(completion.token_ids)
                    batch_records.append(
                        {
                            "task_index": task_index,
                            "generation_index": generation_index,
                            "source_index": row["source_index"],
                            "variant_index": row["variant_index"],
                            "mode": str(row["mode"]),
                            "condition": row.get("condition"),
                            "x0": row["x0"],
                            "prompt": effective_prompt,
                            "prompt_token_count": len(prompt_token_ids),
                            # This is deliberately not parsed, stripped,
                            # repaired, or reduced to its <answer> block.
                            "raw_completion": completion.text,
                            "completion_token_count": len(token_ids),
                            "completion_token_ids": token_ids,
                            "finish_reason": (
                                str(completion.finish_reason)
                                if completion.finish_reason is not None
                                else None
                            ),
                            "stop_reason": completion.stop_reason,
                        }
                    )
            records.extend(batch_records)
            if batch_callback is not None:
                batch_callback(batch_records)
            if progress is not None:
                progress.update(len(batch_rows))
    finally:
        if progress is not None:
            progress.close()
    return records


def score_graph_completion_samples(
    records: Iterable[Mapping[str, Any]],
    rows: Any,
    config: RewardConfig,
) -> list[dict[str, Any]]:
    """Attach the training reward, components, metrics, and reference graph."""

    if "no_edit_primary" not in rows.column_names:
        rows = add_no_edit_baselines(rows)
    lookup = {
        graph_completion_task_key(row): row
        for row in rows
    }
    scored: list[dict[str, Any]] = []
    for source_record in records:
        record = dict(source_record)
        key = graph_completion_task_key(record)
        row = lookup.get(key)
        if row is None:
            raise ValueError(f"sample has no matching reference row: {key}")
        result = score_graph_completion(
            record["raw_completion"],
            row,
            config,
            completion_token_count=int(record["completion_token_count"]),
        )
        record.update(
            {
                "reward": result.total,
                "reward_components": result.components,
                "metrics": result.metrics,
                "validation_errors": result.errors,
                "reference_graph_json": row["x1_graph_json"],
            }
        )
        scored.append(record)
    return scored


def print_graph_completion_samples(
    records: Iterable[Mapping[str, Any]],
    *,
    view: str = "raw",
    stream: TextIO = sys.stdout,
) -> None:
    """Print either untouched rollouts or rollout/reference score comparisons."""

    if view not in {"raw", "scored"}:
        raise ValueError("view must be raw or scored")
    for record in records:
        stream.write("\n" + "=" * 100 + "\n")
        stream.write(
            f"TASK {record['task_index']} / RESPONSE {record['generation_index']} | "
            f"mode={record['mode']} source_index={record['source_index']} "
            f"variant_index={record['variant_index']}\n"
        )
        stream.write("\n========== EXACT PROMPT TOKENS GIVEN TO MODEL ==========\n\n")
        stream.write(str(record["prompt"]))
        stream.write("\n\n========== RAW DECODED MODEL COMPLETION ==========\n\n")
        # Keep the actual rollout bytes represented by the Python string
        # unchanged; only the surrounding display markers are added.
        stream.write(str(record["raw_completion"]))
        stream.write("\n\n========== END RAW COMPLETION ==========\n")
        stream.write(
            f"completion_tokens={record['completion_token_count']} "
            f"finish_reason={record.get('finish_reason')} "
            f"stop_reason={record.get('stop_reason')}\n"
        )
        if view == "scored":
            if "reward" not in record:
                raise ValueError("scored view requires score_graph_completion_samples output")
            stream.write("\n========== REFERENCE COMPLETE GRAPH ==========\n\n")
            reference = json.loads(str(record["reference_graph_json"]))
            stream.write(json.dumps(reference, indent=2, ensure_ascii=False))
            stream.write("\n\n========== TRAINING REWARD ==========\n\n")
            summary = {
                "reward": record["reward"],
                "components": record["reward_components"],
                "metrics": record["metrics"],
                "validation_errors": record["validation_errors"],
            }
            stream.write(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
            stream.write("\n")


def write_graph_completion_samples_jsonl(
    records: Iterable[Mapping[str, Any]],
    output_file: str,
) -> None:
    """Persist raw or scored sample records without altering completions."""

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def append_graph_completion_samples_jsonl(
    records: Iterable[Mapping[str, Any]],
    output_file: str,
) -> int:
    """Append and flush one generated batch, returning the record count."""

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer_model", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--modes", default=None, help="Optional comma-separated mode subset.")
    parser.add_argument("--num_tasks", type=int, default=3)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=0,
        help=(
            "Generate this many tasks per vLLM call and show task-level tqdm "
            "progress. Zero keeps the original all-at-once behavior."
        ),
    )
    parser.add_argument(
        "--stream_output_jsonl",
        action="store_true",
        help=(
            "Append and flush each completed generation batch to --output_jsonl. "
            "Requires --view raw and a positive --generation_batch_size."
        ),
    )
    parser.add_argument(
        "--condition",
        default=None,
        help="Manual inference condition; requires a partial-graph argument.",
    )
    manual_graph = parser.add_mutually_exclusive_group()
    manual_graph.add_argument(
        "--partial_graph_json",
        default=None,
        help="Inline partial graph object with nodes and edges arrays.",
    )
    manual_graph.add_argument(
        "--partial_graph_file",
        default=None,
        help="Path to a partial graph JSON file.",
    )
    parser.add_argument(
        "--manual_mode",
        choices=list(GRAPH_COMPLETION_MODES),
        default="partial_subgraph",
    )
    parser.add_argument(
        "--manual_fixed_policy",
        choices=["all", "none"],
        default="all",
        help="Mark every supplied object fixed, or allow the model to repair them.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invalid_pair_policy", choices=["filter", "error"], default="filter")
    parser.add_argument("--view", choices=["raw", "scored"], default="raw")
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument(
        "--output_text_file",
        default=None,
        help="Optional copy of the human-readable terminal output.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.45)
    parser.add_argument("--use_cuda_graphs", action="store_true")
    parser.add_argument("--no_prefix_caching", action="store_true")
    add_chat_template_args(parser)
    parser.set_defaults(chat_template_enable_thinking="true")
    add_reward_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stream_output_jsonl:
        if not args.output_jsonl:
            raise ValueError("--stream_output_jsonl requires --output_jsonl")
        if args.generation_batch_size <= 0:
            raise ValueError(
                "--stream_output_jsonl requires a positive --generation_batch_size"
            )
        if args.view != "raw":
            raise ValueError("--stream_output_jsonl currently requires --view raw")
    manual_requested = bool(args.partial_graph_json or args.partial_graph_file)
    if manual_requested:
        if args.view == "scored":
            raise ValueError(
                "manual inference has no gold graph; use --view raw or dataset sampling"
            )
        rows = [
            build_manual_graph_completion_task(
                args.condition or "",
                partial_graph_json=args.partial_graph_json,
                partial_graph_file=args.partial_graph_file,
                mode=args.manual_mode,
                fixed_policy=args.manual_fixed_policy,
            )
        ]
    else:
        if args.condition:
            raise ValueError(
                "--condition requires --partial_graph_json or --partial_graph_file"
            )
        modes = parse_graph_completion_modes(args.modes)
        rows = sample_graph_completion_tasks(
            args.dataset,
            split=args.split,
            modes=modes,
            num_tasks=args.num_tasks,
            seed=args.seed,
            invalid_pair_policy=args.invalid_pair_policy,
        )
    streamed_records = 0
    batch_callback = None
    if args.stream_output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

        def stream_batch(batch: list[dict[str, Any]]) -> None:
            nonlocal streamed_records
            streamed_records += append_graph_completion_samples_jsonl(
                batch, args.output_jsonl
            )
            print(
                "[graph-completion sampling] "
                f"wrote {streamed_records} intermediate response(s) to "
                f"{args.output_jsonl}",
                flush=True,
            )

        batch_callback = stream_batch

    records = generate_graph_completion_samples(
        rows,
        model=args.model,
        tokenizer_model=args.tokenizer_model,
        revision=args.revision,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        enable_thinking=parse_chat_template_enable_thinking(
            args.chat_template_enable_thinking
        ),
        use_cuda_graphs=args.use_cuda_graphs,
        enable_prefix_caching=not args.no_prefix_caching,
        generation_batch_size=args.generation_batch_size,
        show_progress=args.generation_batch_size > 0,
        batch_callback=batch_callback,
    )
    if args.view == "scored":
        records = score_graph_completion_samples(
            records,
            rows,
            reward_config_from_args(args),
        )
    print_graph_completion_samples(records, view=args.view)
    if args.output_text_file:
        path = Path(args.output_text_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            print_graph_completion_samples(records, view=args.view, stream=handle)
    if args.output_jsonl and not args.stream_output_jsonl:
        write_graph_completion_samples_jsonl(records, args.output_jsonl)


if __name__ == "__main__":
    main()
