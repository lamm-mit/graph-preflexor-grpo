"""Side-effect-free deterministic rewards for direct graph completion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Mapping, Optional

from graph_completion_metrics import compute_graph_metrics, mode_primary_score
from graph_completion_parsing import extract_final_answer, parse_graph_json


@dataclass
class RewardWeights:
    format_parse: float = 0.10
    schema_structure: float = 0.10
    fixed_contract: float = 0.15
    node: float = 0.10
    edge: float = 0.15
    mode_primary: float = 0.15
    improvement: float = 0.10
    exact_match: float = 0.15

    def validate(self) -> None:
        values = list(self.__dict__.values())
        if any(value < 0 for value in values):
            raise ValueError("reward weights must be non-negative")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError(f"positive shaped reward weights must sum to 1.0, got {sum(values):.6f}")


@dataclass
class RewardConfig:
    stage: str = "shaped"
    weights: RewardWeights = field(default_factory=RewardWeights)
    forbidden_change_penalty: float = 0.35
    duplicate_penalty: float = 0.05
    dangling_penalty: float = 0.10
    spurious_content_penalty: float = 0.10
    no_op_penalty: float = 0.15
    truncation_penalty: float = 0.15
    excessive_length_penalty: float = 0.05
    fixed_failure_cap: float = 0.25
    max_completion_length: Optional[int] = None
    preferred_completion_length: Optional[int] = None

    def validate(self) -> None:
        if self.stage not in {"format", "shaped", "exact"}:
            raise ValueError("reward stage must be format, shaped, or exact")
        self.weights.validate()


@dataclass
class RewardResult:
    total: float
    components: dict[str, float]
    metrics: dict[str, float]
    errors: list[str]


def add_reward_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reward_stage", choices=["format", "shaped", "exact"], default="shaped")
    for name, default in RewardWeights().__dict__.items():
        parser.add_argument(f"--reward_weight_{name}", type=float, default=default)
    defaults = RewardConfig()
    parser.add_argument("--reward_forbidden_change_penalty", type=float, default=defaults.forbidden_change_penalty)
    parser.add_argument("--reward_duplicate_penalty", type=float, default=defaults.duplicate_penalty)
    parser.add_argument("--reward_dangling_penalty", type=float, default=defaults.dangling_penalty)
    parser.add_argument("--reward_spurious_content_penalty", type=float, default=defaults.spurious_content_penalty)
    parser.add_argument("--reward_no_op_penalty", type=float, default=defaults.no_op_penalty)
    parser.add_argument("--reward_truncation_penalty", type=float, default=defaults.truncation_penalty)
    parser.add_argument("--reward_excessive_length_penalty", type=float, default=defaults.excessive_length_penalty)
    parser.add_argument("--reward_fixed_failure_cap", type=float, default=defaults.fixed_failure_cap)
    parser.add_argument("--preferred_completion_length", type=int, default=None)


def reward_config_from_args(args: argparse.Namespace) -> RewardConfig:
    weights = RewardWeights(
        **{
            name: getattr(args, f"reward_weight_{name}")
            for name in RewardWeights().__dict__
        }
    )
    config = RewardConfig(
        stage=args.reward_stage,
        weights=weights,
        forbidden_change_penalty=args.reward_forbidden_change_penalty,
        duplicate_penalty=args.reward_duplicate_penalty,
        dangling_penalty=args.reward_dangling_penalty,
        spurious_content_penalty=args.reward_spurious_content_penalty,
        no_op_penalty=args.reward_no_op_penalty,
        truncation_penalty=args.reward_truncation_penalty,
        excessive_length_penalty=args.reward_excessive_length_penalty,
        fixed_failure_cap=args.reward_fixed_failure_cap,
        max_completion_length=getattr(args, "max_completion_length", None),
        preferred_completion_length=args.preferred_completion_length,
    )
    config.validate()
    return config


def _format_component(extraction: Any, structural_validity: float) -> float:
    return (
        0.20 * float(extraction.answer_tags_valid)
        + 0.15 * float(extraction.termination_valid)
        + 0.30 * float(extraction.json_valid)
        + 0.20 * float(extraction.schema_valid)
        + 0.15 * structural_validity
    )


def score_graph_completion(
    completion: Any,
    row: Mapping[str, Any],
    config: RewardConfig,
    *,
    completion_token_count: Optional[int] = None,
) -> RewardResult:
    """Score one completion. Invalid output never receives semantic reward."""

    config.validate()
    extraction = extract_final_answer(completion)
    metrics: dict[str, float] = {}
    structural_validity = 0.0
    if extraction.graph is not None and extraction.schema_valid:
        target = parse_graph_json(row["x1_graph_json"])
        x0 = parse_graph_json(row["x0_graph_json"])
        metrics = compute_graph_metrics(
            extraction.graph,
            target,
            x0,
            fixed_node_ids=list(row.get("fixed_node_ids", []) or []),
            fixed_edge_keys=list(row.get("fixed_edge_keys", []) or []),
        )
        structural_validity = metrics["structural_validity"]
    format_score = _format_component(extraction, structural_validity)
    components: dict[str, float] = {
        "format_parse": format_score,
        "schema_structure": 0.0,
        "fixed_contract": 0.0,
        "node": 0.0,
        "edge": 0.0,
        "mode_primary": 0.0,
        "improvement": 0.0,
        "exact_match": 0.0,
        "penalty_forbidden_change": 0.0,
        "penalty_duplicate": 0.0,
        "penalty_dangling": 0.0,
        "penalty_spurious_content": 0.0,
        "penalty_no_op": 0.0,
        "penalty_truncation": 0.0,
        "penalty_excessive_length": 0.0,
    }
    if config.stage == "format":
        return RewardResult(format_score, components, metrics, extraction.errors)

    if not extraction.valid:
        # Only a small format signal survives. Semantic components remain zero.
        return RewardResult(0.10 * format_score, components, metrics, extraction.errors)

    fixed = metrics["fixed_object_exact"]
    components.update(
        {
            "schema_structure": 0.5 * metrics["schema_valid"] + 0.5 * metrics["structural_validity"],
            "fixed_contract": fixed,
            "node": metrics["node_f1"],
            "edge": metrics["edge_f1"],
            "mode_primary": mode_primary_score(str(row["mode"]), metrics),
            "exact_match": metrics["exact_canonical_match"],
        }
    )
    baseline = float(row.get("no_edit_primary", 0.0))
    improvement = (components["mode_primary"] - baseline) / max(0.10, 1.0 - baseline)
    components["improvement"] = max(-1.0, min(1.0, improvement))

    if config.stage == "exact":
        total = (
            0.80 * metrics["exact_canonical_match"]
            + 0.05 * float(extraction.answer_tags_valid and extraction.termination_valid)
            + 0.05 * metrics["schema_valid"]
            + 0.05 * metrics["structural_validity"]
            + 0.05 * fixed
        )
    else:
        weights = config.weights
        total = sum(
            getattr(weights, name) * components[name]
            for name in weights.__dict__
        )

    components["penalty_forbidden_change"] = config.forbidden_change_penalty * (1.0 - fixed)
    duplicate_count = metrics["duplicate_nodes"] + metrics["duplicate_edges"]
    components["penalty_duplicate"] = config.duplicate_penalty * min(1.0, duplicate_count)
    components["penalty_dangling"] = config.dangling_penalty * min(1.0, metrics["dangling_edges"])
    target_size = max(1.0, len(parse_graph_json(row["x1_graph_json"])["nodes"]) + len(parse_graph_json(row["x1_graph_json"])["edges"]))
    spurious_fraction = min(1.0, (metrics["spurious_nodes"] + metrics["spurious_edges"]) / target_size)
    components["penalty_spurious_content"] = config.spurious_content_penalty * spurious_fraction
    if metrics["no_edit_match"] and not metrics["exact_canonical_match"]:
        components["penalty_no_op"] = config.no_op_penalty
    if (
        completion_token_count is not None
        and config.max_completion_length is not None
        and completion_token_count >= config.max_completion_length
        and not extraction.termination_valid
    ):
        components["penalty_truncation"] = config.truncation_penalty
    if (
        completion_token_count is not None
        and config.preferred_completion_length is not None
        and completion_token_count > config.preferred_completion_length
    ):
        overflow = completion_token_count / max(1, config.preferred_completion_length) - 1.0
        components["penalty_excessive_length"] = config.excessive_length_penalty * min(1.0, overflow)
    total -= sum(value for name, value in components.items() if name.startswith("penalty_"))
    if fixed < 1.0:
        total = min(total, config.fixed_failure_cap)
    return RewardResult(max(-1.0, min(1.0, total)), components, metrics, extraction.errors)


def make_grpo_reward_function(config: RewardConfig):
    """Return a TRL-compatible batched reward function with trainer-native logging."""

    def reward_function(
        completions: list[Any],
        completion_ids: Optional[list[list[int]]] = None,
        log_metric: Optional[Any] = None,
        log_extra: Optional[Any] = None,
        **columns: Any,
    ) -> list[float]:
        results: list[RewardResult] = []
        for index, completion in enumerate(completions):
            row = {
                key: values[index]
                for key, values in columns.items()
                if key not in {"prompts", "trainer_state", "log_metric", "log_extra"}
                and isinstance(values, list)
                and index < len(values)
            }
            token_count = None
            if completion_ids is not None and index < len(completion_ids):
                token_count = len(completion_ids[index])
            results.append(
                score_graph_completion(
                    completion, row, config, completion_token_count=token_count
                )
            )
        if results and callable(log_metric):
            log_metric("graph_completion/reward", mean(result.total for result in results))
            component_names = sorted(results[0].components)
            for name in component_names:
                log_metric(
                    f"graph_completion/{name}",
                    mean(result.components[name] for result in results),
                )
            metric_names = sorted(set().union(*(result.metrics for result in results)))
            for name in metric_names:
                values = [result.metrics[name] for result in results if name in result.metrics]
                if values:
                    log_metric(f"graph_completion_metric/{name}", mean(values))
        if results and callable(log_extra):
            log_extra("graph_validation_errors", ["; ".join(result.errors) for result in results])
        return [result.total for result in results]

    reward_function.__name__ = f"graph_completion_{config.stage}_reward"
    return reward_function
