from graph_completion_rewards import RewardConfig
from validate_graph_completion import (
    _build_reference_lookups,
    _completion_token_count,
    _generation_summary,
    _prediction_analysis,
    _resolve_reference,
)


def _row(source, variant, mode):
    return {
        "source_index": source,
        "variant_index": variant,
        "mode": mode,
    }


def test_reference_resolution_uses_mode_and_rejects_ambiguous_legacy_key():
    rows = [
        _row(7, 0, "prior_empty"),
        _row(7, 0, "missing_edges"),
        _row(8, 0, "extra_edges"),
    ]
    full, by_pair = _build_reference_lookups(rows)

    resolved, ambiguous = _resolve_reference(
        _row(7, 0, "missing_edges"),
        full,
        by_pair,
    )
    assert resolved["mode"] == "missing_edges"
    assert not ambiguous

    resolved, ambiguous = _resolve_reference(
        {"source_index": 7, "variant_index": 0},
        full,
        by_pair,
    )
    assert resolved is None
    assert ambiguous

    resolved, ambiguous = _resolve_reference(
        {"source_index": 8, "variant_index": 0},
        full,
        by_pair,
    )
    assert resolved["mode"] == "extra_edges"
    assert not ambiguous


def test_prediction_analysis_is_unconditional_and_generation_stats_keep_lengths():
    config = RewardConfig()
    component_names = list(config.weights.__dict__) + [
        "penalty_forbidden_change",
        "penalty_duplicate",
        "penalty_dangling",
        "penalty_spurious_content",
        "penalty_no_op",
        "penalty_truncation",
        "penalty_excessive_length",
    ]
    zero_components = {name: 0.0 for name in component_names}
    records = [
        {
            "reward": 1.0,
            "components": {**zero_components, "exact_match": 1.0},
            "metrics": {"exact_canonical_match": 1.0},
            "errors": [],
            "completion_token_count": 100,
            "finish_reason": "stop",
            "stop_reason": "</answer>",
        },
        {
            "reward": 0.0,
            "components": zero_components,
            "metrics": {},
            "errors": ["missing <answer> block"],
            "completion_token_count": 4096,
            "finish_reason": "length",
            "stop_reason": None,
        },
    ]

    analysis = _prediction_analysis(records)
    generation = _generation_summary(records, max_completion_length=4096)

    assert analysis["valid_completion_rate"] == 0.5
    assert analysis["graph_metric_coverage"] == 0.5
    assert analysis["component_means_all_completions"]["exact_match"] == 0.5
    assert analysis["metrics_conditional_on_parsed_graph"]["exact_canonical_match"] == 1.0
    assert generation["rows_with_token_counts"] == 2
    assert generation["at_max_completion_length_count"] == 1
    assert generation["finish_reasons"] == {"length": 1, "stop": 1}
    assert _completion_token_count(records[1]) == 4096
