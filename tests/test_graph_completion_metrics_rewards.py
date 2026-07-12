import json

import pytest

from graph_completion_metrics import compute_graph_metrics, mode_primary_score
from graph_completion_parsing import canonicalize_graph, edge_key
from graph_completion_rewards import (
    RewardConfig,
    make_grpo_reward_function,
    score_graph_completion,
)


TARGET = {
    "nodes": [
        {"id": "A", "kind": "concept"},
        {"id": "B", "kind": "concept"},
        {"id": "C", "kind": "concept"},
    ],
    "edges": [
        {"source": "A", "relation": "r1", "target": "B", "confidence": 1.0},
        {"source": "B", "relation": "r2", "target": "C"},
        {"source": "A", "relation": "r3", "target": "C"},
    ],
}


def answer(graph):
    return f"<answer>{json.dumps(graph)}</answer>"


def row(mode, x0, fixed_nodes=(), fixed_edges=()):
    metrics = compute_graph_metrics(
        x0,
        TARGET,
        x0,
        fixed_node_ids=list(fixed_nodes),
        fixed_edge_keys=list(fixed_edges),
    )
    return {
        "mode": mode,
        "x0_graph_json": json.dumps(x0),
        "x1_graph_json": json.dumps(TARGET),
        "fixed_node_ids": list(fixed_nodes),
        "fixed_edge_keys": list(fixed_edges),
        "no_edit_primary": mode_primary_score(mode, metrics),
    }


CASES = {}
empty = {"nodes": [], "edges": []}
CASES["prior_empty"] = (empty, {"nodes": TARGET["nodes"][:1], "edges": []}, [], [])

fixed_x0 = {"nodes": TARGET["nodes"][:1], "edges": []}
fixed_improved = {"nodes": TARGET["nodes"], "edges": TARGET["edges"][:1]}
CASES["fixed_nodes_only"] = (fixed_x0, fixed_improved, ["A"], [])

missing_x0 = {"nodes": TARGET["nodes"], "edges": TARGET["edges"][:1]}
missing_improved = {"nodes": TARGET["nodes"], "edges": TARGET["edges"][:2]}
CASES["missing_edges"] = (
    missing_x0,
    missing_improved,
    ["A", "B", "C"],
    [edge_key(TARGET["edges"][0])],
)

partial_x0 = {"nodes": TARGET["nodes"][:1], "edges": []}
partial_improved = {"nodes": TARGET["nodes"][:2], "edges": TARGET["edges"][:1]}
CASES["partial_subgraph"] = (partial_x0, partial_improved, ["A"], [])

wrong_x0 = canonicalize_graph(TARGET)
wrong_x0["edges"][0]["relation"] = "wrong_1"
wrong_x0["edges"][2]["relation"] = "wrong_2"
wrong_improved = canonicalize_graph(wrong_x0)
wrong_improved["edges"] = [TARGET["edges"][0], wrong_x0["edges"][1], wrong_x0["edges"][2]]
fixed_wrong_edge = edge_key(TARGET["edges"][2])
CASES["wrong_relations"] = (
    wrong_x0,
    wrong_improved,
    ["A", "B", "C"],
    [fixed_wrong_edge],
)

extra_1 = {"source": "C", "relation": "spurious_1", "target": "A"}
extra_2 = {"source": "B", "relation": "spurious_2", "target": "A"}
extra_x0 = {"nodes": TARGET["nodes"], "edges": TARGET["edges"] + [extra_1, extra_2]}
extra_improved = {"nodes": TARGET["nodes"], "edges": TARGET["edges"] + [extra_2]}
CASES["extra_edges"] = (
    extra_x0,
    extra_improved,
    ["A", "B", "C"],
    [edge_key(edge) for edge in TARGET["edges"]],
)


@pytest.mark.parametrize("mode", list(CASES))
def test_reward_ordering_for_every_mode(mode):
    x0, improved, fixed_nodes, fixed_edges = CASES[mode]
    reference = row(mode, x0, fixed_nodes, fixed_edges)
    config = RewardConfig()
    perfect_reward = score_graph_completion(answer(TARGET), reference, config).total
    improved_reward = score_graph_completion(answer(improved), reference, config).total
    unchanged_reward = score_graph_completion(answer(x0), reference, config).total
    malformed_reward = score_graph_completion("<answer>{bad}</answer>", reference, config).total
    assert perfect_reward > improved_reward > unchanged_reward > malformed_reward


def test_metrics_detect_duplicates_dangling_and_exact_payload_mutation():
    prediction = canonicalize_graph(TARGET)
    prediction["nodes"].append(dict(prediction["nodes"][0]))
    prediction["edges"].append({"source": "A", "relation": "dangling", "target": "Z"})
    prediction["nodes"][0]["kind"] = "mutated"
    metrics = compute_graph_metrics(
        prediction,
        TARGET,
        TARGET,
        fixed_node_ids=["A"],
        fixed_edge_keys=[],
    )
    assert metrics["duplicate_nodes"] == 1
    assert metrics["dangling_edges"] == 1
    assert metrics["fixed_node_payload_exact"] == 0
    assert metrics["exact_canonical_match"] == 0


def test_trailing_content_gets_no_semantic_reward():
    reference = row("prior_empty", {"nodes": [], "edges": []})
    result = score_graph_completion(answer(TARGET) + " trailing", reference, RewardConfig())
    assert result.total <= 0.10
    assert result.components["node"] == 0


def test_trl_batch_signature_and_metric_hook():
    reference = row("prior_empty", {"nodes": [], "edges": []})
    logged = {}
    reward_fn = make_grpo_reward_function(RewardConfig())
    rewards = reward_fn(
        completions=[answer(TARGET), "<answer>{bad}</answer>"],
        completion_ids=[[1, 2], [1]],
        mode=[reference["mode"], reference["mode"]],
        x0_graph_json=[reference["x0_graph_json"]] * 2,
        x1_graph_json=[reference["x1_graph_json"]] * 2,
        fixed_node_ids=[[], []],
        fixed_edge_keys=[[], []],
        no_edit_primary=[reference["no_edit_primary"]] * 2,
        log_metric=lambda name, value: logged.setdefault(name, value),
    )
    assert len(rewards) == 2
    assert rewards[0] > rewards[1]
    assert "graph_completion/reward" in logged
