import json

from graph_completion_parsing import (
    canonicalize_graph,
    extract_final_answer,
    graph_pair_contract_errors,
    parse_graph_canvas,
    render_graph_canvas,
)


def graph():
    return {
        "nodes": [
            {"id": "b", "kind": "protein", "metadata": {"score": 0.8}},
            {"id": "a", "label": "Alpha"},
        ],
        "edges": [
            {
                "source": "a",
                "relation": "activates",
                "target": "b",
                "evidence": "paper-1",
            }
        ],
    }


def test_canonicalization_is_idempotent_and_preserves_extra_fields():
    once = canonicalize_graph(graph())
    twice = canonicalize_graph(once)
    assert once == twice
    assert once["nodes"][1]["metadata"] == {"score": 0.8}
    assert once["edges"][0]["evidence"] == "paper-1"


def test_canvas_round_trip_preserves_payloads():
    canvas = render_graph_canvas(
        graph(), fixed_node_ids=["a"], fixed_edge_keys=["a\tactivates\tb"]
    )
    assert "[FIXED]" in canvas
    assert parse_graph_canvas(canvas) == canonicalize_graph(graph())


def test_strict_answer_uses_final_block_not_reasoning_json():
    output = (
        'Reason about {"nodes": [], "edges": []}.\n'
        '<answer>' + json.dumps(graph()) + '</answer>'
    )
    result = extract_final_answer(output)
    assert result.valid
    assert result.canonical_graph == canonicalize_graph(graph())
    assert "Reason about" in result.reasoning


def test_missing_close_and_trailing_content_are_rejected():
    missing = extract_final_answer('<answer>{"nodes":[],"edges":[]}')
    assert not missing.valid
    assert "missing closing" in missing.errors[0]
    trailing = extract_final_answer(
        '<answer>{"nodes":[],"edges":[]}</answer> trailing'
    )
    assert not trailing.valid
    assert not trailing.termination_valid


def test_evaluation_recovery_is_never_strictly_valid():
    recovered = extract_final_answer(
        '```json\n{"nodes": [], "edges": []}\n```',
        allow_fenced_json_recovery=True,
    )
    assert recovered.fallback_recovered
    assert recovered.json_valid and recovered.schema_valid
    assert not recovered.valid


def test_known_wrong_relation_fixed_conflict_is_invalid():
    target = graph()
    wrong = canonicalize_graph(target)
    wrong["edges"][0]["relation"] = "inhibits"
    wrong_key = "a\tinhibits\tb"
    row = {
        "mode": "wrong_relations",
        "x0": render_graph_canvas(wrong, fixed_node_ids=["a", "b"], fixed_edge_keys=[wrong_key]),
        "x0_graph_json": json.dumps(wrong),
        "x1_graph_json": json.dumps(target),
        "fixed_node_ids": ["a", "b"],
        "fixed_edge_keys": [wrong_key],
    }
    errors = graph_pair_contract_errors(row)
    assert any("fixed edge" in error and "absent from target" in error for error in errors)


def test_wrong_relation_duplicate_collapse_preserves_endpoint_contract():
    target = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [
            {"source": "a", "relation": "activates", "target": "b"},
            {"source": "a", "relation": "enables", "target": "b"},
        ],
    }
    corrupted = {
        "nodes": target["nodes"],
        "edges": [
            {"source": "a", "relation": "incorrect_relation", "target": "b"},
            {"source": "a", "relation": "incorrect_relation", "target": "b"},
        ],
    }
    row = {
        "mode": "wrong_relations",
        "x0": render_graph_canvas(corrupted, fixed_node_ids=["a", "b"]),
        "x0_graph_json": json.dumps(corrupted),
        "x1_graph_json": json.dumps(target),
        "fixed_node_ids": ["a", "b"],
        "fixed_edge_keys": [],
    }
    assert graph_pair_contract_errors(row) == []
