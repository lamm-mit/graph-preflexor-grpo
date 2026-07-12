"""Deterministic exact-match metrics for direct graph completion."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from graph_completion_parsing import audit_raw_graph, canonicalize_graph, edge_key


PRIMARY_METRIC = {
    "prior_empty": "edge_f1",
    "fixed_nodes_only": "edge_recall",
    "missing_edges": "addition_recall",
    "partial_subgraph": "edge_f1",
    "wrong_relations": "relation_repair_recall",
    "extra_edges": "removal_rate",
}


def _ratio(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
    return empty if denominator == 0 else numerator / denominator


def _prf(matched: int, predicted: int, target: int) -> tuple[float, float, float]:
    if predicted == 0 and target == 0:
        return 1.0, 1.0, 1.0
    precision = _ratio(matched, predicted, empty=0.0)
    recall = _ratio(matched, target, empty=1.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _nodes(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): dict(node) for node in graph.get("nodes", [])}


def _edges(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {edge_key(edge): dict(edge) for edge in graph.get("edges", [])}


def _endpoint_map(graph: Mapping[str, Any]) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = {}
    for edge in graph.get("edges", []):
        endpoint = (str(edge["source"]), str(edge["target"]))
        result.setdefault(endpoint, set()).add(str(edge["relation"]))
    return result


def compute_graph_metrics(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    x0: Mapping[str, Any],
    *,
    fixed_node_ids: Optional[list[str]] = None,
    fixed_edge_keys: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute exact semantic and payload-preservation metrics.

    Node identity is ``id`` and edge identity is ``(source, relation, target)``.
    Exact graph/fixed-object comparisons also include every payload field.
    """

    raw_audit = audit_raw_graph(prediction)
    pred = canonicalize_graph(prediction)
    gold = canonicalize_graph(target)
    source = canonicalize_graph(x0)
    p_nodes, g_nodes, x_nodes = _nodes(pred), _nodes(gold), _nodes(source)
    p_edges, g_edges, x_edges = _edges(pred), _edges(gold), _edges(source)

    matched_nodes = len(set(p_nodes) & set(g_nodes))
    matched_edges = len(set(p_edges) & set(g_edges))
    node_precision, node_recall, node_f1 = _prf(matched_nodes, len(p_nodes), len(g_nodes))
    edge_precision, edge_recall, edge_f1 = _prf(matched_edges, len(p_edges), len(g_edges))

    pred_endpoints = _endpoint_map(pred)
    gold_endpoints = _endpoint_map(gold)
    comparable = set(pred_endpoints) & set(gold_endpoints)
    relation_matches = sum(
        len(pred_endpoints[pair] & gold_endpoints[pair]) for pair in comparable
    )
    comparable_pred = sum(len(pred_endpoints[pair]) for pair in comparable)

    added_nodes = set(g_nodes) - set(x_nodes)
    added_edges = set(g_edges) - set(x_edges)
    removed_edges = set(x_edges) - set(g_edges)
    node_add_recall = _ratio(len(added_nodes & set(p_nodes)), len(added_nodes))
    edge_add_recall = _ratio(len(added_edges & set(p_edges)), len(added_edges))
    addition_needed = len(added_nodes) + len(added_edges)
    addition_found = len(added_nodes & set(p_nodes)) + len(added_edges & set(p_edges))
    addition_recall = _ratio(addition_found, addition_needed)
    removal_rate = _ratio(len(removed_edges - set(p_edges)), len(removed_edges))

    x_endpoints = _endpoint_map(source)
    relation_repair_keys = {
        key
        for key, edge in g_edges.items()
        if (str(edge["source"]), str(edge["target"])) in x_endpoints and key not in x_edges
    }
    wrong_relation_keys = {
        key
        for key, edge in x_edges.items()
        if (str(edge["source"]), str(edge["target"])) in gold_endpoints and key not in g_edges
    }
    relation_repair_recall = _ratio(
        len(relation_repair_keys & set(p_edges)), len(relation_repair_keys)
    )
    wrong_relation_removal_rate = _ratio(
        len(wrong_relation_keys - set(p_edges)), len(wrong_relation_keys)
    )

    fixed_nodes = set(map(str, fixed_node_ids or []))
    fixed_edges = set(map(str, fixed_edge_keys or []))
    fixed_node_keep = _ratio(len(fixed_nodes & set(p_nodes)), len(fixed_nodes))
    fixed_edge_keep = _ratio(len(fixed_edges & set(p_edges)), len(fixed_edges))
    fixed_node_payload_exact = _ratio(
        sum(
            node_id in p_nodes
            and node_id in x_nodes
            and p_nodes[node_id] == x_nodes[node_id]
            for node_id in fixed_nodes
        ),
        len(fixed_nodes),
    )
    fixed_edge_payload_exact = _ratio(
        sum(
            key in p_edges and key in x_edges and p_edges[key] == x_edges[key]
            for key in fixed_edges
        ),
        len(fixed_edges),
    )
    fixed_object_exact = min(fixed_node_payload_exact, fixed_edge_payload_exact)

    exact_payload_nodes = sum(
        node_id in p_nodes and p_nodes[node_id] == payload
        for node_id, payload in g_nodes.items()
    )
    exact_payload_edges = sum(
        key in p_edges and p_edges[key] == payload for key, payload in g_edges.items()
    )
    payload_precision = _ratio(
        exact_payload_nodes + exact_payload_edges,
        len(p_nodes) + len(p_edges),
        empty=1.0 if not g_nodes and not g_edges else 0.0,
    )

    metrics = {
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "relation_accuracy": _ratio(relation_matches, comparable_pred, empty=0.0),
        "relation_recall": _ratio(relation_matches, len(g_edges)),
        "relation_repair_recall": relation_repair_recall,
        "wrong_relation_removal_rate": wrong_relation_removal_rate,
        "fixed_node_preservation": fixed_node_keep,
        "fixed_edge_preservation": fixed_edge_keep,
        "fixed_node_payload_exact": fixed_node_payload_exact,
        "fixed_edge_payload_exact": fixed_edge_payload_exact,
        "fixed_object_exact": fixed_object_exact,
        "node_add_recall": node_add_recall,
        "edge_add_recall": edge_add_recall,
        "addition_recall": addition_recall,
        "removal_rate": removal_rate,
        "spurious_nodes": float(len(set(p_nodes) - set(g_nodes))),
        "spurious_edges": float(len(set(p_edges) - set(g_edges))),
        "payload_precision": payload_precision,
        "exact_canonical_match": float(pred == gold and raw_audit["structural_validity"] == 1.0),
        "no_edit_match": float(pred == source),
    }
    metrics.update(raw_audit)
    return metrics


def mode_primary_score(mode: str, metrics: Mapping[str, float]) -> float:
    """Return a stable shaped primary score for one corruption mode."""

    if mode == "prior_empty":
        return (
            0.35 * metrics["node_f1"]
            + 0.45 * metrics["edge_f1"]
            + 0.20 * metrics["relation_accuracy"]
        )
    if mode == "fixed_nodes_only":
        return (
            0.25 * metrics["node_add_recall"]
            + 0.50 * metrics["edge_recall"]
            + 0.25 * metrics["edge_f1"]
        )
    if mode == "missing_edges":
        return (
            0.55 * metrics["edge_add_recall"]
            + 0.30 * metrics["edge_recall"]
            + 0.15 * metrics["edge_precision"]
        )
    if mode == "partial_subgraph":
        return (
            0.25 * metrics["node_add_recall"]
            + 0.35 * metrics["edge_add_recall"]
            + 0.40 * metrics["edge_f1"]
        )
    if mode == "wrong_relations":
        return (
            0.45 * metrics["relation_repair_recall"]
            + 0.35 * metrics["wrong_relation_removal_rate"]
            + 0.20 * metrics["edge_f1"]
        )
    if mode == "extra_edges":
        return (
            0.45 * metrics["removal_rate"]
            + 0.35 * metrics["edge_precision"]
            + 0.20 * metrics["edge_f1"]
        )
    raise ValueError(f"unsupported graph-completion mode: {mode!r}")

