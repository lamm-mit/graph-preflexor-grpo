"""Lossless graph-canvas parsing and strict graph-completion answer extraction.

The graph-canvas helpers are method-neutral adaptations of the MIT-licensed
implementation in DiscoverydLLM/examples/editflow_graph_canvas/graph_canvas.py.
No EditFlow or diffusion code is included here.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


GRAPH_COMPLETION_MODES = (
    "prior_empty",
    "fixed_nodes_only",
    "missing_edges",
    "partial_subgraph",
    "wrong_relations",
    "extra_edges",
)


def edge_key(edge: Mapping[str, Any]) -> str:
    """Return the dataset's semantic edge identity."""

    return "\t".join(
        str(edge.get(name, "")).strip()
        for name in ("source", "relation", "target")
    )


def _compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_graph_json(value: Any) -> dict[str, Any]:
    """Parse a graph JSON field while preserving every node/edge payload field."""

    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("graph must be a JSON object")
    return value


def validate_graph_schema(graph: Any) -> list[str]:
    """Validate required graph structure without stripping optional fields."""

    errors: list[str] = []
    if not isinstance(graph, dict):
        return ["graph is not a JSON object"]
    for collection in ("nodes", "edges"):
        if collection not in graph:
            errors.append(f"missing required {collection!r} array")
        elif not isinstance(graph[collection], list):
            errors.append(f"{collection!r} is not an array")
    if errors:
        return errors

    for index, node in enumerate(graph["nodes"]):
        if not isinstance(node, dict):
            errors.append(f"nodes[{index}] is not an object")
            continue
        if not str(node.get("id", "")).strip():
            errors.append(f"nodes[{index}] has no non-empty id")
    for index, edge in enumerate(graph["edges"]):
        if not isinstance(edge, dict):
            errors.append(f"edges[{index}] is not an object")
            continue
        for key in ("source", "relation", "target"):
            if not str(edge.get(key, "")).strip():
                errors.append(f"edges[{index}] has no non-empty {key}")
    return errors


def audit_raw_graph(graph: Any) -> dict[str, float]:
    """Report defects before canonicalization can hide them."""

    schema_errors = validate_graph_schema(graph)
    if not isinstance(graph, dict):
        return {
            "schema_valid": 0.0,
            "duplicate_nodes": 0.0,
            "duplicate_edges": 0.0,
            "dangling_edges": 0.0,
            "structural_validity": 0.0,
        }
    nodes = graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges", []) if isinstance(graph.get("edges"), list) else []
    node_ids = [str(n.get("id", "")).strip() for n in nodes if isinstance(n, dict)]
    edge_keys = [edge_key(e) for e in edges if isinstance(e, dict)]
    node_counts = Counter(node_ids)
    edge_counts = Counter(edge_keys)
    duplicates_n = sum(max(0, count - 1) for key, count in node_counts.items() if key)
    duplicates_e = sum(max(0, count - 1) for key, count in edge_counts.items() if key.strip("\t"))
    node_set = {node_id for node_id in node_ids if node_id}
    dangling = sum(
        str(edge.get("source", "")).strip() not in node_set
        or str(edge.get("target", "")).strip() not in node_set
        for edge in edges
        if isinstance(edge, dict)
    )
    schema_valid = not schema_errors
    return {
        "schema_valid": float(schema_valid),
        "duplicate_nodes": float(duplicates_n),
        "duplicate_edges": float(duplicates_e),
        "dangling_edges": float(dangling),
        "structural_validity": float(
            schema_valid and duplicates_n == 0 and duplicates_e == 0 and dangling == 0
        ),
    }


def canonicalize_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic graph while retaining node/edge extra fields.

    Validation and raw-structure auditing must happen before this operation.
    The first occurrence of a duplicate semantic object is retained.
    """

    if not isinstance(graph, Mapping):
        raise TypeError("graph must be a mapping")
    node_by_id: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes", []) or []:
        if not isinstance(node, Mapping):
            continue
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        clean = {str(key): value for key, value in node.items()}
        clean["id"] = node_id
        node_by_id.setdefault(node_id, clean)

    edge_by_key: dict[str, dict[str, Any]] = {}
    for edge in graph.get("edges", []) or []:
        if not isinstance(edge, Mapping):
            continue
        clean = {str(key): value for key, value in edge.items()}
        for key in ("source", "relation", "target"):
            clean[key] = str(clean.get(key, "")).strip()
        if not all(clean[key] for key in ("source", "relation", "target")):
            continue
        edge_by_key.setdefault(edge_key(clean), clean)

    return {
        "nodes": sorted(node_by_id.values(), key=lambda item: str(item["id"])),
        "edges": sorted(
            edge_by_key.values(),
            key=lambda item: (
                str(item["source"]),
                str(item["relation"]),
                str(item["target"]),
                _compact_json(item),
            ),
        ),
    }


def graph_to_json_text(graph: Mapping[str, Any]) -> str:
    return json.dumps(canonicalize_graph(graph), ensure_ascii=False, indent=2)


def render_graph_canvas(
    graph: Mapping[str, Any],
    *,
    fixed_node_ids: Iterable[str] = (),
    fixed_edge_keys: Iterable[str] = (),
) -> str:
    fixed_nodes = set(map(str, fixed_node_ids))
    fixed_edges = set(map(str, fixed_edge_keys))
    canonical = canonicalize_graph(graph)
    lines = ["<graph_canvas>", "<nodes>"]
    for node in canonical["nodes"]:
        prefix = "[FIXED] " if str(node["id"]) in fixed_nodes else ""
        lines.append(f"{prefix}N {_compact_json(node)}")
    lines.extend(["</nodes>", "<edges>"])
    for edge in canonical["edges"]:
        prefix = "[FIXED] " if edge_key(edge) in fixed_edges else ""
        lines.append(f"{prefix}E {_compact_json(edge)}")
    lines.extend(["</edges>", "</graph_canvas>"])
    return "\n".join(lines)


def parse_graph_canvas(text: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("[FIXED]"):
            line = line[len("[FIXED]") :].strip()
        if line.startswith("N "):
            value = json.loads(line[2:].strip())
            if not isinstance(value, dict):
                raise ValueError("graph-canvas node line is not a JSON object")
            nodes.append(value)
        elif line.startswith("E "):
            value = json.loads(line[2:].strip())
            if not isinstance(value, dict):
                raise ValueError("graph-canvas edge line is not a JSON object")
            edges.append(value)
    graph = {"nodes": nodes, "edges": edges}
    errors = validate_graph_schema(graph)
    if errors:
        raise ValueError("invalid graph canvas: " + "; ".join(errors))
    return canonicalize_graph(graph)


@dataclass
class AnswerExtraction:
    raw_output: str
    reasoning: str = ""
    answer_text: Optional[str] = None
    graph: Optional[dict[str, Any]] = None
    canonical_graph: Optional[dict[str, Any]] = None
    errors: list[str] = field(default_factory=list)
    answer_tags_valid: bool = False
    termination_valid: bool = False
    json_valid: bool = False
    schema_valid: bool = False
    fallback_recovered: bool = False

    @property
    def valid(self) -> bool:
        return bool(
            self.answer_tags_valid
            and self.termination_valid
            and self.json_valid
            and self.schema_valid
            and not self.fallback_recovered
        )


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_final_answer(
    output: Any,
    *,
    allow_fenced_json_recovery: bool = False,
) -> AnswerExtraction:
    """Extract only the final complete answer block, never JSON from reasoning."""

    if isinstance(output, list):
        # Defensive support for conversational TRL completion payloads.
        output = "".join(
            str(item.get("content", "")) if isinstance(item, dict) else str(item)
            for item in output
        )
    raw = str(output or "")
    result = AnswerExtraction(raw_output=raw)
    matches = list(_ANSWER_RE.finditer(raw))
    if not matches:
        if "<answer>" in raw:
            result.errors.append("missing closing </answer> tag")
        else:
            result.errors.append("missing <answer> block")
        if allow_fenced_json_recovery:
            fenced = list(_FENCED_JSON_RE.finditer(raw))
            if fenced:
                result.answer_text = fenced[-1].group(1).strip()
                result.fallback_recovered = True
        if result.answer_text is None:
            return result
    else:
        match = matches[-1]
        result.reasoning = raw[: match.start()]
        result.answer_text = match.group(1).strip()
        result.answer_tags_valid = True
        trailing = raw[match.end() :]
        result.termination_valid = not trailing.strip()
        if not result.termination_valid:
            result.errors.append("non-whitespace content follows </answer>")

    try:
        parsed = json.loads(result.answer_text or "")
    except Exception as exc:
        result.errors.append(f"answer JSON parse failed: {exc}")
        return result
    if not isinstance(parsed, dict):
        result.errors.append("answer JSON is not an object")
        return result
    result.graph = parsed
    result.json_valid = True
    schema_errors = validate_graph_schema(parsed)
    if schema_errors:
        result.errors.extend(schema_errors)
        return result
    result.schema_valid = True
    result.canonical_graph = canonicalize_graph(parsed)
    return result


def _node_map(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): dict(node) for node in graph.get("nodes", [])}


def _edge_map(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {edge_key(edge): dict(edge) for edge in graph.get("edges", [])}


def graph_pair_contract_errors(row: Mapping[str, Any]) -> list[str]:
    """Audit dataset metadata, frozen payloads, and the six corruption modes."""

    errors: list[str] = []
    mode = str(row.get("mode", ""))
    if mode not in GRAPH_COMPLETION_MODES:
        errors.append(f"unknown mode {mode!r}")
    try:
        raw_x0 = parse_graph_json(row.get("x0_graph_json"))
        raw_target = parse_graph_json(row.get("x1_graph_json"))
        x0_schema_errors = validate_graph_schema(raw_x0)
        target_schema_errors = validate_graph_schema(raw_target)
        if x0_schema_errors:
            errors.extend(f"x0 schema: {error}" for error in x0_schema_errors)
        if target_schema_errors:
            errors.extend(f"target schema: {error}" for error in target_schema_errors)
        x0 = canonicalize_graph(raw_x0)
        target = canonicalize_graph(raw_target)
    except Exception as exc:
        return [f"graph metadata is not parseable: {exc}"]
    if "x0" in row:
        try:
            if parse_graph_canvas(str(row["x0"])) != x0:
                errors.append("x0 canvas does not match x0_graph_json")
        except Exception as exc:
            errors.append(f"x0 canvas is invalid: {exc}")
    if "x1" in row:
        try:
            if parse_graph_canvas(str(row["x1"])) != target:
                errors.append("x1 canvas does not match x1_graph_json")
        except Exception as exc:
            errors.append(f"x1 canvas is invalid: {exc}")

    x0_nodes, target_nodes = _node_map(x0), _node_map(target)
    x0_edges, target_edges = _edge_map(x0), _edge_map(target)
    fixed_nodes = set(map(str, row.get("fixed_node_ids", []) or []))
    fixed_edges = set(map(str, row.get("fixed_edge_keys", []) or []))
    for node_id in fixed_nodes:
        if node_id not in x0_nodes:
            errors.append(f"fixed node {node_id!r} is absent from x0")
        elif node_id not in target_nodes:
            errors.append(f"fixed node {node_id!r} is absent from target")
        elif x0_nodes[node_id] != target_nodes[node_id]:
            errors.append(f"fixed node {node_id!r} payload changes in target")
    for key in fixed_edges:
        if key not in x0_edges:
            errors.append(f"fixed edge {key!r} is absent from x0")
        elif key not in target_edges:
            errors.append(f"fixed edge {key!r} is absent from target")
        elif x0_edges[key] != target_edges[key]:
            errors.append(f"fixed edge {key!r} payload changes in target")

    x0_node_keys, target_node_keys = set(x0_nodes), set(target_nodes)
    x0_edge_keys, target_edge_keys = set(x0_edges), set(target_edges)
    if mode == "prior_empty" and (x0_node_keys or x0_edge_keys or fixed_nodes or fixed_edges):
        errors.append("prior_empty x0 or fixed metadata is not empty")
    elif mode == "fixed_nodes_only":
        if x0_edge_keys:
            errors.append("fixed_nodes_only x0 contains edges")
        if not x0_node_keys <= target_node_keys or fixed_nodes != x0_node_keys:
            errors.append("fixed_nodes_only nodes are not an exactly fixed target subset")
    elif mode == "missing_edges":
        if x0_nodes != target_nodes:
            errors.append("missing_edges changes node objects")
        if not x0_edge_keys <= target_edge_keys:
            errors.append("missing_edges x0 edges are not a target subset")
    elif mode == "partial_subgraph":
        if not x0_node_keys <= target_node_keys or not x0_edge_keys <= target_edge_keys:
            errors.append("partial_subgraph x0 is not a target subgraph")
    elif mode == "wrong_relations":
        if x0_nodes != target_nodes:
            errors.append("wrong_relations changes node objects")
        x0_endpoints = Counter((edge["source"], edge["target"]) for edge in x0_edges.values())
        target_endpoints = Counter((edge["source"], edge["target"]) for edge in target_edges.values())
        if x0_endpoints != target_endpoints:
            errors.append("wrong_relations changes edge endpoints")
    elif mode == "extra_edges":
        if x0_nodes != target_nodes:
            errors.append("extra_edges changes node objects")
        if not target_edge_keys <= x0_edge_keys:
            errors.append("extra_edges target is not an x0 edge subset")
    return errors
