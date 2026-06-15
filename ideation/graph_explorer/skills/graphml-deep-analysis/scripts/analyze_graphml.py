#!/usr/bin/env python3
"""Deterministic GraphML evidence pass for graphml-deep-analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover - dependency guidance for hosted shells
    raise SystemExit(
        "This script requires networkx. Install it with `python -m pip install networkx` "
        f"and retry. Import error: {exc}"
    )


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "how",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "through", "to", "via", "with", "without", "using", "use", "between", "based",
    "concept", "concepts", "idea", "ideas", "system", "systems", "approach",
}


def _short(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "..."


def _label(G: nx.Graph, node: Any) -> str:
    attrs = G.nodes[node] if node in G else {}
    for key in ("label", "name", "title", "text"):
        value = attrs.get(key)
        if value not in (None, ""):
            return str(value)
    return str(node)


def _edge_relation(data: Mapping[str, Any]) -> str:
    if not isinstance(data, Mapping):
        return "related_to"
    value = data.get("relation", data.get("label", data.get("type", "related_to")))
    return str(value or "related_to").strip() or "related_to"


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _iter_attr(attrs: Mapping[str, Any]) -> Optional[int]:
    return _as_int(attrs.get("iter"), None)


def _depth_attr(attrs: Mapping[str, Any]) -> Optional[int]:
    return _as_int(attrs.get("depth"), None)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return str(value)


def _as_simple_graph(G: nx.Graph) -> Tuple[nx.Graph, Dict[str, Any]]:
    meta = {
        "input_nodes": G.number_of_nodes(),
        "input_edges": G.number_of_edges(),
        "input_directed": G.is_directed(),
        "input_multigraph": G.is_multigraph(),
    }
    if not G.is_multigraph():
        return G.copy(), meta

    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for edge in G.edges(keys=True, data=True):
        u, v, _key, data = edge
        rel = _edge_relation(data)
        if H.has_edge(u, v):
            old = H[u][v]
            old["multiplicity"] = int(old.get("multiplicity", 1)) + 1
            rels = {part.strip() for part in str(old.get("relation", "")).split(";") if part.strip()}
            rels.add(rel)
            old["relation"] = "; ".join(sorted(rels)) or rel
        else:
            attrs = dict(data)
            attrs["relation"] = rel
            attrs["multiplicity"] = 1
            H.add_edge(u, v, **attrs)
    meta["simplified_edges"] = H.number_of_edges()
    return H, meta


def _undirected_simple(G: nx.Graph) -> nx.Graph:
    U = nx.Graph()
    U.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        if U.has_edge(u, v):
            old = U[u][v]
            old["multiplicity"] = int(old.get("multiplicity", 1)) + 1
            rels = {part.strip() for part in str(old.get("relation", "")).split(";") if part.strip()}
            rels.add(_edge_relation(data))
            old["relation"] = "; ".join(sorted(rels))
        else:
            attrs = dict(data)
            attrs["relation"] = _edge_relation(data)
            attrs["multiplicity"] = int(attrs.get("multiplicity", 1) or 1)
            U.add_edge(u, v, **attrs)
    return U


def _component_sets(G: nx.Graph) -> List[set]:
    if G.number_of_nodes() == 0:
        return []
    if G.is_directed():
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))
    return sorted((set(c) for c in comps), key=len, reverse=True)


def _iter_summary(values: Iterable[Optional[int]]) -> Dict[str, Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"min": None, "median": None, "max": None}
    return {"min": int(min(vals)), "median": float(statistics.median(vals)), "max": int(max(vals))}


def _tokens(text: str) -> List[str]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text))
    text = re.sub(r"[_/.-]+", " ", text)
    return [
        tok for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
        if tok not in STOPWORDS
    ]


def _top_terms(labels: Iterable[str], top: int = 12) -> List[Tuple[str, int]]:
    counts = Counter()
    for label in labels:
        counts.update(_tokens(label))
    return counts.most_common(top)


def _safe_density(G: nx.Graph) -> float:
    try:
        return float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0
    except Exception:
        return 0.0


def _centralities(G: nx.Graph, approx_above: int, pivots: int) -> Dict[str, Dict[Any, float]]:
    U = _undirected_simple(G)
    n = G.number_of_nodes()
    degree = {node: float(deg) for node, deg in G.degree()}
    in_degree = {node: float(G.in_degree(node)) for node in G.nodes()} if G.is_directed() else {}
    out_degree = {node: float(G.out_degree(node)) for node in G.nodes()} if G.is_directed() else {}

    if G.number_of_edges():
        try:
            pagerank = nx.pagerank(G)
        except Exception:
            pagerank = nx.pagerank(U) if U.number_of_edges() else {}
    else:
        pagerank = {}
    if not pagerank:
        pagerank = {node: 1.0 / max(1, n) for node in G.nodes()}

    if n <= 2 or U.number_of_edges() == 0:
        betweenness = {node: 0.0 for node in G.nodes()}
    elif n > approx_above:
        betweenness = nx.betweenness_centrality(U, k=min(max(1, pivots), n), seed=0)
    else:
        betweenness = nx.betweenness_centrality(U)

    if n <= approx_above:
        try:
            closeness = nx.closeness_centrality(U)
        except Exception:
            closeness = {node: 0.0 for node in G.nodes()}
    else:
        closeness = {}

    return {
        "degree": degree,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "pagerank": pagerank,
        "betweenness": betweenness,
        "closeness": closeness,
    }


def _ranked(metric: Mapping[Any, float], G: nx.Graph, top: int) -> List[Dict[str, Any]]:
    rows = []
    for node, value in sorted(metric.items(), key=lambda item: item[1], reverse=True)[:top]:
        attrs = G.nodes[node]
        rows.append({
            "id": str(node),
            "label": _label(G, node),
            "value": float(value),
            "iter": _iter_attr(attrs),
            "depth": _depth_attr(attrs),
            "question": _short(attrs.get("question", ""), 180),
        })
    return rows


def _communities(G: nx.Graph, centralities: Mapping[str, Mapping[Any, float]], max_modules: int) -> Tuple[List[Dict[str, Any]], Dict[Any, int], float]:
    U = _undirected_simple(G)
    if U.number_of_edges() == 0:
        comms = [{node} for node in U.nodes()]
        modularity = 0.0
    else:
        try:
            comms = [set(c) for c in nx.community.greedy_modularity_communities(U)]
            modularity = float(nx.community.modularity(U, comms))
        except Exception:
            comms = _component_sets(U)
            modularity = float("nan")

    comms.sort(key=len, reverse=True)
    module_of = {node: idx for idx, comm in enumerate(comms) for node in comm}
    profiles = []

    for idx, nodes in enumerate(comms[:max_modules]):
        sub = U.subgraph(nodes)
        boundary = Counter()
        source_questions = Counter()
        for node in nodes:
            outside = [nb for nb in U.neighbors(node) if nb not in nodes]
            if outside:
                boundary[node] = len(outside)
            question = _short(G.nodes[node].get("question", ""), 180)
            if question:
                source_questions[question] += 1

        top_nodes = sorted(
            nodes,
            key=lambda node: (
                centralities["pagerank"].get(node, 0.0),
                centralities["degree"].get(node, 0.0),
                centralities["betweenness"].get(node, 0.0),
            ),
            reverse=True,
        )[:12]

        relation_counts = Counter()
        external_edges = 0
        for u, v, data in G.edges(data=True):
            u_in, v_in = u in nodes, v in nodes
            if u_in and v_in:
                relation_counts[_edge_relation(data)] += 1
            elif u_in != v_in:
                external_edges += 1

        profiles.append({
            "id": idx,
            "size": len(nodes),
            "internal_edges": int(sub.number_of_edges()),
            "external_edges": int(external_edges),
            "iter": _iter_summary(_iter_attr(G.nodes[n]) for n in nodes),
            "depth": _iter_summary(_depth_attr(G.nodes[n]) for n in nodes),
            "top_terms": _top_terms((_label(G, n) for n in nodes), 15),
            "top_nodes": [
                {
                    "id": str(node),
                    "label": _label(G, node),
                    "degree": centralities["degree"].get(node, 0.0),
                    "pagerank": centralities["pagerank"].get(node, 0.0),
                    "betweenness": centralities["betweenness"].get(node, 0.0),
                    "iter": _iter_attr(G.nodes[node]),
                    "question": _short(G.nodes[node].get("question", ""), 160),
                }
                for node in top_nodes
            ],
            "boundary_nodes": [
                {"id": str(node), "label": _label(G, node), "external_neighbors": int(count)}
                for node, count in boundary.most_common(10)
            ],
            "relations": relation_counts.most_common(12),
            "source_questions": source_questions.most_common(8),
        })
    return profiles, module_of, modularity


def _max_normalized(values: Mapping[Any, float]) -> Dict[Any, float]:
    max_value = max((abs(v) for v in values.values()), default=0.0)
    if max_value <= 0:
        return {key: 0.0 for key in values}
    return {key: float(value) / max_value for key, value in values.items()}


def _bridge_nodes(G: nx.Graph, module_of: Mapping[Any, int], centralities: Mapping[str, Mapping[Any, float]], top: int) -> List[Dict[str, Any]]:
    U = _undirected_simple(G)
    bet_norm = _max_normalized(centralities["betweenness"])
    pr_norm = _max_normalized(centralities["pagerank"])
    deg_norm = _max_normalized(centralities["degree"])

    rows = []
    for node in G.nodes():
        neighbor_modules = {
            module_of.get(nb)
            for nb in U.neighbors(node)
            if module_of.get(nb) is not None and module_of.get(nb) != module_of.get(node)
        }
        cross_edges = len(neighbor_modules)
        module_span = len({module_of.get(node)} | neighbor_modules)
        if module_span <= 1 and centralities["betweenness"].get(node, 0.0) <= 0:
            continue
        score = (
            0.42 * bet_norm.get(node, 0.0)
            + 0.22 * pr_norm.get(node, 0.0)
            + 0.18 * deg_norm.get(node, 0.0)
            + 0.12 * min(1.0, cross_edges / 4.0)
            + 0.06 * min(1.0, module_span / 5.0)
        )
        attrs = G.nodes[node]
        rows.append({
            "id": str(node),
            "label": _label(G, node),
            "score": float(score),
            "module": module_of.get(node),
            "neighbor_modules": sorted(m for m in neighbor_modules if m is not None),
            "module_span": int(module_span),
            "degree": centralities["degree"].get(node, 0.0),
            "pagerank": centralities["pagerank"].get(node, 0.0),
            "betweenness": centralities["betweenness"].get(node, 0.0),
            "iter": _iter_attr(attrs),
            "depth": _depth_attr(attrs),
            "question": _short(attrs.get("question", ""), 180),
        })

    rows.sort(key=lambda row: (row["score"], row["betweenness"], row["pagerank"]), reverse=True)
    return rows[:top]


def _critical_connectors(G: nx.Graph, centralities: Mapping[str, Mapping[Any, float]], top: int) -> Dict[str, Any]:
    U = _undirected_simple(G)
    if U.number_of_nodes() == 0:
        return {"articulation_points": [], "bridge_edges": []}
    comps = sorted(nx.connected_components(U), key=len, reverse=True)
    lcc = U.subgraph(comps[0]).copy() if comps else U

    articulation_rows = []
    if lcc.number_of_nodes() > 2:
        base_components = nx.number_connected_components(U)
        for node in nx.articulation_points(lcc):
            H = U.copy()
            H.remove_node(node)
            articulation_rows.append({
                "id": str(node),
                "label": _label(G, node),
                "extra_fragments": int(nx.number_connected_components(H) - base_components),
                "degree": centralities["degree"].get(node, 0.0),
                "betweenness": centralities["betweenness"].get(node, 0.0),
                "iter": _iter_attr(G.nodes[node]),
                "question": _short(G.nodes[node].get("question", ""), 180),
            })
    articulation_rows.sort(key=lambda row: (row["extra_fragments"], row["betweenness"]), reverse=True)

    bridge_edges = []
    try:
        for u, v in nx.bridges(lcc):
            data = U.get_edge_data(u, v) or {}
            bridge_edges.append({
                "source": _label(G, u),
                "target": _label(G, v),
                "relation": _edge_relation(data),
                "source_degree": centralities["degree"].get(u, 0.0),
                "target_degree": centralities["degree"].get(v, 0.0),
            })
        bridge_edges.sort(key=lambda row: row["source_degree"] + row["target_degree"], reverse=True)
    except Exception:
        bridge_edges = []

    return {"articulation_points": articulation_rows[:top], "bridge_edges": bridge_edges[:top]}


def _module_edges(G: nx.Graph, module_of: Mapping[Any, int], top: int) -> List[Dict[str, Any]]:
    counts: Counter = Counter()
    examples: Dict[Tuple[int, int], Tuple[Any, Any, str]] = {}
    for u, v, data in G.edges(data=True):
        mu, mv = module_of.get(u), module_of.get(v)
        if mu is None or mv is None or mu == mv:
            continue
        key = tuple(sorted((int(mu), int(mv))))
        counts[key] += 1
        examples.setdefault(key, (u, v, _edge_relation(data)))

    rows = []
    for (a, b), count in counts.most_common(top):
        u, v, relation = examples[(a, b)]
        rows.append({
            "module_a": a,
            "module_b": b,
            "edge_count": int(count),
            "example": {"source": _label(G, u), "relation": relation, "target": _label(G, v)},
        })
    return rows


def _edge_data_between(G: nx.Graph, u: Any, v: Any) -> Dict[str, Any]:
    if G.has_edge(u, v):
        data = G.get_edge_data(u, v) or {}
        return dict(data) if isinstance(data, Mapping) else {}
    if G.has_edge(v, u):
        data = G.get_edge_data(v, u) or {}
        return dict(data) if isinstance(data, Mapping) else {}
    return {}


def _path_text(G: nx.Graph, path: Sequence[Any]) -> str:
    pieces = []
    for idx, node in enumerate(path):
        if idx:
            rel = _edge_relation(_edge_data_between(G, path[idx - 1], node)).replace("_", " ")
            pieces.append(f"--{rel}-->")
        pieces.append(_label(G, node))
    return " ".join(pieces)


def _representative_paths(
    G: nx.Graph,
    module_of: Mapping[Any, int],
    centralities: Mapping[str, Mapping[Any, float]],
    bridges: Sequence[Mapping[str, Any]],
    max_paths: int,
    max_len: int,
) -> List[Dict[str, Any]]:
    U = _undirected_simple(G)
    if U.number_of_nodes() < 2:
        return []
    node_by_id = {str(node): node for node in G.nodes()}
    bridge_nodes = [node_by_id[row["id"]] for row in bridges if row.get("id") in node_by_id]
    ranked = [node for node, _ in sorted(centralities["pagerank"].items(), key=lambda item: item[1], reverse=True)]
    candidates = []
    seen_nodes = set()
    for node in bridge_nodes + ranked:
        if node not in seen_nodes:
            candidates.append(node)
            seen_nodes.add(node)
        if len(candidates) >= 70:
            break

    paths = []
    seen_paths = set()
    for i, source in enumerate(candidates):
        for target in candidates[i + 1:]:
            if module_of.get(source) == module_of.get(target):
                continue
            try:
                path = nx.shortest_path(U, source, target)
            except Exception:
                continue
            if len(path) < 3 or len(path) > max_len:
                continue
            key = tuple(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            modules = [module_of.get(node) for node in path]
            module_count = len({m for m in modules if m is not None})
            endpoint_score = centralities["pagerank"].get(source, 0.0) + centralities["pagerank"].get(target, 0.0)
            bridge_score = max(centralities["betweenness"].get(node, 0.0) for node in path)
            score = endpoint_score * max(1, module_count) + bridge_score
            paths.append((score, path))

    paths.sort(key=lambda item: item[0], reverse=True)
    rows = []
    for score, path in paths[:max_paths]:
        rows.append({
            "score": float(score),
            "nodes": [_label(G, node) for node in path],
            "node_ids": [str(node) for node in path],
            "modules": [module_of.get(node) for node in path],
            "relations": [_edge_relation(_edge_data_between(G, u, v)) for u, v in zip(path, path[1:])],
            "text": _path_text(G, path),
        })
    return rows


def _query_relevant_nodes(G: nx.Graph, question: str, centralities: Mapping[str, Mapping[Any, float]], module_of: Mapping[Any, int], top: int) -> List[Dict[str, Any]]:
    q_tokens = set(_tokens(question))
    if not q_tokens:
        return []
    rows = []
    pr_norm = _max_normalized(centralities["pagerank"])
    bet_norm = _max_normalized(centralities["betweenness"])
    for node in G.nodes():
        attrs = G.nodes[node]
        haystack = " ".join([
            _label(G, node),
            str(attrs.get("question", "")),
            str(attrs.get("description", "")),
            str(attrs.get("summary", "")),
        ])
        node_tokens = set(_tokens(haystack))
        overlap = sorted(q_tokens & node_tokens)
        if not overlap:
            continue
        score = len(overlap) + 0.25 * pr_norm.get(node, 0.0) + 0.25 * bet_norm.get(node, 0.0)
        rows.append({
            "id": str(node),
            "label": _label(G, node),
            "score": float(score),
            "matched_terms": overlap[:12],
            "module": module_of.get(node),
            "pagerank": centralities["pagerank"].get(node, 0.0),
            "betweenness": centralities["betweenness"].get(node, 0.0),
            "iter": _iter_attr(attrs),
            "question": _short(attrs.get("question", ""), 180),
        })
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:top]


def _quality_flags(G: nx.Graph, U: nx.Graph) -> Dict[str, Any]:
    labels = [_label(G, node) for node in G.nodes()]
    label_counts = Counter(label.lower().strip() for label in labels)
    duplicate_labels = [(label, count) for label, count in label_counts.items() if count > 1]
    generic_labels = [
        label for label in labels
        if not _tokens(label) or label.lower().strip() in STOPWORDS
    ]
    relation_counts = Counter(_edge_relation(data) for _u, _v, data in G.edges(data=True))
    weak_relations = [
        (rel, count) for rel, count in relation_counts.items()
        if rel.lower().replace("_", " ") in {"related to", "related_to", "relates to", "associated with", "links to"}
    ]
    return {
        "isolates": int(nx.number_of_isolates(U)),
        "self_loops": int(nx.number_of_selfloops(G)),
        "duplicate_labels": duplicate_labels[:20],
        "generic_label_examples": generic_labels[:20],
        "weak_relation_counts": weak_relations[:20],
    }


def _global_stats(G: nx.Graph, U: nx.Graph, metadata: Mapping[str, Any], modularity: float, module_count: int) -> Dict[str, Any]:
    comps = _component_sets(G)
    edge_iters = [_iter_attr(data) for _u, _v, data in G.edges(data=True)]
    node_iters = [_iter_attr(G.nodes[node]) for node in G.nodes()]
    return {
        **metadata,
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "directed": bool(G.is_directed()),
        "density": _safe_density(G),
        "undirected_components": int(nx.number_connected_components(U)) if U.number_of_nodes() else 0,
        "weak_components": len(comps),
        "largest_component_nodes": len(comps[0]) if comps else 0,
        "isolates": int(nx.number_of_isolates(U)),
        "self_loops": int(nx.number_of_selfloops(G)),
        "module_count": int(module_count),
        "modularity": float(modularity) if not math.isnan(modularity) else None,
        "node_iter": _iter_summary(node_iters),
        "edge_iter": _iter_summary(edge_iters),
        "relation_counts": Counter(_edge_relation(data) for _u, _v, data in G.edges(data=True)).most_common(20),
    }


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    graph_path = Path(args.graphml).expanduser().resolve()
    raw = nx.read_graphml(graph_path)
    G, metadata = _as_simple_graph(raw)
    U = _undirected_simple(G)
    centralities = _centralities(G, args.approx_above, args.pivots)
    communities, module_of, modularity = _communities(G, centralities, args.max_modules)
    bridge_nodes = _bridge_nodes(G, module_of, centralities, args.top)
    critical = _critical_connectors(G, centralities, args.top)
    module_edges = _module_edges(G, module_of, args.top)
    paths = _representative_paths(G, module_of, centralities, bridge_nodes, args.max_paths, args.max_path_len)
    query_nodes = _query_relevant_nodes(G, args.question or "", centralities, module_of, args.top)

    profile = {
        "source": {
            "graph_path": str(graph_path),
            "filename": graph_path.name,
            "question": args.question or "",
        },
        "global_stats": _global_stats(G, U, metadata, modularity, len(communities)),
        "top_nodes": {
            "degree": _ranked(centralities["degree"], G, args.top),
            "pagerank": _ranked(centralities["pagerank"], G, args.top),
            "betweenness": _ranked(centralities["betweenness"], G, args.top),
            "in_degree": _ranked(centralities["in_degree"], G, args.top) if centralities["in_degree"] else [],
            "out_degree": _ranked(centralities["out_degree"], G, args.top) if centralities["out_degree"] else [],
        },
        "communities": communities,
        "bridge_nodes": bridge_nodes,
        "critical_connectors": critical,
        "module_edges": module_edges,
        "representative_paths": paths,
        "query_relevant_nodes": query_nodes,
        "quality": _quality_flags(G, U),
    }
    return profile


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    if not rows:
        return ["_None found._", ""]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    out.append("")
    return out


def write_markdown(profile: Mapping[str, Any], out_dir: Path) -> Path:
    source = profile["source"]
    stats = profile["global_stats"]
    lines: List[str] = []
    lines.append("# GraphML Deep Analysis")
    lines.append("")
    lines.append(f"- Source: `{source['filename']}`")
    if source.get("question"):
        lines.append(f"- Question: {source['question']}")
    lines.append(
        f"- Graph: {stats['nodes']} nodes, {stats['edges']} edges, "
        f"directed={stats['directed']}, density={_fmt_float(stats['density'])}"
    )
    lines.append(
        f"- Components: {stats['undirected_components']} undirected, "
        f"largest={stats['largest_component_nodes']} nodes, isolates={stats['isolates']}"
    )
    lines.append(
        f"- Modules: {stats['module_count']} detected, modularity={_fmt_float(stats['modularity'])}"
    )
    lines.append(f"- Node iterations: {stats['node_iter']}; edge iterations: {stats['edge_iter']}")
    lines.append("")

    lines.append("## Major Modules")
    module_rows = []
    for comm in profile["communities"][:12]:
        terms = ", ".join(f"{term}({count})" for term, count in comm["top_terms"][:6])
        top_node = comm["top_nodes"][0]["label"] if comm["top_nodes"] else ""
        module_rows.append([
            comm["id"],
            comm["size"],
            comm["internal_edges"],
            comm["external_edges"],
            terms,
            _short(top_node, 80),
        ])
    lines.extend(_md_table(["id", "nodes", "internal", "external", "terms", "top node"], module_rows))

    lines.append("## Bridge Concepts")
    bridge_rows = []
    for row in profile["bridge_nodes"][:15]:
        bridge_rows.append([
            _short(row["label"], 70),
            row["module"],
            ",".join(str(m) for m in row["neighbor_modules"]),
            _fmt_float(row["score"]),
            _fmt_float(row["betweenness"]),
            row["iter"],
        ])
    lines.extend(_md_table(["node", "module", "neighbor modules", "score", "betweenness", "iter"], bridge_rows))

    lines.append("## Cross-Module Edges")
    edge_rows = []
    for row in profile["module_edges"][:15]:
        ex = row["example"]
        edge_rows.append([
            f"{row['module_a']} - {row['module_b']}",
            row["edge_count"],
            _short(f"{ex['source']} --{ex['relation']}--> {ex['target']}", 120),
        ])
    lines.extend(_md_table(["modules", "edges", "example"], edge_rows))

    lines.append("## Representative Relation Chains")
    path_rows = []
    for row in profile["representative_paths"][:12]:
        path_rows.append([
            _fmt_float(row["score"]),
            " -> ".join(str(m) for m in row["modules"]),
            _short(row["text"], 180),
        ])
    lines.extend(_md_table(["score", "modules", "path"], path_rows))

    if profile["query_relevant_nodes"]:
        lines.append("## Query-Relevant Nodes")
        query_rows = []
        for row in profile["query_relevant_nodes"][:12]:
            query_rows.append([
                _short(row["label"], 80),
                row["module"],
                ", ".join(row["matched_terms"][:8]),
                _fmt_float(row["pagerank"]),
                _fmt_float(row["betweenness"]),
            ])
        lines.extend(_md_table(["node", "module", "matched terms", "pagerank", "betweenness"], query_rows))

    lines.append("## Critical Connectors")
    ap_rows = []
    for row in profile["critical_connectors"]["articulation_points"][:12]:
        ap_rows.append([
            _short(row["label"], 80),
            row["extra_fragments"],
            _fmt_float(row["betweenness"]),
            row["iter"],
        ])
    lines.extend(_md_table(["articulation node", "extra fragments", "betweenness", "iter"], ap_rows))

    lines.append("## Quality Flags")
    q = profile["quality"]
    lines.append(f"- Isolates: {q['isolates']}")
    lines.append(f"- Self loops: {q['self_loops']}")
    if q["weak_relation_counts"]:
        weak = ", ".join(f"{rel}={count}" for rel, count in q["weak_relation_counts"][:8])
        lines.append(f"- Generic relation labels: {weak}")
    if q["duplicate_labels"]:
        dupes = ", ".join(f"{label}={count}" for label, count in q["duplicate_labels"][:8])
        lines.append(f"- Duplicate labels: {dupes}")
    if q["generic_label_examples"]:
        lines.append(f"- Generic label examples: {', '.join(q['generic_label_examples'][:8])}")
    lines.append("")

    path = out_dir / "analysis.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphml", help="GraphML file to analyze")
    parser.add_argument("--question", default="", help="User question to bias relevance reporting")
    parser.add_argument("--out", default="graphml_analysis", help="Output directory")
    parser.add_argument("--top", type=int, default=25, help="Rows to keep for ranked outputs")
    parser.add_argument("--max-modules", type=int, default=30)
    parser.add_argument("--max-paths", type=int, default=30)
    parser.add_argument("--max-path-len", type=int, default=7)
    parser.add_argument("--approx-above", type=int, default=1200, help="Approximate betweenness above this node count")
    parser.add_argument("--pivots", type=int, default=250, help="Betweenness pivots for large graphs")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = analyze(args)
    json_path = out_dir / "analysis.json"
    json_path.write_text(json.dumps(_json_safe(profile), indent=2), encoding="utf-8")
    md_path = write_markdown(profile, out_dir)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
