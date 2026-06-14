#!/usr/bin/env python
"""Deep graph profile / audit report for Graph-PRefLexOR or any GraphML file.

The normal plotting scripts answer "how well did the run do?" This script answers
"what is actually in this graph?" It builds a graph atlas with global statistics,
components, modularity communities, top nodes, critical connectors, relation
patterns, provenance, graph-mined insight candidates, data-quality warnings,
figures, and optional LLM-written module summaries.

Examples
--------
    # Profile a Graph-PRefLexOR run directory.
    python profile_graph.py --run runs/exp_leap --out runs/exp_leap/profile

    # Profile any GraphML file.
    python profile_graph.py --graph /path/to/graph.graphml --out graph_profile

    # Add semantic figures/mined bridges by re-embedding labels.
    python profile_graph.py --run runs/exp_leap --embed-model all-MiniLM-L6-v2

    # Use the Responses API with high reasoning effort for summaries and deep dive.
    # Writes report.md, report.pdf, profile.json, and figures/ by default.
    python profile_graph.py --run runs/exp_leap --llm --model gpt-5.5 \
        --reasoning-effort high --deep-pass-tokens 5000 --deep-dive-tokens 12000 \
        --report-review-tokens 10000

    # Local servers without Responses API can use the chat backend.
    python profile_graph.py --graph graph.graphml --llm --backend chat \
        --model meta-llama/Llama-3.2-3B-Instruct --base-url http://localhost:8000/v1 \
        --deep-pass-tokens 5000 --deep-dive-tokens 12000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "under", "via",
    "with", "within", "what", "why", "which", "key", "role", "system", "systems", "process",
    "mechanism", "mechanisms", "effect", "effects", "property", "properties", "material",
    "materials", "concept", "idea", "related", "relation",
    "question", "questions", "unresolved", "underlying", "general",
}

GENERIC_LABELS = {
    "a", "b", "c", "d", "e", "f", "g", "h", "node", "concept", "idea", "thing",
    "entity", "mechanism", "process", "system", "property", "properties", "effect",
    "effects", "response", "interaction", "interactions", "structure", "function",
    "question", "questions", "unresolvedquestions", "underlyingmechanisms", "phenomena",
}


@dataclass
class LLMOptions:
    enabled: bool = False
    backend: str = "responses"
    model: str = "gpt-5.5"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1200
    deep_pass_tokens: int = 5000
    deep_tokens: int = 10000
    reasoning_effort: str = "high"
    deep_passes: int = 4
    report_review: bool = True
    report_review_tokens: int = 8000
    report_review_max_chunks: int = 0
    report_review_chunk_chars: int = 0
    report_review_memo_chars: int = 0
    device: Optional[str] = None
    dtype: str = "auto"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class _Progress:
    def __init__(self, enabled: bool, total: int):
        self.enabled = enabled
        self.total = total
        self.index = 0
        self.started = time.time()

    def step(self, message: str) -> None:
        if not self.enabled:
            return
        self.index += 1
        print(f"[{self.index:02d}/{self.total:02d}] {message}", flush=True)

    def detail(self, message: str) -> None:
        if self.enabled:
            print(f"    {message}", flush=True)

    def finish(self) -> None:
        if self.enabled:
            print(f"[done] completed in {time.time() - self.started:.1f}s", flush=True)


def _short(s: Any, n: int = 80) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: max(0, n - 1)] + "..."


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    return s.strip("_") or "graph"


def _label(G: nx.Graph, n: Any) -> str:
    return str(G.nodes[n].get("label", n))


def _edge_relation(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return "related_to"
    rel = data.get("relation")
    if rel is None:
        rel = data.get("label")
    return str(rel if rel is not None else "related_to").strip() or "related_to"


def _int_attr(attrs: Dict[str, Any], key: str, default: Optional[int] = None) -> Optional[int]:
    try:
        v = attrs.get(key, default)
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _float_attr(attrs: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    try:
        v = attrs.get(key, default)
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _node_iter(G: nx.Graph, n: Any) -> Optional[int]:
    return _int_attr(G.nodes[n], "iter")


def _edge_iter(d: Dict[str, Any]) -> Optional[int]:
    return _int_attr(d, "iter")


def _node_depth(G: nx.Graph, n: Any) -> Optional[int]:
    return _int_attr(G.nodes[n], "depth")


def _as_simple_graph(G: nx.Graph) -> Tuple[nx.Graph, Dict[str, Any]]:
    """Return a simple Graph/DiGraph for analytics, preserving node/edge attrs.

    GraphML from our loop is a DiGraph, but arbitrary GraphML may be multigraph.
    Many analytics are better behaved on a simple graph, so parallel edges are
    folded with a multiplicity counter and a semicolon-separated relation list.
    """
    meta = {"input_nodes": G.number_of_nodes(), "input_edges": G.number_of_edges(),
            "input_multigraph": G.is_multigraph(), "input_directed": G.is_directed()}
    if not G.is_multigraph():
        return G.copy(), meta

    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, _k, d in G.edges(keys=True, data=True):
        rel = _edge_relation(d)
        if H.has_edge(u, v):
            old = H[u][v]
            old["multiplicity"] = int(old.get("multiplicity", 1)) + 1
            rels = set(str(old.get("relation", "related_to")).split("; "))
            rels.add(rel)
            old["relation"] = "; ".join(sorted(r for r in rels if r))
        else:
            nd = dict(d)
            nd["relation"] = rel
            nd["multiplicity"] = 1
            H.add_edge(u, v, **nd)
    meta["simplified_edges"] = H.number_of_edges()
    return H, meta


def _read_transcript(run_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if not run_dir:
        return []
    path = run_dir / "transcript.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"_parse_error": line[:500]})
    return rows


def _read_run_summary(run_dir: Optional[Path]) -> Dict[str, Any]:
    if not run_dir:
        return {}
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _infer_topic(G: nx.Graph, run_summary: Dict[str, Any], transcript: List[Dict[str, Any]]) -> Optional[str]:
    for key in ("topic", "seed_topic", "query"):
        value = run_summary.get(key)
        if value:
            return _short(value, 260)
    for key in ("topic", "seed_topic", "query"):
        value = G.graph.get(key)
        if value:
            return _short(value, 260)
    for row in transcript:
        q = row.get("question")
        if q:
            return _short(q, 260)
    min_iter = None
    questions = Counter()
    for _n, d in G.nodes(data=True):
        it = _int_attr(d, "iter")
        q = d.get("question")
        if q and (min_iter is None or it is not None and it < min_iter):
            min_iter = it
            questions.clear()
        if q and (min_iter is None or it == min_iter):
            questions[_short(q, 260)] += 1
    if questions:
        return questions.most_common(1)[0][0]
    return None


def _read_growth(run_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if not run_dir:
        return []
    path = run_dir / "growth.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _load_graph(graph: Optional[str], run: Optional[str]) -> Tuple[nx.Graph, Dict[str, Any]]:
    if bool(graph) == bool(run):
        raise SystemExit("Provide exactly one of --graph <graph.graphml> or --run <run_dir>.")

    source: Dict[str, Any] = {"loaded_at": _now()}
    if run:
        run_dir = Path(run).expanduser().resolve()
        candidates = []
        gp = run_dir / "graph.graphml"
        if gp.exists():
            candidates.append(gp)
        snap_dir = run_dir / "graphml"
        if snap_dir.exists():
            candidates += sorted(snap_dir.glob("iter_*.graphml"), reverse=True)
        if not candidates:
            raise SystemExit(f"No graph.graphml or graphml/iter_*.graphml found in {run_dir}")
        last_error = None
        for path in candidates:
            try:
                G0 = nx.read_graphml(path)
                source.update({"kind": "run", "run_dir": str(run_dir), "graph_path": str(path)})
                G, meta = _as_simple_graph(G0)
                source.update(meta)
                return G, source
            except Exception as exc:
                last_error = exc
        raise SystemExit(f"Could not read a GraphML snapshot from {run_dir}: {last_error}")

    path = Path(graph).expanduser().resolve()
    G0 = nx.read_graphml(path)
    G, meta = _as_simple_graph(G0)
    source.update({"kind": "graphml", "graph_path": str(path)})
    source.update(meta)
    return G, source


def _get_undirected(G: nx.Graph) -> nx.Graph:
    U = nx.Graph(G.to_undirected())
    U.remove_edges_from(nx.selfloop_edges(U))
    return U


def _weak_components(G: nx.Graph) -> List[set]:
    if G.number_of_nodes() == 0:
        return []
    if G.is_directed():
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))
    comps.sort(key=len, reverse=True)
    return [set(c) for c in comps]


def _safe_density(G: nx.Graph) -> float:
    try:
        return float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0
    except Exception:
        return 0.0


def _top_terms(labels: Sequence[str], top: int = 12) -> List[Tuple[str, int]]:
    c = Counter()
    for label in labels:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(label))
        for w in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", text.lower()):
            if w not in STOPWORDS:
                c[w] += 1
    return c.most_common(top)


def _label_flags(label: str) -> List[str]:
    s = str(label).strip()
    norm = re.sub(r"[^a-z0-9]+", "", s.lower())
    flags = []
    if not s:
        flags.append("empty")
    if len(norm) <= 3 and s not in {"DNA", "RNA"}:
        flags.append("too_short")
    if re.fullmatch(r"[A-Z][a-z]{1,3}", s):
        flags.append("short_fragment")
    if s in GENERIC_LABELS or norm in GENERIC_LABELS:
        flags.append("generic")
    if re.fullmatch(r"[A-Z]{1,3}", s) and s not in {"DNA", "RNA"}:
        flags.append("short_acronym")
    if re.search(r"\d/\d|\d{4,}|(/[0-9]){4,}", s):
        flags.append("numeric_artifact")
    if len(s) > 90:
        flags.append("very_long")
    if len(re.findall(r"[A-Za-z]", s)) < max(1, len(s) * 0.25):
        flags.append("low_alpha")
    return flags


def _mining_label_ok(G: nx.Graph, n: Any) -> bool:
    flags = set(_label_flags(_label(G, n)))
    noisy = {"empty", "too_short", "generic", "short_acronym", "short_fragment",
             "numeric_artifact", "low_alpha"}
    return not (flags & noisy)


def _relation_counts(G: nx.Graph, nodes: Optional[set] = None) -> Counter:
    counts = Counter()
    for u, v, d in G.edges(data=True):
        if nodes is not None and not (u in nodes and v in nodes):
            continue
        counts[_edge_relation(d)] += 1
    return counts


def _external_edge_count(G: nx.Graph, nodes: set) -> int:
    n = 0
    for u, v in G.edges():
        if (u in nodes) != (v in nodes):
            n += 1
    return n


def _subgraph_edge_count(G: nx.Graph, nodes: set) -> int:
    return G.subgraph(nodes).number_of_edges()


def _iter_summary(values: Sequence[Optional[int]]) -> Dict[str, Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"min": None, "max": None, "median": None}
    return {"min": int(min(vals)), "max": int(max(vals)), "median": float(statistics.median(vals))}


def _centralities(G: nx.Graph, approx_above: int = 1500, pivots: int = 400) -> Dict[str, Dict[Any, float]]:
    U = _get_undirected(G)
    n = G.number_of_nodes()
    deg = {node: float(d) for node, d in G.degree()}
    indeg = {node: float(G.in_degree(node)) for node in G} if G.is_directed() else {}
    outdeg = {node: float(G.out_degree(node)) for node in G} if G.is_directed() else {}
    if G.number_of_edges():
        try:
            pagerank = nx.pagerank(G)
        except Exception:
            pagerank = nx.pagerank(U) if U.number_of_edges() else {node: 1.0 / max(1, n) for node in G}
    else:
        pagerank = {node: 1.0 / max(1, n) for node in G}
    if n <= 2:
        bet = {node: 0.0 for node in G}
    elif n > approx_above:
        bet = nx.betweenness_centrality(U, k=min(pivots, n), seed=0)
    else:
        bet = nx.betweenness_centrality(U)
    if n <= approx_above:
        try:
            close = nx.closeness_centrality(U)
        except Exception:
            close = {node: 0.0 for node in G}
    else:
        close = {}
    return {"degree": deg, "in_degree": indeg, "out_degree": outdeg,
            "pagerank": pagerank, "betweenness": bet, "closeness": close}


def _top_items(d: Dict[Any, float], G: nx.Graph, k: int = 20) -> List[Dict[str, Any]]:
    rows = []
    for n, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]:
        rows.append({"id": str(n), "label": _label(G, n), "value": float(v),
                     "iter": _node_iter(G, n), "depth": _node_depth(G, n),
                     "question": _short(G.nodes[n].get("question", ""), 160)})
    return rows


def _component_profiles(G: nx.Graph, cents: Dict[str, Dict[Any, float]], top: int = 20) -> List[Dict[str, Any]]:
    comps = _weak_components(G)
    out = []
    for i, nodes in enumerate(comps[:top]):
        labels = [_label(G, n) for n in nodes]
        top_nodes = sorted(nodes, key=lambda n: cents["degree"].get(n, 0.0), reverse=True)[:10]
        out.append({
            "id": i,
            "size": len(nodes),
            "edges": _subgraph_edge_count(G, nodes),
            "external_edges": _external_edge_count(G, nodes),
            "iter": _iter_summary([_node_iter(G, n) for n in nodes]),
            "depth": _iter_summary([_node_depth(G, n) for n in nodes]),
            "top_terms": _top_terms(labels),
            "top_nodes": [{"id": str(n), "label": _label(G, n),
                           "degree": cents["degree"].get(n, 0.0),
                           "pagerank": cents["pagerank"].get(n, 0.0)} for n in top_nodes],
            "relations": _relation_counts(G, nodes).most_common(12),
        })
    return out


def _community_profiles(G: nx.Graph, cents: Dict[str, Dict[Any, float]], max_modules: int = 30) -> Tuple[List[Dict[str, Any]], Dict[Any, int], float]:
    U = _get_undirected(G)
    if U.number_of_edges() == 0:
        comms = [{n} for n in U.nodes()]
        modularity = 0.0
    else:
        try:
            comms = list(nx.community.greedy_modularity_communities(U))
            modularity = float(nx.community.modularity(U, comms))
        except Exception:
            comms = _weak_components(U)
            modularity = float("nan")
    comms = [set(c) for c in comms]
    comms.sort(key=len, reverse=True)
    module_of = {n: i for i, c in enumerate(comms) for n in c}

    profiles = []
    for i, nodes in enumerate(comms[:max_modules]):
        labels = [_label(G, n) for n in nodes]
        top_nodes = sorted(nodes, key=lambda n: (
            cents["pagerank"].get(n, 0.0), cents["degree"].get(n, 0.0)), reverse=True)[:12]
        boundary = Counter()
        for n in nodes:
            ext = 0
            for nb in U.neighbors(n):
                if nb not in nodes:
                    ext += 1
            if ext:
                boundary[n] = ext
        profiles.append({
            "id": i,
            "size": len(nodes),
            "internal_edges": _subgraph_edge_count(G, nodes),
            "external_edges": _external_edge_count(G, nodes),
            "iter": _iter_summary([_node_iter(G, n) for n in nodes]),
            "depth": _iter_summary([_node_depth(G, n) for n in nodes]),
            "top_terms": _top_terms(labels, top=15),
            "top_nodes": [{"id": str(n), "label": _label(G, n),
                           "degree": cents["degree"].get(n, 0.0),
                           "pagerank": cents["pagerank"].get(n, 0.0),
                           "betweenness": cents["betweenness"].get(n, 0.0),
                           "iter": _node_iter(G, n),
                           "question": _short(G.nodes[n].get("question", ""), 180)}
                          for n in top_nodes],
            "boundary_nodes": [{"id": str(n), "label": _label(G, n), "external_neighbors": int(c)}
                               for n, c in boundary.most_common(10)],
            "relations": _relation_counts(G, nodes).most_common(12),
            "source_questions": Counter(_short(G.nodes[n].get("question", ""), 180)
                                        for n in nodes if G.nodes[n].get("question")).most_common(8),
        })
    return profiles, module_of, modularity


def _module_edges(G: nx.Graph, module_of: Dict[Any, int], max_edges: int = 40) -> List[Dict[str, Any]]:
    counts = Counter()
    examples: Dict[Tuple[int, int], Tuple[Any, Any, str]] = {}
    for u, v, d in G.edges(data=True):
        mu, mv = module_of.get(u), module_of.get(v)
        if mu is None or mv is None or mu == mv:
            continue
        key = tuple(sorted((mu, mv)))
        counts[key] += 1
        examples.setdefault(key, (u, v, _edge_relation(d)))
    out = []
    for (a, b), c in counts.most_common(max_edges):
        u, v, rel = examples[(a, b)]
        out.append({"module_a": a, "module_b": b, "edge_count": int(c),
                    "example": {"source": _label(G, u), "relation": rel, "target": _label(G, v)}})
    return out


def _critical_connectors(G: nx.Graph, cents: Dict[str, Dict[Any, float]], max_rows: int = 20) -> Dict[str, Any]:
    U = _get_undirected(G)
    comps = list(nx.connected_components(U)) if U.number_of_nodes() else []
    lcc = U.subgraph(max(comps, key=len)).copy() if comps else U
    base = nx.number_connected_components(U) if U.number_of_nodes() else 0
    articulation_rows = []
    if lcc.number_of_nodes() > 2:
        for n in nx.articulation_points(lcc):
            H = U.copy()
            H.remove_node(n)
            articulation_rows.append({
                "id": str(n),
                "label": _label(G, n),
                "extra_fragments": nx.number_connected_components(H) - base,
                "degree": cents["degree"].get(n, 0.0),
                "betweenness": cents["betweenness"].get(n, 0.0),
                "iter": _node_iter(G, n),
                "question": _short(G.nodes[n].get("question", ""), 180),
            })
    articulation_rows.sort(key=lambda r: (r["extra_fragments"], r["betweenness"]), reverse=True)

    bridge_rows = []
    try:
        for u, v in nx.bridges(lcc):
            bridge_rows.append({
                "source": _label(G, u),
                "target": _label(G, v),
                "source_degree": cents["degree"].get(u, 0.0),
                "target_degree": cents["degree"].get(v, 0.0),
            })
        bridge_rows.sort(key=lambda r: r["source_degree"] + r["target_degree"], reverse=True)
    except Exception:
        bridge_rows = []

    return {"articulation_points": articulation_rows[:max_rows],
            "bridge_edges": bridge_rows[:max_rows]}


def _representative_paths(G: nx.Graph, cents: Dict[str, Dict[Any, float]], module_of: Dict[Any, int],
                          max_paths: int = 20) -> List[Dict[str, Any]]:
    U = _get_undirected(G)
    if U.number_of_nodes() < 2:
        return []
    top = [n for n, _ in sorted(cents["pagerank"].items(), key=lambda x: x[1], reverse=True)[:40]]
    paths = []
    seen = set()
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            if module_of.get(a) == module_of.get(b):
                continue
            try:
                p = nx.shortest_path(U, a, b)
            except Exception:
                continue
            if len(p) < 3 or len(p) > 7:
                continue
            key = tuple(p)
            if key in seen:
                continue
            seen.add(key)
            score = (cents["pagerank"].get(a, 0.0) + cents["pagerank"].get(b, 0.0)) * len(set(module_of.get(x) for x in p))
            paths.append((score, p))
    paths.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, p in paths[:max_paths]:
        out.append({
            "score": float(score),
            "nodes": [_label(G, n) for n in p],
            "node_ids": [str(n) for n in p],
            "modules": [module_of.get(n) for n in p],
            "relations": _path_relations(G, p),
            "text": _path_text(G, p),
        })
    return out


def _edge_data_between(G: nx.Graph, u: Any, v: Any) -> Tuple[Dict[str, Any], str]:
    if G.has_edge(u, v):
        data = G.get_edge_data(u, v) or {}
        return data if isinstance(data, dict) else {}, "undirected" if not G.is_directed() else "forward"
    if G.has_edge(v, u):
        data = G.get_edge_data(v, u) or {}
        return data if isinstance(data, dict) else {}, "reverse"
    return {}, "missing"


def _path_relations(G: nx.Graph, path: Sequence[Any]) -> List[str]:
    rels = []
    for u, v in zip(path, path[1:]):
        d, _orientation = _edge_data_between(G, u, v)
        rels.append(_edge_relation(d))
    return rels


def _path_text(G: nx.Graph, path: Sequence[Any]) -> str:
    if not path:
        return ""
    bits = [_label(G, path[0])]
    for u, v in zip(path, path[1:]):
        d, orientation = _edge_data_between(G, u, v)
        rel = _edge_relation(d)
        if orientation == "reverse":
            bits.append(f"<--{rel}--")
        elif orientation == "undirected":
            bits.append(f"--{rel}--")
        else:
            bits.append(f"--{rel}-->")
        bits.append(_label(G, v))
    return " ".join(bits)


def _label_tokens(label: str) -> List[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(label))
    return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", text.lower())
            if w not in STOPWORDS and w not in GENERIC_LABELS]


def _lexical_distance(a: str, b: str) -> float:
    sa, sb = set(_label_tokens(a)), set(_label_tokens(b))
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 1.0
    return 1.0 - (len(sa & sb) / max(1, len(sa | sb)))


def _node_topic_distance(G: nx.Graph, a: Any, b: Any,
                         vecs: Optional[Dict[Any, np.ndarray]]) -> float:
    if vecs and a in vecs and b in vecs:
        va = np.asarray(vecs[a], dtype=np.float32)
        vb = np.asarray(vecs[b], dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom:
            return float(max(0.0, min(2.0, 1.0 - (float(np.dot(va, vb)) / denom))))
    return _lexical_distance(_label(G, a), _label(G, b))


def _hub_nodes_for_mining(G: nx.Graph, cents: Dict[str, Dict[Any, float]]) -> set:
    n = max(1, G.number_of_nodes())
    k = min(n, max(10, min(50, int(math.ceil(n * 0.01)))))
    return {node for node, _ in sorted(cents["degree"].items(), key=lambda x: x[1], reverse=True)[:k]}


def _candidate_nodes_for_mining(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                                module_of: Dict[Any, int], limit: int = 180,
                                include_hubs: bool = False) -> List[Any]:
    hubs = set() if include_hubs else _hub_nodes_for_mining(G, cents)
    ordered: List[Any] = []
    for key, cap in (("pagerank", limit), ("betweenness", limit), ("degree", max(40, limit // 2))):
        for n, _score in sorted(cents.get(key, {}).items(), key=lambda x: x[1], reverse=True)[:cap]:
            if n not in hubs and _mining_label_ok(G, n):
                ordered.append(n)
    by_module: Dict[int, List[Any]] = defaultdict(list)
    for n in G.nodes:
        if n not in hubs and _mining_label_ok(G, n) and module_of.get(n) is not None:
            by_module[int(module_of[n])].append(n)
    for nodes in by_module.values():
        nodes.sort(key=lambda n: (cents["pagerank"].get(n, 0.0),
                                  cents["betweenness"].get(n, 0.0),
                                  cents["degree"].get(n, 0.0)), reverse=True)
        ordered.extend(nodes[:5])
    seen = set()
    out = []
    for n in ordered:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= limit:
            break
    return out


def _path_profile_row(G: nx.Graph, path: Sequence[Any], cents: Dict[str, Dict[Any, float]],
                      module_of: Dict[Any, int], vecs: Optional[Dict[Any, np.ndarray]],
                      score: float, hub_nodes: set) -> Dict[str, Any]:
    rels = _path_relations(G, path)
    modules = [module_of.get(n) for n in path]
    internal = list(path[1:-1])
    topic_distance = _node_topic_distance(G, path[0], path[-1], vecs) if len(path) >= 2 else 0.0
    row = {
        "score": float(score),
        "hops": max(0, len(path) - 1),
        "endpoint_topic_distance": float(topic_distance),
        "module_diversity": len({m for m in modules if m is not None}),
        "relation_diversity": len(set(rels)),
        "internal_hub_count": sum(1 for n in internal if n in hub_nodes),
        "internal_mean_betweenness": float(statistics.mean([cents["betweenness"].get(n, 0.0)
                                                            for n in internal])) if internal else 0.0,
        "source": _node_packet(G, path[0], cents, module_of),
        "target": _node_packet(G, path[-1], cents, module_of),
        "bridge_nodes": [_node_packet(G, n, cents, module_of) for n in internal],
        "nodes": [_label(G, n) for n in path],
        "node_ids": [str(n) for n in path],
        "modules": modules,
        "relations": rels,
        "text": _path_text(G, path),
    }
    row["why"] = (
        f"{row['hops']}-hop path crossing {row['module_diversity']} modules with "
        f"endpoint topic distance {row['endpoint_topic_distance']:.3g} and "
        f"{row['relation_diversity']} relation types"
    )
    return row


def _mine_long_range_transitive_paths(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                                      module_of: Dict[Any, int],
                                      vecs: Optional[Dict[Any, np.ndarray]],
                                      max_paths: int = 30,
                                      min_hops: int = 4,
                                      max_hops: int = 8) -> List[Dict[str, Any]]:
    U = _get_undirected(G)
    candidates = _candidate_nodes_for_mining(G, cents, module_of, limit=180)
    candidate_set = set(candidates)
    hub_nodes = _hub_nodes_for_mining(G, cents)
    scored: List[Tuple[float, Sequence[Any]]] = []
    seen = set()
    for a in candidates:
        if not U.has_node(a):
            continue
        try:
            paths = nx.single_source_shortest_path(U, a, cutoff=max_hops)
        except Exception:
            continue
        for b in candidate_set:
            if a == b or module_of.get(a) == module_of.get(b) or b not in paths:
                continue
            key = frozenset((a, b))
            if key in seen:
                continue
            p = paths[b]
            hops = len(p) - 1
            if hops < min_hops or hops > max_hops:
                continue
            topic_distance = _node_topic_distance(G, a, b, vecs)
            if topic_distance < (0.18 if vecs else 0.35):
                continue
            rels = _path_relations(G, p)
            modules = {module_of.get(n) for n in p if module_of.get(n) is not None}
            internal = p[1:-1]
            internal_bet = sum(cents["betweenness"].get(n, 0.0) for n in internal)
            internal_hubs = sum(1 for n in internal if n in hub_nodes)
            score = (
                2.5 * topic_distance
                + 0.35 * hops
                + 0.45 * len(modules)
                + 0.20 * len(set(rels))
                + 1500.0 * (cents["pagerank"].get(a, 0.0) + cents["pagerank"].get(b, 0.0))
                + 20.0 * internal_bet
                - 0.55 * internal_hubs
            )
            seen.add(key)
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [_path_profile_row(G, p, cents, module_of, vecs, score, hub_nodes)
            for score, p in scored[:max_paths]]


def _mine_short_cross_module_bridges(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                                     module_of: Dict[Any, int],
                                     vecs: Optional[Dict[Any, np.ndarray]],
                                     max_paths: int = 30,
                                     max_hops: int = 3) -> List[Dict[str, Any]]:
    U = _get_undirected(G)
    candidates = _candidate_nodes_for_mining(G, cents, module_of, limit=220, include_hubs=False)
    candidate_set = set(candidates)
    hub_nodes = _hub_nodes_for_mining(G, cents)
    scored: List[Tuple[float, Sequence[Any]]] = []
    seen = set()
    for a in candidates:
        if not U.has_node(a):
            continue
        try:
            paths = nx.single_source_shortest_path(U, a, cutoff=max_hops)
        except Exception:
            continue
        for b in candidate_set:
            if a == b or module_of.get(a) == module_of.get(b) or b not in paths:
                continue
            key = frozenset((a, b))
            if key in seen:
                continue
            p = paths[b]
            hops = len(p) - 1
            if hops < 1 or hops > max_hops:
                continue
            topic_distance = _node_topic_distance(G, a, b, vecs)
            rels = _path_relations(G, p)
            modules = {module_of.get(n) for n in p if module_of.get(n) is not None}
            internal = p[1:-1]
            internal_bet = sum(cents["betweenness"].get(n, 0.0) for n in internal)
            internal_hubs = sum(1 for n in internal if n in hub_nodes)
            score = (
                2.0 * topic_distance
                + 0.80 * len(modules)
                + 0.25 * len(set(rels))
                + 20.0 * internal_bet
                + 1200.0 * (cents["pagerank"].get(a, 0.0) + cents["pagerank"].get(b, 0.0))
                - 0.20 * hops
                - 0.60 * internal_hubs
            )
            seen.add(key)
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [_path_profile_row(G, p, cents, module_of, vecs, score, hub_nodes)
            for score, p in scored[:max_paths]]


def _module_role_pattern(mu: Optional[int], mv: Optional[int], mw: Optional[int]) -> str:
    sm = "same" if mu == mv else "cross"
    mt = "same" if mv == mw else "cross"
    st = "same" if mu == mw else "different"
    return f"source-middle:{sm}; middle-target:{mt}; endpoints:{st}"


def _capped_edges_for_motif(G: nx.Graph, n: Any, cents: Dict[str, Dict[Any, float]],
                            incoming: bool, cap: int = 36) -> List[Tuple[Any, Any, Dict[str, Any]]]:
    if G.is_directed():
        edges = list(G.in_edges(n, data=True)) if incoming else list(G.out_edges(n, data=True))
        endpoint = (lambda e: e[0]) if incoming else (lambda e: e[1])
    else:
        edges = [(u, v, d) for u, v, d in G.edges(n, data=True)]
        endpoint = lambda e: e[1] if e[0] == n else e[0]
    edges.sort(key=lambda e: (cents["pagerank"].get(endpoint(e), 0.0),
                              cents["betweenness"].get(endpoint(e), 0.0),
                              cents["degree"].get(endpoint(e), 0.0)), reverse=True)
    return edges[:cap]


def _mine_relational_motifs(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                            module_of: Dict[Any, int], max_motifs: int = 30) -> List[Dict[str, Any]]:
    counts: Counter = Counter()
    examples: Dict[Tuple[str, str, str], List[Sequence[Any]]] = defaultdict(list)
    exact_modules: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    middle_nodes: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    for mid in G.nodes:
        incoming = _capped_edges_for_motif(G, mid, cents, incoming=True)
        outgoing = _capped_edges_for_motif(G, mid, cents, incoming=False)
        for u, _v, d1 in incoming:
            src = u if u != mid else _v
            for _x, w, d2 in outgoing:
                tgt = w if w != mid else _x
                if src == tgt:
                    continue
                mu, mv, mw = module_of.get(src), module_of.get(mid), module_of.get(tgt)
                if len({m for m in (mu, mv, mw) if m is not None}) <= 1:
                    continue
                rel1, rel2 = _edge_relation(d1), _edge_relation(d2)
                role = _module_role_pattern(mu, mv, mw)
                key = (rel1, rel2, role)
                counts[key] += 1
                exact_modules[key][f"{mu}->{mv}->{mw}"] += 1
                middle_nodes[key][_label(G, mid)] += 1
                if len(examples[key]) < 4:
                    examples[key].append((src, mid, tgt))
    rows = []
    for key, count in counts.most_common(max_motifs):
        rel1, rel2, role = key
        rows.append({
            "count": int(count),
            "relation_chain": f"{rel1} -> {rel2}",
            "role_pattern": role,
            "top_exact_module_paths": exact_modules[key].most_common(5),
            "top_middle_nodes": [{"label": lab, "count": int(c)}
                                 for lab, c in middle_nodes[key].most_common(8)],
            "examples": [{"text": _path_text(G, p),
                          "nodes": [_label(G, n) for n in p],
                          "modules": [module_of.get(n) for n in p]}
                         for p in examples[key]],
        })
    return rows


def _relation_role_signatures(G: nx.Graph, module_of: Dict[Any, int]) -> Dict[Any, Counter]:
    sig: Dict[Any, Counter] = defaultdict(Counter)
    for u, v, d in G.edges(data=True):
        rel = _edge_relation(d)
        cross = module_of.get(u) != module_of.get(v)
        edge_scope = "cross" if cross else "within"
        if G.is_directed():
            sig[u][f"out:{rel}"] += 1
            sig[v][f"in:{rel}"] += 1
            sig[u][f"out_{edge_scope}:{rel}"] += 1
            sig[v][f"in_{edge_scope}:{rel}"] += 1
        else:
            sig[u][f"edge:{rel}"] += 1
            sig[v][f"edge:{rel}"] += 1
            sig[u][f"{edge_scope}:{rel}"] += 1
            sig[v][f"{edge_scope}:{rel}"] += 1
    return sig


def _counter_cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(float(v) * float(b.get(k, 0.0)) for k, v in a.items())
    na = math.sqrt(sum(float(v) ** 2 for v in a.values()))
    nb = math.sqrt(sum(float(v) ** 2 for v in b.values()))
    if not na or not nb:
        return 0.0
    return float(dot / (na * nb))


def _node_context_edges(G: nx.Graph, n: Any, cents: Dict[str, Dict[Any, float]],
                        limit: int = 6) -> List[str]:
    edges: List[Tuple[float, str]] = []
    if G.is_directed():
        iterable = list(G.out_edges(n, data=True)) + list(G.in_edges(n, data=True))
    else:
        iterable = list(G.edges(n, data=True))
    for u, v, d in iterable:
        other = v if u == n else u
        score = (cents["pagerank"].get(other, 0.0),
                 cents["betweenness"].get(other, 0.0),
                 cents["degree"].get(other, 0.0))
        edges.append((float(score[0] + score[1] + score[2] * 1e-3), _path_text(G, [u, v])))
    edges.sort(key=lambda x: x[0], reverse=True)
    return [text for _score, text in edges[:limit]]


def _mine_structural_analogies(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                               module_of: Dict[Any, int],
                               vecs: Optional[Dict[Any, np.ndarray]],
                               max_pairs: int = 30) -> List[Dict[str, Any]]:
    sigs = _relation_role_signatures(G, module_of)
    hubs = _hub_nodes_for_mining(G, cents)
    candidates = [n for n in _candidate_nodes_for_mining(G, cents, module_of, limit=260)
                  if n not in hubs and sigs.get(n)]
    scored = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if module_of.get(a) == module_of.get(b):
                continue
            sim = _counter_cosine(sigs[a], sigs[b])
            if sim < 0.55:
                continue
            topic_distance = _node_topic_distance(G, a, b, vecs)
            if topic_distance < (0.15 if vecs else 0.30):
                continue
            shared = sorted(((k, min(sigs[a][k], sigs[b][k])) for k in (set(sigs[a]) & set(sigs[b]))),
                            key=lambda x: x[1], reverse=True)[:10]
            if not shared:
                continue
            shared_relation_names = {role.rsplit(":", 1)[-1] for role, _count in shared}
            if len(shared_relation_names) < 2:
                continue
            centrality = (
                math.log1p(cents["degree"].get(a, 0.0) + cents["degree"].get(b, 0.0))
                + 1200.0 * (cents["pagerank"].get(a, 0.0) + cents["pagerank"].get(b, 0.0))
            )
            deg_a, deg_b = cents["degree"].get(a, 0.0), cents["degree"].get(b, 0.0)
            imbalance = abs(deg_a - deg_b) / max(1.0, deg_a + deg_b)
            score = 2.2 * sim + 1.6 * topic_distance + 0.20 * centrality - 0.40 * imbalance
            scored.append((score, sim, topic_distance, shared, a, b))
    scored.sort(key=lambda x: x[0], reverse=True)
    rows = []
    for score, sim, topic_distance, shared, a, b in scored[:max_pairs]:
        rows.append({
            "score": float(score),
            "signature_similarity": float(sim),
            "topic_distance": float(topic_distance),
            "node_a": _node_packet(G, a, cents, module_of),
            "node_b": _node_packet(G, b, cents, module_of),
            "shared_relation_roles": [{"role": k, "shared_count": int(v)} for k, v in shared],
            "node_a_context": _node_context_edges(G, a, cents),
            "node_b_context": _node_context_edges(G, b, cents),
            "interpretation_hint": (
                "Local role-equivalence candidate: the two concepts occupy similar relation "
                "roles in different modules while remaining semantically/lexically separated."
            ),
        })
    return rows


def _mine_brokerage_nodes(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                          module_of: Dict[Any, int], max_nodes: int = 30) -> List[Dict[str, Any]]:
    U = _get_undirected(G)
    hubs = _hub_nodes_for_mining(G, cents)
    rows = []
    for n in G.nodes:
        if n in hubs or not U.has_node(n) or not _mining_label_ok(G, n):
            continue
        neighbor_modules = {module_of.get(nb) for nb in U.neighbors(n)
                            if module_of.get(nb) is not None and module_of.get(nb) != module_of.get(n)}
        if not neighbor_modules:
            continue
        incident_rels = []
        if G.is_directed():
            incident_iter = list(G.out_edges(n, data=True)) + list(G.in_edges(n, data=True))
        else:
            incident_iter = list(G.edges(n, data=True))
        for _u, _v, d in incident_iter:
            incident_rels.append(_edge_relation(d))
        score = (
            1.2 * len(neighbor_modules)
            + 0.25 * len(set(incident_rels))
            + 25.0 * cents["betweenness"].get(n, 0.0)
            + 1200.0 * cents["pagerank"].get(n, 0.0)
            + math.log1p(cents["degree"].get(n, 0.0))
        )
        row = _node_packet(G, n, cents, module_of)
        row.update({
            "score": float(score),
            "external_module_count": len(neighbor_modules),
            "external_modules": sorted(neighbor_modules),
            "relation_diversity": len(set(incident_rels)),
            "context": _node_context_edges(G, n, cents),
        })
        rows.append(row)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:max_nodes]


def _stable_hash_payload(payload: Any, n: int = 16) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _iso_graph(G: nx.Graph, nodes: Optional[Iterable[Any]] = None,
               root: Optional[Any] = None) -> nx.Graph:
    selected = list(nodes) if nodes is not None else list(G.nodes)
    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    selected_set = set(selected)
    for n in selected:
        H.add_node(n, iso_role="root" if root is not None and n == root else "node")
    for u, v, d in G.edges(data=True):
        if u in selected_set and v in selected_set:
            H.add_edge(u, v, relation=_edge_relation(d))
    return H


def _wl_graph_hash(H: nx.Graph, iterations: int = 3) -> str:
    try:
        return nx.weisfeiler_lehman_graph_hash(
            H, node_attr="iso_role", edge_attr="relation", iterations=iterations)
    except Exception:
        rels = Counter(_edge_relation(d) for _u, _v, d in H.edges(data=True))
        degs = sorted((int(H.degree(n)), H.nodes[n].get("iso_role", "node")) for n in H.nodes)
        return _stable_hash_payload({
            "directed": H.is_directed(),
            "nodes": H.number_of_nodes(),
            "edges": H.number_of_edges(),
            "degrees": degs,
            "relations": rels.most_common(),
        })


def _iso_match(H1: nx.Graph, H2: nx.Graph) -> bool:
    if H1.number_of_nodes() != H2.number_of_nodes() or H1.number_of_edges() != H2.number_of_edges():
        return False
    try:
        from networkx.algorithms import isomorphism as iso
        node_match = iso.categorical_node_match("iso_role", "node")
        edge_match = iso.categorical_edge_match("relation", "related_to")
        if H1.is_directed() or H2.is_directed():
            return iso.DiGraphMatcher(H1, H2, node_match=node_match,
                                      edge_match=edge_match).is_isomorphic()
        return iso.GraphMatcher(H1, H2, node_match=node_match,
                                edge_match=edge_match).is_isomorphic()
    except Exception:
        return _wl_graph_hash(H1, iterations=4) == _wl_graph_hash(H2, iterations=4)


def _relation_counter_for_nodes(G: nx.Graph, nodes: Iterable[Any]) -> Counter:
    selected = set(nodes)
    c = Counter()
    for u, v, d in G.edges(data=True):
        if u in selected and v in selected:
            c[_edge_relation(d)] += 1
    return c


def _wl_orbit_candidates(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                         module_of: Dict[Any, int],
                         max_classes: int = 30, iterations: int = 5) -> Dict[str, Any]:
    colors = {}
    for n in G.nodes:
        if G.is_directed():
            in_rels = Counter(_edge_relation(d) for _u, _v, d in G.in_edges(n, data=True))
            out_rels = Counter(_edge_relation(d) for _u, _v, d in G.out_edges(n, data=True))
            payload = {
                "in_degree": int(G.in_degree(n)),
                "out_degree": int(G.out_degree(n)),
                "in_relations": sorted(in_rels.items()),
                "out_relations": sorted(out_rels.items()),
            }
        else:
            rels = Counter(_edge_relation(d) for _u, _v, d in G.edges(n, data=True))
            payload = {"degree": int(G.degree(n)), "relations": sorted(rels.items())}
        colors[n] = _stable_hash_payload(payload)

    for _i in range(iterations):
        new_colors = {}
        for n in G.nodes:
            if G.is_directed():
                incoming = sorted((_edge_relation(d), colors[u]) for u, _v, d in G.in_edges(n, data=True))
                outgoing = sorted((_edge_relation(d), colors[v]) for _u, v, d in G.out_edges(n, data=True))
                payload = {"self": colors[n], "in": incoming, "out": outgoing}
            else:
                neighbors = []
                for u, v, d in G.edges(n, data=True):
                    other = v if u == n else u
                    neighbors.append((_edge_relation(d), colors[other]))
                payload = {"self": colors[n], "neighbors": sorted(neighbors)}
            new_colors[n] = _stable_hash_payload(payload)
        if all(new_colors[n] == colors[n] for n in G.nodes):
            colors = new_colors
            break
        colors = new_colors

    buckets: Dict[str, List[Any]] = defaultdict(list)
    for n, color in colors.items():
        if _mining_label_ok(G, n):
            buckets[color].append(n)

    rows = []
    for color, nodes in buckets.items():
        if len(nodes) < 2:
            continue
        modules = {module_of.get(n) for n in nodes if module_of.get(n) is not None}
        degree_values = sorted({int(cents["degree"].get(n, 0.0)) for n in nodes})
        score = len(nodes) * (1 + len(modules)) + sum(cents["pagerank"].get(n, 0.0) for n in nodes) * 1200.0
        nodes_sorted = sorted(nodes, key=lambda n: (cents["pagerank"].get(n, 0.0),
                                                    cents["degree"].get(n, 0.0)), reverse=True)
        rows.append({
            "score": float(score),
            "wl_color": color,
            "class_size": len(nodes),
            "modules": sorted(modules),
            "degree_values": degree_values,
            "nodes": [_node_packet(G, n, cents, module_of) for n in nodes_sorted[:12]],
            "interpretation": (
                "WL-indistinguishable node class. These are automorphism/orbit candidates, "
                "not guaranteed exact automorphisms unless confirmed by exact matching."
            ),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {
        "iterations": iterations,
        "candidate_classes": rows[:max_classes],
        "candidate_class_count": len(rows),
        "candidate_node_count": int(sum(r["class_size"] for r in rows)),
    }


def _exact_rooted_ego_isomorphisms(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                                   module_of: Dict[Any, int],
                                   max_classes: int = 30,
                                   candidate_limit: int = 500,
                                   max_ego_nodes: int = 36) -> List[Dict[str, Any]]:
    U = _get_undirected(G)
    candidates = _candidate_nodes_for_mining(G, cents, module_of, limit=candidate_limit)
    buckets: Dict[Tuple[int, str, int, int], List[Tuple[Any, nx.Graph]]] = defaultdict(list)
    for radius in (1, 2):
        for root in candidates:
            if not U.has_node(root):
                continue
            lengths = nx.single_source_shortest_path_length(U, root, cutoff=radius)
            nodes = list(lengths)
            if len(nodes) < 3 or len(nodes) > max_ego_nodes:
                continue
            H = _iso_graph(G, nodes=nodes, root=root)
            if H.number_of_edges() == 0:
                continue
            key = (radius, _wl_graph_hash(H, iterations=4), H.number_of_nodes(), H.number_of_edges())
            buckets[key].append((root, H))

    classes = []
    for (radius, _hash, n_nodes, n_edges), items in buckets.items():
        if len(items) < 2:
            continue
        exact_groups: List[Dict[str, Any]] = []
        for root, H in items[:90]:
            matched = False
            for group in exact_groups:
                if _iso_match(group["rep"], H):
                    group["roots"].append(root)
                    matched = True
                    break
            if not matched:
                exact_groups.append({"rep": H, "roots": [root]})
        for group in exact_groups:
            roots = group["roots"]
            if len(roots) < 2:
                continue
            modules = {module_of.get(n) for n in roots if module_of.get(n) is not None}
            relation_counts = Counter(_edge_relation(d) for _u, _v, d in group["rep"].edges(data=True))
            score = (
                len(roots) * (1.0 + len(modules))
                + radius
                + sum(cents["pagerank"].get(n, 0.0) for n in roots) * 1500.0
            )
            roots_sorted = sorted(roots, key=lambda n: (cents["pagerank"].get(n, 0.0),
                                                        cents["degree"].get(n, 0.0)), reverse=True)
            classes.append({
                "score": float(score),
                "radius": radius,
                "class_size": len(roots),
                "ego_nodes": n_nodes,
                "ego_edges": n_edges,
                "modules": sorted(modules),
                "relation_counts": relation_counts.most_common(12),
                "roots": [_node_packet(G, n, cents, module_of) for n in roots_sorted[:12]],
                "example_contexts": [{"root": _label(G, n), "context": _node_context_edges(G, n, cents)}
                                     for n in roots_sorted[:5]],
                "interpretation": (
                    "Exact rooted ego-net isomorphism class preserving root position, edge direction, "
                    "and relation labels while ignoring concept labels."
                ),
            })
    classes.sort(key=lambda r: r["score"], reverse=True)
    return classes[:max_classes]


def _exact_small_module_isomorphisms(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                                     module_of: Dict[Any, int],
                                     max_classes: int = 20,
                                     max_module_nodes: int = 40) -> List[Dict[str, Any]]:
    by_module: Dict[int, List[Any]] = defaultdict(list)
    for n, m in module_of.items():
        if m is not None:
            by_module[int(m)].append(n)

    buckets: Dict[Tuple[str, int, int], List[Tuple[int, nx.Graph, List[Any]]]] = defaultdict(list)
    for m, nodes in by_module.items():
        if len(nodes) < 2 or len(nodes) > max_module_nodes:
            continue
        H = _iso_graph(G, nodes=nodes)
        if H.number_of_edges() == 0:
            continue
        key = (_wl_graph_hash(H, iterations=4), H.number_of_nodes(), H.number_of_edges())
        buckets[key].append((m, H, nodes))

    rows = []
    for (_hash, n_nodes, n_edges), items in buckets.items():
        if len(items) < 2:
            continue
        exact_groups: List[Dict[str, Any]] = []
        for m, H, nodes in items:
            matched = False
            for group in exact_groups:
                if _iso_match(group["rep"], H):
                    group["modules"].append((m, nodes))
                    matched = True
                    break
            if not matched:
                exact_groups.append({"rep": H, "modules": [(m, nodes)]})
        for group in exact_groups:
            modules = group["modules"]
            if len(modules) < 2:
                continue
            score = len(modules) * n_nodes + sum(
                sum(cents["pagerank"].get(n, 0.0) for n in nodes) for _m, nodes in modules) * 1200.0
            module_rows = []
            for m, nodes in modules[:10]:
                top_nodes = sorted(nodes, key=lambda n: (cents["pagerank"].get(n, 0.0),
                                                         cents["degree"].get(n, 0.0)), reverse=True)[:8]
                module_rows.append({
                    "module": m,
                    "size": len(nodes),
                    "top_nodes": [_node_packet(G, n, cents, module_of) for n in top_nodes],
                    "top_terms": _top_terms([_label(G, n) for n in nodes], top=8),
                })
            rows.append({
                "score": float(score),
                "class_size": len(modules),
                "module_nodes": n_nodes,
                "module_edges": n_edges,
                "relation_counts": Counter(_edge_relation(d)
                                           for _u, _v, d in group["rep"].edges(data=True)).most_common(12),
                "modules": module_rows,
                "interpretation": (
                    "Exact module-induced subgraph isomorphism preserving edge direction and relation "
                    "labels while ignoring concept labels."
                ),
            })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:max_classes]


def _whole_graph_automorphism_summary(G: nx.Graph, max_nodes: int = 45,
                                      max_edges: int = 140, cap: int = 1000) -> Dict[str, Any]:
    if G.number_of_nodes() > max_nodes or G.number_of_edges() > max_edges:
        return {
            "attempted": False,
            "reason": (
                f"skipped exact whole-graph automorphism enumeration because graph has "
                f"{G.number_of_nodes()} nodes and {G.number_of_edges()} edges; limits are "
                f"{max_nodes} nodes and {max_edges} edges"
            ),
        }
    H = _iso_graph(G)
    try:
        from networkx.algorithms import isomorphism as iso
        edge_match = iso.categorical_edge_match("relation", "related_to")
        matcher = iso.DiGraphMatcher(H, H, edge_match=edge_match) if H.is_directed() else iso.GraphMatcher(H, H, edge_match=edge_match)
        count = 0
        nontrivial_examples = []
        for mapping in matcher.isomorphisms_iter():
            count += 1
            if len(nontrivial_examples) < 5:
                moved = [(str(k), str(v)) for k, v in mapping.items() if k != v][:10]
                if moved:
                    nontrivial_examples.append(moved)
            if count > cap:
                return {
                    "attempted": True,
                    "count_lower_bound": cap,
                    "capped": True,
                    "nontrivial_examples": nontrivial_examples,
                }
        return {
            "attempted": True,
            "automorphism_count": count,
            "capped": False,
            "nontrivial_examples": nontrivial_examples,
        }
    except Exception as exc:
        return {"attempted": True, "error": str(exc)}


def _mine_isomorphism_analysis(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                               module_of: Dict[Any, int]) -> Dict[str, Any]:
    return {
        "scope_notes": [
            "Unbounded subgraph-isomorphism enumeration is NP-complete and is not attempted on large ideation graphs.",
            "This audit performs bounded exact isomorphism checks for rooted ego-nets and small modules, plus WL color-refinement orbit candidates for the full graph.",
            "Exact ego/module classes preserve edge direction and relation labels but intentionally ignore concept labels, so they reveal repeated graph roles rather than duplicate text.",
            "WL orbit candidates are useful automorphism/role-equivalence leads but are not proof of exact automorphism.",
        ],
        "whole_graph_automorphism": _whole_graph_automorphism_summary(G),
        "wl_orbit_candidates": _wl_orbit_candidates(G, cents, module_of),
        "rooted_ego_isomorphism_classes": _exact_rooted_ego_isomorphisms(G, cents, module_of),
        "small_module_isomorphism_classes": _exact_small_module_isomorphisms(G, cents, module_of),
    }


def _mine_graph_insights(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                         module_of: Dict[Any, int],
                         vecs: Optional[Dict[Any, np.ndarray]]) -> Dict[str, Any]:
    return {
        "method_notes": [
            "Long-range transitive paths are shortest paths of 4-8 hops between high-signal nodes in different modules, scored for topic distance, module diversity, relation diversity, endpoint centrality, and low hub dependence.",
            "Short bridges are 1-3 hop cross-module paths that expose compact conceptual joins.",
            "Relational motifs count recurring two-step relation chains across module boundaries.",
            "Structural analogies are local role-equivalence candidates: nodes in different modules with similar incoming/outgoing relation signatures but separated labels or embeddings.",
            "If structural analogies are empty, no pair met the stricter role-equivalence threshold; recurring motifs may still expose weaker analogy patterns.",
            "The isomorphism audit combines exact rooted ego-net matching, exact small-module matching, WL orbit candidates, and bounded whole-graph automorphism checks.",
            "Brokerage nodes are non-hub concepts that touch multiple external modules with diverse incident relations.",
        ],
        "long_range_transitive_paths": _mine_long_range_transitive_paths(G, cents, module_of, vecs),
        "short_cross_module_bridges": _mine_short_cross_module_bridges(G, cents, module_of, vecs),
        "relational_motifs": _mine_relational_motifs(G, cents, module_of),
        "structural_analogies": _mine_structural_analogies(G, cents, module_of, vecs),
        "isomorphism_analysis": _mine_isomorphism_analysis(G, cents, module_of),
        "brokerage_nodes": _mine_brokerage_nodes(G, cents, module_of),
    }


def _provenance_report(G: nx.Graph, run_dir: Optional[Path], transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    q_nodes = Counter()
    for _n, d in G.nodes(data=True):
        q = _short(d.get("question", ""), 220)
        if q:
            q_nodes[q] += 1
    zero_yield = []
    high_token_zero = []
    for row in transcript:
        if row.get("_parse_error"):
            continue
        nn = row.get("new_nodes", [])
        if isinstance(nn, list):
            new_count = len(nn)
        else:
            try:
                new_count = int(nn)
            except Exception:
                new_count = 0
        rec = {"iter": row.get("iter"), "depth": row.get("depth"),
               "tokens": row.get("tokens"), "question": _short(row.get("question", ""), 220),
               "answer_excerpt": _short(row.get("answer", ""), 300)}
        if new_count == 0:
            zero_yield.append(rec)
            try:
                if float(row.get("tokens", 0)) >= 1000:
                    high_token_zero.append(rec)
            except Exception:
                pass
    by_iter = defaultdict(lambda: {"nodes": 0, "edges": 0})
    for _n, d in G.nodes(data=True):
        it = _int_attr(d, "iter")
        if it is not None:
            by_iter[it]["nodes"] += 1
    for _u, _v, d in G.edges(data=True):
        it = _edge_iter(d)
        if it is not None:
            by_iter[it]["edges"] += 1
    return {
        "run_dir": str(run_dir) if run_dir else None,
        "source_questions_by_new_nodes": q_nodes.most_common(30),
        "zero_yield_iterations": zero_yield[:50],
        "high_token_zero_yield_iterations": high_token_zero[:30],
        "n_transcript_rows": len(transcript),
        "births_by_iter": {int(k): v for k, v in sorted(by_iter.items())},
    }


def _quality_report(G: nx.Graph) -> Dict[str, Any]:
    flags = []
    normalized = defaultdict(list)
    for n in G.nodes:
        lab = _label(G, n)
        fs = _label_flags(lab)
        if fs:
            flags.append({"id": str(n), "label": lab, "flags": fs,
                          "iter": _node_iter(G, n),
                          "question": _short(G.nodes[n].get("question", ""), 180)})
        key = re.sub(r"[^a-z0-9]+", "", lab.lower())
        if key:
            normalized[key].append(lab)
    dupish = [{"key": k, "labels": sorted(set(v))[:20], "count": len(v)}
              for k, v in normalized.items() if len(set(v)) > 1]
    dupish.sort(key=lambda x: x["count"], reverse=True)
    return {"flagged_labels": flags[:200], "duplicate_normalized_labels": dupish[:50],
            "n_flagged_labels": len(flags)}


def _global_stats(G: nx.Graph, source: Dict[str, Any], components: List[Dict[str, Any]],
                  communities: List[Dict[str, Any]], modularity: float) -> Dict[str, Any]:
    U = _get_undirected(G)
    node_iters = [_node_iter(G, n) for n in G.nodes]
    edge_iters = [_edge_iter(d) for _, _, d in G.edges(data=True)]
    depths = [_node_depth(G, n) for n in G.nodes]
    comps = _weak_components(G)
    scc = list(nx.strongly_connected_components(G)) if G.is_directed() else []
    largest = max((len(c) for c in comps), default=0)
    return {
        "source": source,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "directed": G.is_directed(),
        "density": _safe_density(G),
        "avg_degree": (2 * G.number_of_edges() / G.number_of_nodes()) if G.number_of_nodes() else 0.0,
        "self_loops": nx.number_of_selfloops(G),
        "isolates": len(list(nx.isolates(U))),
        "weak_or_connected_components": len(comps),
        "strong_components": len(scc) if G.is_directed() else None,
        "largest_component_frac": largest / G.number_of_nodes() if G.number_of_nodes() else 0.0,
        "relations": _relation_counts(G).most_common(50),
        "iter": {"nodes": _iter_summary(node_iters), "edges": _iter_summary(edge_iters)},
        "depth": _iter_summary(depths),
        "communities_reported": len(communities),
        "modularity": modularity,
        "component_profiles_reported": len(components),
    }


def _try_embed(G: nx.Graph, run_dir: Optional[Path], embed_model: Optional[str]) -> Tuple[Optional[Dict[Any, np.ndarray]], Optional[str], Optional[str]]:
    if not embed_model:
        return None, None, "embedding skipped (pass --embed-model MODEL or --embed-model auto)"
    try:
        from graphstore import embed_texts, resolve_embed_model
        model = resolve_embed_model(str(run_dir) if run_dir else None, None if embed_model == "auto" else embed_model)
        nodes = list(G.nodes)
        V = embed_texts([_label(G, n) for n in nodes], model)
        return {n: V[i] for i, n in enumerate(nodes)}, model, None
    except Exception as exc:
        return None, None, str(exc)


def _semantic_audit(G: nx.Graph, vecs: Optional[Dict[Any, np.ndarray]], module_of: Dict[Any, int],
                    top: int = 20) -> Dict[str, Any]:
    if not vecs:
        return {}
    nodes = list(vecs)
    X = np.stack([vecs[n] for n in nodes]).astype(np.float32)
    idx = {n: i for i, n in enumerate(nodes)}
    centroid = X.mean(axis=0)
    center_norm = np.linalg.norm(centroid) or 1.0
    dist_centroid = {n: float(1.0 - np.dot(vecs[n], centroid / center_norm)) for n in nodes}
    outliers = sorted(nodes, key=lambda n: dist_centroid[n], reverse=True)[:top]
    U = _get_undirected(G)
    pairs = []
    # Bound cost for large graphs while still surfacing distant connected concepts.
    candidate_nodes = outliers + sorted(nodes, key=lambda n: float(G.degree(n)), reverse=True)[:200]
    candidate_nodes = list(dict.fromkeys(candidate_nodes))[:300]
    for i, a in enumerate(candidate_nodes):
        for b in candidate_nodes[i + 1:]:
            if module_of.get(a) == module_of.get(b):
                continue
            cos = float(np.dot(vecs[a], vecs[b]))
            if cos > 0.35:
                continue
            if not (U.has_node(a) and U.has_node(b)):
                continue
            try:
                p = nx.shortest_path(U, a, b)
            except Exception:
                continue
            if 2 <= len(p) <= 7:
                pairs.append((1.0 - cos, p))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return {
        "embedding_spread": float(X.var(axis=0).sum()) if len(X) else 0.0,
        "semantic_outliers": [{"id": str(n), "label": _label(G, n),
                               "distance_from_centroid": dist_centroid[n],
                               "module": module_of.get(n),
                               "iter": _node_iter(G, n)}
                              for n in outliers],
        "distant_connected_paths": [{"distance": float(d), "text": _path_text(G, p),
                                     "nodes": [_label(G, n) for n in p],
                                     "modules": [module_of.get(n) for n in p]}
                                    for d, p in pairs[:top]],
    }


def _node_packet(G: nx.Graph, n: Any, cents: Dict[str, Dict[Any, float]],
                 module_of: Dict[Any, int]) -> Dict[str, Any]:
    return {
        "id": str(n),
        "label": _label(G, n),
        "module": module_of.get(n),
        "degree": cents["degree"].get(n, 0.0),
        "pagerank": cents["pagerank"].get(n, 0.0),
        "betweenness": cents["betweenness"].get(n, 0.0),
        "iter": _node_iter(G, n),
        "depth": _node_depth(G, n),
        "question": _short(G.nodes[n].get("question", ""), 180),
    }


def _epoch_label(it: Optional[int], q25: float, q75: float) -> str:
    if it is None:
        return "unknown"
    if it <= q25:
        return "early"
    if it <= q75:
        return "middle"
    return "late"


def _hub_free_paths(G: nx.Graph, cents: Dict[str, Dict[Any, float]], module_of: Dict[Any, int],
                    hub_nodes: set, max_paths: int = 20) -> List[Dict[str, Any]]:
    U = _get_undirected(G)
    candidates = [n for n, _ in sorted(cents["pagerank"].items(), key=lambda x: x[1], reverse=True)
                  if n not in hub_nodes][:80]
    paths = []
    seen = set()
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if module_of.get(a) == module_of.get(b):
                continue
            try:
                p = nx.shortest_path(U, a, b)
            except Exception:
                continue
            if len(p) < 3 or len(p) > 8:
                continue
            if any(n in hub_nodes for n in p):
                continue
            key = tuple(p)
            if key in seen:
                continue
            seen.add(key)
            score = (
                cents["pagerank"].get(a, 0.0) + cents["pagerank"].get(b, 0.0)
                + sum(cents["betweenness"].get(n, 0.0) for n in p[1:-1])
            )
            paths.append((score, p))
    paths.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, p in paths[:max_paths]:
        out.append({
            "score": float(score),
            "text": _path_text(G, p),
            "nodes": [_label(G, n) for n in p],
            "modules": [module_of.get(n) for n in p],
        })
    return out


def _deep_graph_evidence(G: nx.Graph, cents: Dict[str, Dict[Any, float]],
                         module_of: Dict[Any, int], profile: Dict[str, Any]) -> Dict[str, Any]:
    nodes = list(G.nodes)
    hub_nodes = {n for n, _ in sorted(cents["degree"].items(), key=lambda x: x[1], reverse=True)[:10]}
    non_hubs = [n for n in nodes if n not in hub_nodes]
    iter_values = [_node_iter(G, n) for n in nodes if _node_iter(G, n) is not None]
    q25 = float(np.quantile(iter_values, 0.25)) if iter_values else 0.0
    q75 = float(np.quantile(iter_values, 0.75)) if iter_values else 0.0

    late_nodes = [n for n in non_hubs if (_node_iter(G, n) is not None and _node_iter(G, n) >= q75)]
    early_nodes = [n for n in nodes if (_node_iter(G, n) is not None and _node_iter(G, n) <= q25)]
    U = _get_undirected(G)
    boundary = []
    for n in non_hubs:
        ext = 0
        if U.has_node(n):
            for nb in U.neighbors(n):
                if module_of.get(nb) != module_of.get(n):
                    ext += 1
        if ext:
            row = _node_packet(G, n, cents, module_of)
            row["external_neighbors"] = ext
            boundary.append(row)
    boundary.sort(key=lambda r: (r["external_neighbors"], r["betweenness"], r["pagerank"]), reverse=True)

    epoch_labels: Dict[str, List[str]] = defaultdict(list)
    epoch_relations: Dict[str, Counter] = defaultdict(Counter)
    for n in nodes:
        epoch_labels[_epoch_label(_node_iter(G, n), q25, q75)].append(_label(G, n))
    for u, v, d in G.edges(data=True):
        its = [_node_iter(G, u), _node_iter(G, v), _edge_iter(d)]
        valid = [it for it in its if it is not None]
        it = max(valid) if valid else None
        epoch_relations[_epoch_label(it, q25, q75)][_edge_relation(d)] += 1

    module_nonhub = []
    for m in profile["communities"][:15]:
        mids = {str(n) for n in hub_nodes}
        top = [r for r in m["top_nodes"] if r["id"] not in mids][:8]
        module_nonhub.append({
            "module": m["id"],
            "size": m["size"],
            "top_terms": m["top_terms"][:10],
            "top_nonhub_nodes": top,
            "boundary_nodes": [b for b in m["boundary_nodes"] if b["id"] not in mids][:8],
        })

    return {
        "global_hubs_removed_for_some_views": [_node_packet(G, n, cents, module_of)
                                               for n in sorted(hub_nodes, key=lambda x: cents["degree"].get(x, 0.0), reverse=True)],
        "non_hub_top_pagerank": [_node_packet(G, n, cents, module_of)
                                 for n in sorted(non_hubs, key=lambda x: cents["pagerank"].get(x, 0.0), reverse=True)[:25]],
        "non_hub_top_betweenness": [_node_packet(G, n, cents, module_of)
                                    for n in sorted(non_hubs, key=lambda x: cents["betweenness"].get(x, 0.0), reverse=True)[:25]],
        "late_arriving_high_degree": [_node_packet(G, n, cents, module_of)
                                      for n in sorted(late_nodes, key=lambda x: cents["degree"].get(x, 0.0), reverse=True)[:25]],
        "late_arriving_high_pagerank": [_node_packet(G, n, cents, module_of)
                                        for n in sorted(late_nodes, key=lambda x: cents["pagerank"].get(x, 0.0), reverse=True)[:25]],
        "early_hubs": [_node_packet(G, n, cents, module_of)
                       for n in sorted(early_nodes, key=lambda x: cents["degree"].get(x, 0.0), reverse=True)[:20]],
        "non_hub_boundary_bridges": boundary[:30],
        "hub_free_cross_module_paths": _hub_free_paths(G, cents, module_of, hub_nodes, max_paths=20),
        "module_nonhub_summaries": module_nonhub,
        "epoch_terms": {k: _top_terms(v, top=15) for k, v in epoch_labels.items()},
        "epoch_relations": {k: c.most_common(15) for k, c in epoch_relations.items()},
    }


def _response_text(r: Any) -> str:
    parts = []
    if isinstance(r, dict):
        output_text = r.get("output_text")
        output = r.get("output") or []
        reasoning = r.get("reasoning")
        status = r.get("status")
        incomplete = r.get("incomplete_details")
    else:
        output_text = getattr(r, "output_text", None)
        output = getattr(r, "output", None) or []
        reasoning = getattr(r, "reasoning", None)
        status = getattr(r, "status", None)
        incomplete = getattr(r, "incomplete_details", None)
    if output_text:
        parts.append(str(output_text))
    if isinstance(reasoning, str) and reasoning:
        parts.append(reasoning)
    for item in output:
        contents = item.get("content", []) if isinstance(item, dict) else (getattr(item, "content", None) or [])
        for content in contents:
            if isinstance(content, dict):
                text = content.get("text") or content.get("content")
            else:
                text = getattr(content, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
    text = "\n".join(parts).strip()
    if text:
        return text
    if status or incomplete:
        return (
            f"**No visible text was returned by the Responses API.** "
            f"status={status!r}; incomplete_details={incomplete!r}. "
            "This usually means the output budget was consumed by reasoning or the response was incomplete; "
            "rerun with larger token budgets."
        )
    return ""


def _call_responses_llm(system: str, user: str, opts: LLMOptions) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=opts.base_url or None,
                    api_key=opts.api_key or os.environ.get("OPENAI_API_KEY") or "x")
    kwargs = {
        "model": opts.model,
        "instructions": system,
        "input": user,
        "max_output_tokens": opts.max_tokens,
        "reasoning": {"effort": opts.reasoning_effort},
    }
    r = client.responses.create(**kwargs)
    return _response_text(r)


def _call_chat_llm(system: str, user: str, opts: LLMOptions) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=opts.base_url or None,
                    api_key=opts.api_key or os.environ.get("OPENAI_API_KEY") or "x")
    kwargs = {
        "model": opts.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": opts.temperature,
        "max_completion_tokens": opts.max_tokens,
    }
    if opts.reasoning_effort:
        kwargs["reasoning_effort"] = opts.reasoning_effort
    for _ in range(5):
        try:
            r = client.chat.completions.create(**kwargs)
            return (r.choices[0].message.content or "").strip()
        except Exception as exc:
            msg = str(exc).lower()
            if "reasoning_effort" in kwargs and "reasoning_effort" in msg:
                kwargs.pop("reasoning_effort")
                continue
            if "max_completion_tokens" in kwargs and "max_completion_tokens" in msg:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                continue
            if "temperature" in kwargs and "temperature" in msg and "unsupported" in msg:
                kwargs.pop("temperature")
                continue
            raise
    return ""


def _call_llm(system: str, user: str, opts: LLMOptions) -> str:
    if opts.backend == "hf":
        import synthesize as S
        return S.answer_hf(system, user, model=opts.model, temperature=opts.temperature,
                           max_tokens=opts.max_tokens, device=opts.device, dtype=opts.dtype)
    if opts.backend == "chat":
        return _call_chat_llm(system, user, opts)
    if opts.backend in {"responses", "openai"}:
        return _call_responses_llm(system, user, opts)
    raise ValueError(f"unknown LLM backend: {opts.backend}")


def _call_llm_checked(system: str, user: str, opts: LLMOptions, label: str,
                      min_chars: int = 40) -> str:
    text = _call_llm(system, user, opts).strip()
    no_visible = text.startswith("**No visible text was returned")
    if len(text) >= min_chars and not no_visible:
        return text
    if opts.backend in {"responses", "openai"}:
        retry_opts = replace(opts, max_tokens=max(opts.max_tokens * 2, opts.max_tokens + 2000))
        text = _call_llm(system, user, retry_opts).strip()
        if len(text) >= min_chars and not text.startswith("**No visible text was returned"):
            return text
    if text:
        return text
    return (
        f"**LLM call `{label}` returned no visible text.** "
        "For Responses models with high reasoning effort, increase `--max-summary-tokens`, "
        "`--deep-pass-tokens`, or `--deep-dive-tokens`."
    )


def _module_prompt(module: Dict[str, Any]) -> str:
    payload = {
        "module_id": module["id"],
        "size": module["size"],
        "internal_edges": module["internal_edges"],
        "external_edges": module["external_edges"],
        "iteration_range": module["iter"],
        "top_terms": module["top_terms"],
        "top_nodes": module["top_nodes"][:12],
        "boundary_nodes": module["boundary_nodes"][:8],
        "relations": module["relations"][:10],
        "source_questions": module["source_questions"][:6],
    }
    return (
        "Interpret this graph community as an evidence-based research module in a knowledge graph.\n"
        "Use only the provided labels, relations, and provenance. Do not invent facts.\n"
        "Return Markdown with these headings: Theme, Evidence In The Graph, Mechanistic Reading, "
        "Bridge Role, Noise Or Ambiguity, and Research Questions To Inspect Next. Be specific: "
        "cite node labels, relation verbs, boundary nodes, and source-question clues.\n\n"
        + json.dumps(payload, indent=2)
    )


def _slim_module(m: Dict[str, Any], top_nodes: int = 8) -> Dict[str, Any]:
    return {
        "id": m["id"],
        "size": m["size"],
        "internal_edges": m["internal_edges"],
        "external_edges": m["external_edges"],
        "iter": m["iter"],
        "depth": m["depth"],
        "top_terms": m["top_terms"][:12],
        "top_nodes": m["top_nodes"][:top_nodes],
        "boundary_nodes": m["boundary_nodes"][:8],
        "relations": m["relations"][:10],
        "source_questions": m["source_questions"][:5],
    }


def _compact_deep_evidence(profile: Dict[str, Any]) -> Dict[str, Any]:
    deep = profile.get("deep_evidence", {})
    return {
        "global_hubs_removed_for_some_views": deep.get("global_hubs_removed_for_some_views", [])[:10],
        "non_hub_top_pagerank": deep.get("non_hub_top_pagerank", [])[:12],
        "non_hub_top_betweenness": deep.get("non_hub_top_betweenness", [])[:12],
        "late_arriving_high_degree": deep.get("late_arriving_high_degree", [])[:12],
        "late_arriving_high_pagerank": deep.get("late_arriving_high_pagerank", [])[:12],
        "non_hub_boundary_bridges": deep.get("non_hub_boundary_bridges", [])[:15],
        "hub_free_cross_module_paths": deep.get("hub_free_cross_module_paths", [])[:12],
        "epoch_terms": deep.get("epoch_terms", {}),
        "epoch_relations": deep.get("epoch_relations", {}),
    }


def _compact_mined_insights(profile: Dict[str, Any]) -> Dict[str, Any]:
    mined = profile.get("mined_insights", {})
    iso = mined.get("isomorphism_analysis", {})
    compact_iso = {
        "scope_notes": iso.get("scope_notes", []),
        "whole_graph_automorphism": iso.get("whole_graph_automorphism", {}),
        "wl_orbit_candidates": {
            **{k: v for k, v in iso.get("wl_orbit_candidates", {}).items()
               if k != "candidate_classes"},
            "candidate_classes": iso.get("wl_orbit_candidates", {}).get("candidate_classes", [])[:12],
        },
        "rooted_ego_isomorphism_classes": iso.get("rooted_ego_isomorphism_classes", [])[:12],
        "small_module_isomorphism_classes": iso.get("small_module_isomorphism_classes", [])[:8],
    }
    return {
        "method_notes": mined.get("method_notes", []),
        "long_range_transitive_paths": mined.get("long_range_transitive_paths", [])[:15],
        "short_cross_module_bridges": mined.get("short_cross_module_bridges", [])[:15],
        "relational_motifs": mined.get("relational_motifs", [])[:15],
        "structural_analogies": mined.get("structural_analogies", [])[:15],
        "isomorphism_analysis": compact_iso,
        "brokerage_nodes": mined.get("brokerage_nodes", [])[:15],
    }


def _paper_deep_dive_prompt(profile: Dict[str, Any], modules: Sequence[Dict[str, Any]]) -> str:
    gs = profile["global_stats"]
    prov = profile["provenance"]
    sem = profile.get("semantic_audit") or {}
    payload = {
        "original_topic": profile.get("topic"),
        "graph_scope": {
            "nodes": gs["nodes"],
            "edges": gs["edges"],
            "directed": gs["directed"],
            "density": gs["density"],
            "average_degree": gs["avg_degree"],
            "weak_or_connected_components": gs["weak_or_connected_components"],
            "largest_component_fraction": gs["largest_component_frac"],
            "strong_components": gs.get("strong_components"),
            "modularity": gs.get("modularity"),
            "node_iter_range": gs["iter"]["nodes"],
            "edge_iter_range": gs["iter"]["edges"],
            "node_depth_range": gs["depth"],
        },
        "top_relations": gs["relations"][:30],
        "top_nodes": {
            "degree": profile["top_nodes"]["degree"][:15],
            "pagerank": profile["top_nodes"]["pagerank"][:15],
            "betweenness": profile["top_nodes"]["betweenness"][:15],
        },
        "major_modules": [_slim_module(m) for m in modules],
        "inter_module_edges": profile["module_edges"][:30],
        "critical_connectors": {
            "articulation_points": profile["critical_connectors"]["articulation_points"][:20],
            "bridge_edges": profile["critical_connectors"]["bridge_edges"][:20],
        },
        "representative_paths": profile["representative_paths"][:25],
        "provenance": {
            "transcript_rows": prov["n_transcript_rows"],
            "zero_yield_count": len(prov["zero_yield_iterations"]),
            "high_token_zero_yield_examples": prov["high_token_zero_yield_iterations"][:10],
            "top_source_questions_by_new_nodes": prov["source_questions_by_new_nodes"][:20],
        },
        "data_quality": {
            "n_flagged_labels": profile["quality"]["n_flagged_labels"],
            "flagged_examples": profile["quality"]["flagged_labels"][:25],
            "duplicate_normalized_examples": profile["quality"]["duplicate_normalized_labels"][:20],
        },
        "semantic_audit": {
            "embedding_model": profile.get("embedding_model"),
            "embedding_spread": sem.get("embedding_spread"),
            "semantic_outliers": sem.get("semantic_outliers", [])[:15],
            "distant_connected_paths": sem.get("distant_connected_paths", [])[:15],
            "embedding_error": profile.get("embedding_error") if not sem else None,
        },
        "deep_evidence_digest": _compact_deep_evidence(profile),
        "mined_graph_insights": _compact_mined_insights(profile),
    }
    return (
        "Write a paper-level deep dive interpreting this knowledge graph as the final artifact of an "
        "iterative ideation run. The original topic is in the payload. The goal is to explain what the "
        "graph is actually telling us about that topic, not just restate graph statistics.\n\n"
        "Use only the provided graph evidence. You may synthesize and interpret, but distinguish "
        "direct graph evidence from speculative research implications. Cite concrete labels, module ids, "
        "relation verbs, bridge concepts, paths, provenance patterns, and quality caveats.\n\n"
        "Return a substantial Markdown section with these headings:\n"
        "1. Central Thesis\n"
        "2. What The Graph Discovered About The Topic\n"
        "3. Major Mechanistic Programs\n"
        "4. Cross-Module Bridges And Critical Concepts\n"
        "5. Long-Range Paths, Motifs, Isomorphisms, And Structural Analogies\n"
        "6. Novel Or Speculative Hypotheses Worth Human Review\n"
        "7. What Looks Mature Versus What Looks Like Frontier Drift\n"
        "8. Provenance And Search Dynamics\n"
        "9. Reliability, Noise, And Failure Modes\n"
        "10. Concrete Next Analyses Or Experiments\n"
        "11. Short Abstract Suitable For A Paper Or Lab Notebook\n\n"
        "Be deep and analytical. Prefer dense paragraphs and short evidence tables over generic prose. "
        "Do not claim biological, chemical, or materials-science truth unless the graph evidence supports "
        "it; phrase such claims as hypotheses generated by the graph.\n\n"
        + json.dumps(payload, indent=2)
    )


def _deep_pass_prompt(name: str, profile: Dict[str, Any], modules: Sequence[Dict[str, Any]]) -> str:
    deep = profile.get("deep_evidence", {})
    prov = profile["provenance"]
    sem = profile.get("semantic_audit") or {}
    common = {
        "original_topic": profile.get("topic"),
        "graph_scope": {
            "nodes": profile["global_stats"]["nodes"],
            "edges": profile["global_stats"]["edges"],
            "modularity": profile["global_stats"].get("modularity"),
            "largest_component_fraction": profile["global_stats"]["largest_component_frac"],
        },
    }
    if name == "mechanistic_programs":
        payload = {
            **common,
            "major_modules": [_slim_module(m, top_nodes=10) for m in modules],
            "module_nonhub_summaries": deep.get("module_nonhub_summaries", []),
            "epoch_terms": deep.get("epoch_terms", {}),
            "epoch_relations": deep.get("epoch_relations", {}),
            "top_relations": profile["global_stats"]["relations"][:30],
        }
        task = (
            "Analyze the graph's major mechanistic programs for the original topic. Explain the "
            "research programs or design logics represented by the module structure. Identify what "
            "is central, what is secondary after removing generic hubs, and how mechanisms appear "
            "to evolve from early to late search epochs."
        )
    elif name == "bridges_and_hypotheses":
        payload = {
            **common,
            "global_hubs_removed": deep.get("global_hubs_removed_for_some_views", []),
            "non_hub_top_pagerank": deep.get("non_hub_top_pagerank", []),
            "non_hub_top_betweenness": deep.get("non_hub_top_betweenness", []),
            "non_hub_boundary_bridges": deep.get("non_hub_boundary_bridges", []),
            "hub_free_cross_module_paths": deep.get("hub_free_cross_module_paths", []),
            "inter_module_edges": profile["module_edges"][:35],
            "critical_connectors": profile["critical_connectors"],
            "representative_paths": profile["representative_paths"][:25],
            "mined_graph_insights": _compact_mined_insights(profile),
        }
        task = (
            "Analyze bridge concepts and graph-generated hypotheses. Focus on non-obvious bridges "
            "after obvious hubs are removed. Extract candidate hypotheses that a researcher should "
            "inspect, and cite the exact paths, relation verbs, module ids, and node labels that "
            "support each hypothesis."
        )
    elif name == "graph_mining_insights":
        payload = {
            **common,
            "mined_graph_insights": _compact_mined_insights(profile),
            "semantic_audit": {
                "embedding_model": profile.get("embedding_model"),
                "embedding_spread": sem.get("embedding_spread"),
                "semantic_outliers": sem.get("semantic_outliers", [])[:12],
                "distant_connected_paths": sem.get("distant_connected_paths", [])[:12],
                "embedding_error": profile.get("embedding_error") if not sem else None,
            },
            "module_edges": profile["module_edges"][:25],
        }
        task = (
            "Analyze the graph-mining layer. Interpret the long-range transitive paths, short "
            "cross-module bridges, recurring relational motifs, local role-equivalence/structural "
            "analogy pairs, exact rooted ego/module isomorphism classes, WL orbit candidates, and "
            "brokerage nodes. Explain what each class of evidence reveals about the original topic, "
            "and extract high-value insight claims with exact path/motif/isomorphism/analogy evidence. "
            "Treat analogies and WL orbit candidates as hypotheses, not proof."
        )
    elif name == "search_dynamics_reliability":
        payload = {
            **common,
            "late_arriving_high_degree": deep.get("late_arriving_high_degree", []),
            "late_arriving_high_pagerank": deep.get("late_arriving_high_pagerank", []),
            "early_hubs": deep.get("early_hubs", []),
            "provenance": {
                "transcript_rows": prov["n_transcript_rows"],
                "top_source_questions_by_new_nodes": prov["source_questions_by_new_nodes"][:30],
                "zero_yield_count": len(prov["zero_yield_iterations"]),
                "high_token_zero_yield_examples": prov["high_token_zero_yield_iterations"][:15],
            },
            "quality": {
                "n_flagged_labels": profile["quality"]["n_flagged_labels"],
                "flagged_examples": profile["quality"]["flagged_labels"][:40],
                "duplicate_normalized_examples": profile["quality"]["duplicate_normalized_labels"][:25],
            },
            "semantic_audit": {
                "embedding_model": profile.get("embedding_model"),
                "embedding_spread": sem.get("embedding_spread"),
                "semantic_outliers": sem.get("semantic_outliers", [])[:20],
                "distant_connected_paths": sem.get("distant_connected_paths", [])[:20],
                "embedding_error": profile.get("embedding_error") if not sem else None,
            },
        }
        task = (
            "Analyze search dynamics, reliability, and noise. Explain what the provenance says about "
            "how the graph grew, which late concepts look important, where the run may have drifted, "
            "and which data-quality problems should temper interpretation."
        )
    else:
        raise ValueError(f"unknown deep pass: {name}")

    return (
        f"{task}\n\n"
        "Use only the payload. Write a dense Markdown memo with evidence bullets and short analytic "
        "paragraphs. Mark speculative implications as hypotheses, not facts.\n\n"
        + json.dumps(payload, indent=2)
    )


def _deep_pass_names(n: int) -> List[str]:
    names = ["mechanistic_programs", "graph_mining_insights",
             "bridges_and_hypotheses", "search_dynamics_reliability"]
    return names[:max(0, min(n, len(names)))]


DEEP_DIVE_HEADINGS = [
    "Central Thesis",
    "What The Graph Discovered About The Topic",
    "Major Mechanistic Programs",
    "Cross-Module Bridges And Critical Concepts",
    "Long-Range Paths, Motifs, Isomorphisms, And Structural Analogies",
    "Novel Or Speculative Hypotheses Worth Human Review",
    "What Looks Mature Versus What Looks Like Frontier Drift",
    "Provenance And Search Dynamics",
    "Reliability, Noise, And Failure Modes",
    "Concrete Next Analyses Or Experiments",
    "Short Abstract Suitable For A Paper Or Lab Notebook",
]


def _missing_deep_dive_headings(text: str) -> List[str]:
    low = text.lower()
    return [h for h in DEEP_DIVE_HEADINGS if h.lower() not in low]


def _deep_dive_completion_prompt(profile: Dict[str, Any], modules: Sequence[Dict[str, Any]],
                                 draft: str, missing: Sequence[str],
                                 pass_memos: Dict[str, str]) -> str:
    payload = {
        "original_topic": profile.get("topic"),
        "missing_headings": list(missing),
        "graph_scope": {
            "nodes": profile["global_stats"]["nodes"],
            "edges": profile["global_stats"]["edges"],
            "modularity": profile["global_stats"].get("modularity"),
            "largest_component_fraction": profile["global_stats"]["largest_component_frac"],
        },
        "major_modules": [_slim_module(m, top_nodes=8) for m in modules],
        "compact_deep_evidence": _compact_deep_evidence(profile),
        "compact_mined_insights": _compact_mined_insights(profile),
        "supporting_pass_memos": pass_memos,
        "previous_draft_tail": draft[-5000:],
    }
    return (
        "The previous paper-level graph interpretation is incomplete or truncated. "
        "Write ONLY the missing sections listed in the payload, using the exact heading names. "
        "Do not repeat sections already present. Use graph evidence and supporting memos; mark "
        "speculative implications as hypotheses.\n\n"
        + json.dumps(payload, indent=2)
    )


def _llm_summaries(profile: Dict[str, Any], opts: LLMOptions, max_modules: int,
                   progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    if not opts.enabled:
        return {}
    system = ("You are a careful graph analyst. You summarize graph evidence without inventing "
              "domain facts. You are allowed to synthesize research implications only when you mark "
              "them as graph-generated hypotheses. Be specific and cite labels/relations from the payload.")
    modules = profile["communities"][:max_modules]
    out = {"model": opts.model, "backend": opts.backend, "reasoning_effort": opts.reasoning_effort,
           "modules": {}}
    for i, m in enumerate(modules, 1):
        if progress:
            progress(f"LLM module summary {i}/{len(modules)}: module {m['id']} ({m['size']} nodes)")
        out["modules"][str(m["id"])] = _call_llm_checked(
            system, _module_prompt(m), opts, f"module_{m['id']}", min_chars=80)
    if progress:
        progress("LLM executive summary")
    overview_payload = {
        "global_stats": profile["global_stats"],
        "top_degree": profile["top_nodes"]["degree"][:10],
        "top_pagerank": profile["top_nodes"]["pagerank"][:10],
        "top_betweenness": profile["top_nodes"]["betweenness"][:10],
        "major_modules": [{k: m[k] for k in ("id", "size", "top_terms", "top_nodes", "relations")}
                          for m in modules],
        "critical_connectors": profile["critical_connectors"],
        "mined_graph_insights": _compact_mined_insights(profile),
        "quality": {"n_flagged_labels": profile["quality"]["n_flagged_labels"],
                    "examples": profile["quality"]["flagged_labels"][:15]},
    }
    user = (
        "Write an executive summary of this graph audit. Cover what the graph appears to contain, "
        "its major modules, the most important connector concepts, likely data-quality problems, "
        "and what a human should inspect first. Use Markdown.\n\n"
        + json.dumps(overview_payload, indent=2)
    )
    out["overview"] = _call_llm_checked(system, user, opts, "overview", min_chars=120)

    pass_names = _deep_pass_names(opts.deep_passes)
    out["deep_dive_passes"] = {}
    pass_opts = replace(opts, max_tokens=max(opts.max_tokens, opts.deep_pass_tokens))
    for i, name in enumerate(pass_names, 1):
        if progress:
            progress(f"LLM deep evidence pass {i}/{len(pass_names)}: {name}")
        out["deep_dive_passes"][name] = _call_llm_checked(
            system, _deep_pass_prompt(name, profile, modules), pass_opts, name, min_chars=200)

    if progress:
        progress("LLM final paper-level synthesis")
    deep_opts = replace(opts, max_tokens=max(opts.max_tokens, opts.deep_tokens))
    final_prompt = _paper_deep_dive_prompt(profile, modules)
    if out["deep_dive_passes"]:
        final_prompt += (
            "\n\nPrior LLM evidence-pass memos are below. Use them as secondary notes, but keep the "
            "final synthesis grounded in the graph payload above.\n\n"
            + json.dumps(out["deep_dive_passes"], indent=2)
        )
    out["deep_dive"] = _call_llm_checked(system, final_prompt, deep_opts, "deep_dive", min_chars=500)
    missing = _missing_deep_dive_headings(out["deep_dive"])
    out["deep_dive_missing_headings"] = missing
    if missing:
        if progress:
            progress("LLM completion pass for missing deep-dive sections")
        completion = _call_llm_checked(
            system,
            _deep_dive_completion_prompt(profile, modules, out["deep_dive"], missing,
                                         out["deep_dive_passes"]),
            deep_opts,
            "deep_dive_completion",
            min_chars=300,
        )
        out["deep_dive_completion"] = completion
        out["deep_dive"] += "\n\n## Completion Of Missing Deep-Dive Sections\n\n" + completion
        out["deep_dive_missing_headings_after_completion"] = _missing_deep_dive_headings(out["deep_dive"])
    out["deep_dive_tokens_requested"] = deep_opts.max_tokens
    out["deep_pass_tokens_requested"] = pass_opts.max_tokens
    return out


def _split_markdown_major_sections(markdown: str) -> List[Dict[str, str]]:
    sections = []
    title = "Preamble"
    buf: List[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if buf:
                sections.append({"title": title, "text": "\n".join(buf).strip()})
            title = line.lstrip("#").strip()
            buf = [line]
        else:
            buf.append(line)
    if buf:
        sections.append({"title": title, "text": "\n".join(buf).strip()})
    return [s for s in sections if s["text"]]


def _clip_for_report_review(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    head = max(1000, int(max_chars * 0.68))
    tail = max(1000, max_chars - head - 300)
    return (
        text[:head].rstrip()
        + "\n\n[...section clipped for final LLM review...]\n\n"
        + text[-tail:].lstrip()
    )


def _report_review_packets(report_text: str, opts: LLMOptions) -> List[Dict[str, Any]]:
    sections = _split_markdown_major_sections(report_text)
    limit = opts.report_review_max_chunks if opts.report_review_max_chunks > 0 else len(sections)
    priority = [
        "Executive Summary",
        "Mined Graph Insights - Executive Discoveries",
        "Deep Mining Evidence - Executive Audit",
        "Paper-Level Graph Interpretation",
        "Mined Graph Insights",
        "Critical Connectors And Paths",
        "Semantic Audit",
        "Provenance",
        "Data Quality Flags",
        "Module Atlas",
        "Inter-Module Edges",
        "LLM Supporting Evidence Passes",
        "Global Statistics",
    ]
    by_title = {s["title"].lower(): s for s in sections}
    selected: List[Dict[str, str]] = []
    seen = set()
    for title in priority:
        if len(selected) >= limit:
            break
        section = by_title.get(title.lower())
        if section and section["title"] not in seen:
            selected.append(section)
            seen.add(section["title"])
    for section in sections:
        if len(selected) >= limit:
            break
        if section["title"] not in seen:
            selected.append(section)
            seen.add(section["title"])
    packets = []
    for section in selected[:limit]:
        clipped = _clip_for_report_review(section["text"], opts.report_review_chunk_chars)
        packets.append({
            "title": section["title"],
            "chars_original": len(section["text"]),
            "chars_sent": len(clipped),
            "text": clipped,
        })
    return packets


def _report_section_review_prompt(profile: Dict[str, Any], packet: Dict[str, Any],
                                  index: int, total: int) -> str:
    payload = {
        "original_topic": profile.get("topic"),
        "section_index": index,
        "section_count": total,
        "section_title": packet["title"],
        "chars_original": packet["chars_original"],
        "graph_scope": {
            "nodes": profile["global_stats"]["nodes"],
            "edges": profile["global_stats"]["edges"],
            "modularity": profile["global_stats"].get("modularity"),
            "largest_component_fraction": profile["global_stats"]["largest_component_frac"],
        },
        "section_text": packet["text"],
    }
    return (
        "You are doing a final report-review pass for a deep graph profile. Read this report section "
        "as evidence, not as ground truth. Extract the most important insight-bearing claims, "
        "cross-section implications, contradictions, noise risks, and follow-up questions. "
        "Prefer concrete node labels, relation verbs, paths, motifs, modules, and provenance clues. "
        "Do not repeat boilerplate. Mark speculative implications as hypotheses.\n\n"
        "Return a dense Markdown memo with these headings: Key Evidence, Deep Insights, Risks, "
        "Follow-Up Questions.\n\n"
        + json.dumps(payload, indent=2)
    )


def _final_report_synthesis_prompt(profile: Dict[str, Any], section_reviews: Sequence[Dict[str, Any]],
                                   opts: LLMOptions) -> str:
    review_payloads = []
    for review in section_reviews:
        review_payloads.append({
            "title": review.get("title"),
            "chars_original": review.get("chars_original"),
            "chars_sent": review.get("chars_sent"),
            "memo": _clip_for_report_review(str(review.get("memo", "")), opts.report_review_memo_chars),
        })
    payload = {
        "original_topic": profile.get("topic"),
        "graph_scope": {
            "nodes": profile["global_stats"]["nodes"],
            "edges": profile["global_stats"]["edges"],
            "density": profile["global_stats"]["density"],
            "modularity": profile["global_stats"].get("modularity"),
            "largest_component_fraction": profile["global_stats"]["largest_component_frac"],
            "relations": profile["global_stats"]["relations"][:20],
        },
        "near_top_mined_discoveries": _compact_mined_insights(profile),
        "near_top_deep_evidence": _compact_deep_evidence(profile),
        "top_modules": [_slim_module(m, top_nodes=6) for m in profile["communities"][:10]],
        "data_quality": {
            "n_flagged_labels": profile["quality"]["n_flagged_labels"],
            "flagged_examples": profile["quality"]["flagged_labels"][:20],
            "duplicate_normalized_examples": profile["quality"]["duplicate_normalized_labels"][:15],
        },
        "section_review_memos": review_payloads,
    }
    return (
        "You have now reviewed the completed graph profile report section by section. Write the final, "
        "highest-level insight synthesis for the report. This should be more integrative than the earlier "
        "paper-level deep dive: connect graph statistics, mined paths, motifs, analogies, isomorphism audit, "
        "module structure, provenance, semantic audit, and quality risks into a coherent interpretation of "
        "what the graph is telling us about the original topic.\n\n"
        "Use only the report evidence and payload. Distinguish direct graph evidence from hypotheses. "
        "Do not invent external domain facts. Be concrete and cite labels, relation verbs, modules, motifs, "
        "paths, and caveats.\n\n"
        "Return Markdown with these headings:\n"
        "1. Final Integrated Thesis\n"
        "2. Strongest Evidence-Backed Discoveries\n"
        "3. Non-Obvious Cross-Section Connections\n"
        "4. Long-Range, Analogical, Motif, And Isomorphism Signals\n"
        "5. Highest-Value Hypotheses To Investigate\n"
        "6. What To Distrust Or Clean Before Acting\n"
        "7. Next Graph Queries And Human Review Tasks\n"
        "8. One-Page Narrative Summary\n\n"
        + json.dumps(payload, indent=2)
    )


def _llm_final_report_review(profile: Dict[str, Any], report_path: Path, opts: LLMOptions,
                             progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    if not opts.enabled or not opts.report_review:
        return {}
    report_text = report_path.read_text(encoding="utf-8")
    packets = _report_review_packets(report_text, opts)
    system = (
        "You are a senior graph-mining analyst doing a final audit of an already-written graph profile. "
        "Your job is to extract deep, evidence-grounded insights from the report, not to restate sections."
    )
    section_opts = replace(opts, max_tokens=max(opts.max_tokens, opts.deep_pass_tokens))
    reviews = []
    for i, packet in enumerate(packets, 1):
        if progress:
            progress(f"LLM final report review section {i}/{len(packets)}: {packet['title']}")
        memo = _call_llm_checked(
            system,
            _report_section_review_prompt(profile, packet, i, len(packets)),
            section_opts,
            f"report_review_{i}_{_slug(packet['title'])}",
            min_chars=180,
        )
        reviews.append({
            "title": packet["title"],
            "chars_original": packet["chars_original"],
            "chars_sent": packet["chars_sent"],
            "memo": memo,
        })

    if progress:
        progress("LLM final integrated report synthesis")
    final_opts = replace(opts, max_tokens=max(opts.max_tokens, opts.report_review_tokens))
    synthesis = _call_llm_checked(
        system,
        _final_report_synthesis_prompt(profile, reviews, opts),
        final_opts,
        "final_report_review_synthesis",
        min_chars=700,
    )
    return {
        "enabled": True,
        "model": opts.model,
        "backend": opts.backend,
        "reasoning_effort": opts.reasoning_effort,
        "section_count": len(reviews),
        "section_reviews": reviews,
        "synthesis": synthesis,
        "tokens_requested": final_opts.max_tokens,
        "chunk_chars": opts.report_review_chunk_chars,
        "memo_chars": opts.report_review_memo_chars,
        "max_chunks": opts.report_review_max_chunks,
    }


def _save_figures(profile: Dict[str, Any], G: nx.Graph, vecs: Optional[Dict[Any, np.ndarray]],
                  module_of: Dict[Any, int], out_dir: Path) -> List[str]:
    mpl_cache = out_dir / ".mplconfig"
    xdg_cache = out_dir / ".cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name: str):
        path = fig_dir / name
        fig.savefig(path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        written.append(str(path.relative_to(out_dir)))

    # Degree distribution.
    degs = [d for _, d in G.degree()]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if degs:
        bins = min(60, max(10, int(math.sqrt(len(degs)))))
        ax.hist(degs, bins=bins, color="#1f77b4", alpha=0.85)
    ax.set_title("Degree Distribution")
    ax.set_xlabel("degree")
    ax.set_ylabel("nodes")
    ax.grid(True, color="0.92")
    save(fig, "degree_distribution.png")

    # Relation types.
    rels = profile["global_stats"]["relations"][:20]
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.28 * len(rels))))
    if rels:
        labels = [_short(r, 40) for r, _ in rels][::-1]
        vals = [c for _, c in rels][::-1]
        ax.barh(range(len(vals)), vals, color="#2ca02c", alpha=0.85)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Relation Types")
    ax.set_xlabel("edge count")
    ax.grid(True, axis="x", color="0.92")
    save(fig, "relation_types.png")

    # Component sizes.
    comps = profile["components"][:40]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(range(len(comps)), [c["size"] for c in comps], color="#9467bd", alpha=0.85)
    ax.set_title("Weak/Connected Component Sizes")
    ax.set_xlabel("component rank")
    ax.set_ylabel("nodes")
    ax.grid(True, axis="y", color="0.92")
    save(fig, "component_sizes.png")

    # Community sizes.
    comms = profile["communities"][:40]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(range(len(comms)), [c["size"] for c in comms], color="#ff7f0e", alpha=0.85)
    ax.set_title(f"Community Sizes (modularity Q={profile['global_stats'].get('modularity'):.3f})")
    ax.set_xlabel("module rank")
    ax.set_ylabel("nodes")
    ax.grid(True, axis="y", color="0.92")
    save(fig, "community_sizes.png")

    # Growth / birth profile if iter provenance exists.
    births = profile["provenance"]["births_by_iter"]
    if births:
        its = sorted(int(k) for k in births)
        new_nodes = [births[i]["nodes"] if i in births else births[str(i)]["nodes"] for i in its]
        new_edges = [births[i]["edges"] if i in births else births[str(i)]["edges"] for i in its]
        cum_nodes = np.cumsum(new_nodes)
        cum_edges = np.cumsum(new_edges)
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(its, cum_nodes, label="cumulative nodes", color="#1f77b4", lw=2)
        ax.plot(its, cum_edges, label="cumulative edges", color="#d62728", lw=2)
        ax2 = ax.twinx()
        ax2.bar(its, new_nodes, color="0.7", alpha=0.35, label="new nodes")
        ax.set_title("Graph Growth From Provenance")
        ax.set_xlabel("iteration")
        ax.set_ylabel("cumulative count")
        ax2.set_ylabel("new nodes")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(True, color="0.92")
        save(fig, "growth_from_provenance.png")

    # Centrality scatter.
    deg = profile["_centrality_values"]["degree"]
    pr = profile["_centrality_values"]["pagerank"]
    bet = profile["_centrality_values"]["betweenness"]
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    xs = [deg.get(n, 0.0) for n in G.nodes]
    ys = [pr.get(n, 0.0) for n in G.nodes]
    cs = [bet.get(n, 0.0) for n in G.nodes]
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=18, alpha=0.7, linewidths=0)
    fig.colorbar(sc, ax=ax, label="betweenness")
    ax.set_title("Degree vs PageRank")
    ax.set_xlabel("degree")
    ax.set_ylabel("PageRank")
    ax.grid(True, color="0.92")
    save(fig, "degree_pagerank_betweenness.png")

    # Semantic map if embeddings are available.
    if vecs:
        nodes = list(G.nodes)
        X = np.stack([vecs[n] for n in nodes]).astype(np.float32)
        Xc = X - X.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            P = Xc @ vt[:2].T
        except Exception:
            P = Xc[:, :2]
        colors = [module_of.get(n, -1) for n in nodes]
        sizes = [18 + 2500 * pr.get(n, 0.0) for n in nodes]
        fig, ax = plt.subplots(figsize=(8.0, 7.2))
        sc = ax.scatter(P[:, 0], P[:, 1], c=colors, cmap="tab20", s=sizes, alpha=0.72, linewidths=0)
        ax.set_title("Semantic Map (PCA; color=module, size=PageRank)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, color="0.93")
        save(fig, "semantic_module_map.png")

    return written


def _md_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    def cell(x: Any) -> str:
        if isinstance(x, float):
            return f"{x:.4g}"
        return str(x).replace("\n", " ").replace("|", "\\|")
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(cell(x) for x in row) + " |")
    return "\n".join(out)


def _mined_discovery_summary(profile: Dict[str, Any]) -> List[str]:
    mined = profile.get("mined_insights") or {}
    if not mined:
        return []

    long_paths = mined.get("long_range_transitive_paths", [])
    short_bridges = mined.get("short_cross_module_bridges", [])
    analogies = mined.get("structural_analogies", [])
    motifs = mined.get("relational_motifs", [])
    brokers = mined.get("brokerage_nodes", [])
    iso = mined.get("isomorphism_analysis") or {}
    wl = iso.get("wl_orbit_candidates") or {}
    ego_iso = iso.get("rooted_ego_isomorphism_classes") or []
    module_iso = iso.get("small_module_isomorphism_classes") or []
    auto = iso.get("whole_graph_automorphism") or {}

    if auto.get("attempted"):
        auto_status = f"attempted; count={auto.get('automorphism_count', auto.get('count_lower_bound', 'unknown'))}"
    else:
        auto_status = auto.get("reason", "not attempted")

    def analogy_text(row: Dict[str, Any]) -> str:
        a = row.get("node_a", {}).get("label", "?")
        b = row.get("node_b", {}).get("label", "?")
        roles = "; ".join(x.get("role", "") for x in row.get("shared_relation_roles", [])[:4])
        return f"{a} <-> {b}" + (f" ({roles})" if roles else "")

    lines = [
        "## Mined Graph Insights - Executive Discoveries",
        "",
        "This near-top section summarizes the deterministic graph-mining evidence. "
        "The full evidence tables are repeated later under **Mined Graph Insights**.",
        "",
        _md_table([
            ("long-range transitive paths", len(long_paths), long_paths[0]["text"] if long_paths else "(none)"),
            ("short cross-module bridges", len(short_bridges), short_bridges[0]["text"] if short_bridges else "(none)"),
            ("structural analogy candidates", len(analogies), analogy_text(analogies[0]) if analogies else "(none met threshold)"),
            ("recurring relation motifs", len(motifs), motifs[0]["relation_chain"] if motifs else "(none)"),
            ("non-hub brokerage nodes", len(brokers), brokers[0]["label"] if brokers else "(none)"),
            ("WL orbit candidate classes", wl.get("candidate_class_count", 0), f"{wl.get('candidate_node_count', 0)} nodes covered"),
            ("exact rooted ego-net isomorphism classes", len(ego_iso), ego_iso[0]["roots"][0]["label"] if ego_iso else "(none met threshold)"),
            ("exact small-module isomorphism classes", len(module_iso), f"{module_iso[0]['class_size']} modules" if module_iso else "(none met threshold)"),
            ("whole-graph automorphism", "status", auto_status),
        ], ["mined evidence", "count/status", "top example"]),
        "",
    ]

    if long_paths:
        lines += ["### Top Long-Range Transitive Paths", ""]
        lines.append(_md_table([
            (r["score"], r["hops"], r["endpoint_topic_distance"], r["text"])
            for r in long_paths[:5]
        ], ["score", "hops", "topic distance", "path"]))
        lines.append("")

    if analogies:
        lines += ["### Top Structural Analogies", ""]
        lines.append(_md_table([
            (r["score"], r["signature_similarity"], r["topic_distance"],
             r["node_a"]["label"], r["node_b"]["label"],
             "; ".join(x["role"] for x in r.get("shared_relation_roles", [])[:5]))
            for r in analogies[:5]
        ], ["score", "role sim", "topic distance", "node A", "node B", "shared roles"]))
        lines.append("")

    if motifs:
        lines += ["### Top Recurring Relation Motifs", ""]
        lines.append(_md_table([
            (r["count"], r["relation_chain"], r["role_pattern"],
             r.get("examples", [{}])[0].get("text", ""))
            for r in motifs[:5]
        ], ["count", "relation chain", "module role pattern", "example"]))
        lines.append("")

    if wl.get("candidate_classes") or ego_iso or module_iso:
        lines += ["### Isomorphism Snapshot", ""]
        if wl.get("candidate_classes"):
            lines.append(_md_table([
                (r["score"], r["class_size"], ", ".join(map(str, r["modules"])),
                 "; ".join(n["label"] for n in r.get("nodes", [])[:6]))
                for r in wl["candidate_classes"][:5]
            ], ["WL score", "class size", "modules", "example nodes"]))
            lines.append("")
        if ego_iso:
            lines.append(_md_table([
                (r["score"], r["radius"], r["class_size"],
                 "; ".join(n["label"] for n in r.get("roots", [])[:6]))
                for r in ego_iso[:5]
            ], ["ego score", "radius", "class size", "root examples"]))
            lines.append("")
        if module_iso:
            lines.append(_md_table([
                (r["score"], r["class_size"], r["module_nodes"], r["module_edges"])
                for r in module_iso[:5]
            ], ["module score", "class size", "nodes", "edges"]))
            lines.append("")

    return lines


def _deep_mining_summary(profile: Dict[str, Any]) -> List[str]:
    deep = profile.get("deep_evidence") or {}
    sem = profile.get("semantic_audit") or {}
    critical = profile.get("critical_connectors") or {}
    prov = profile.get("provenance") or {}
    quality = profile.get("quality") or {}
    if not any((deep, sem, critical, prov, quality)):
        return []

    nonhub_pr = deep.get("non_hub_top_pagerank", [])
    nonhub_bet = deep.get("non_hub_top_betweenness", [])
    boundary = deep.get("non_hub_boundary_bridges", [])
    hub_free_paths = deep.get("hub_free_cross_module_paths", [])
    late_pr = deep.get("late_arriving_high_pagerank", [])
    late_degree = deep.get("late_arriving_high_degree", [])
    art = critical.get("articulation_points", [])
    bridge_edges = critical.get("bridge_edges", [])
    semantic_outliers = sem.get("semantic_outliers", [])
    distant_paths = sem.get("distant_connected_paths", [])
    source_questions = prov.get("source_questions_by_new_nodes", [])
    high_token_zero = prov.get("high_token_zero_yield_iterations", [])

    def labels(rows: Sequence[Dict[str, Any]], n: int = 5) -> str:
        return "; ".join(str(r.get("label", r.get("source", ""))) for r in rows[:n]) or "(none)"

    lines = [
        "## Deep Mining Evidence - Executive Audit",
        "",
        "This near-top section summarizes the broader deterministic audit beyond the path/motif/"
        "isomorphism miners. It surfaces the evidence the LLM uses for the paper-level interpretation.",
        "",
        _md_table([
            ("non-hub PageRank leaders", len(nonhub_pr), labels(nonhub_pr)),
            ("non-hub betweenness leaders", len(nonhub_bet), labels(nonhub_bet)),
            ("non-hub boundary bridges", len(boundary), labels(boundary)),
            ("hub-free cross-module paths", len(hub_free_paths), hub_free_paths[0]["text"] if hub_free_paths else "(none)"),
            ("late-arriving high-PageRank concepts", len(late_pr), labels(late_pr)),
            ("late-arriving high-degree concepts", len(late_degree), labels(late_degree)),
            ("articulation points", len(art), labels(art)),
            ("bridge edges", len(bridge_edges), f"{bridge_edges[0]['source']} - {bridge_edges[0]['target']}" if bridge_edges else "(none)"),
            ("semantic outliers", len(semantic_outliers), labels(semantic_outliers)),
            ("distant semantic paths", len(distant_paths), distant_paths[0]["text"] if distant_paths else "(none)"),
            ("top source questions", len(source_questions), source_questions[0][0] if source_questions else "(none)"),
            ("high-token zero-yield iterations", len(high_token_zero), high_token_zero[0].get("question", "(none)") if high_token_zero else "(none)"),
            ("flagged labels", quality.get("n_flagged_labels", 0), labels(quality.get("flagged_labels", []))),
        ], ["deep evidence", "count", "top example"]),
        "",
    ]

    central_rows = []
    for row in nonhub_pr[:5]:
        central_rows.append(("PageRank", row["label"], row.get("module"), row.get("pagerank"), row.get("degree"), row.get("iter")))
    for row in nonhub_bet[:5]:
        central_rows.append(("betweenness", row["label"], row.get("module"), row.get("betweenness"), row.get("degree"), row.get("iter")))
    if central_rows:
        lines += ["### Non-Hub Central Concepts", ""]
        lines.append(_md_table(central_rows, ["view", "label", "module", "score", "degree", "iter"]))
        lines.append("")

    if boundary:
        lines += ["### Boundary Bridges After Removing Global Hubs", ""]
        lines.append(_md_table([
            (r["label"], r.get("module"), r.get("external_neighbors"), r.get("betweenness"), r.get("pagerank"), r.get("iter"))
            for r in boundary[:8]
        ], ["label", "module", "external neighbors", "betweenness", "pagerank", "iter"]))
        lines.append("")

    if hub_free_paths:
        lines += ["### Hub-Free Cross-Module Paths", ""]
        for i, path in enumerate(hub_free_paths[:6], 1):
            lines.append(f"{i}. score={path.get('score', 0):.4g}: {path.get('text', '')}")
        lines.append("")

    late_rows = []
    for row in late_pr[:5]:
        late_rows.append(("late PageRank", row["label"], row.get("module"), row.get("pagerank"), row.get("degree"), row.get("iter")))
    for row in late_degree[:5]:
        late_rows.append(("late degree", row["label"], row.get("module"), row.get("pagerank"), row.get("degree"), row.get("iter")))
    if late_rows:
        lines += ["### Late-Emerging Important Concepts", ""]
        lines.append(_md_table(late_rows, ["view", "label", "module", "pagerank", "degree", "iter"]))
        lines.append("")

    if art or bridge_edges:
        lines += ["### Critical Connector Snapshot", ""]
        if art:
            lines.append(_md_table([
                (r["label"], r.get("extra_fragments"), r.get("degree"), r.get("betweenness"), r.get("iter"))
                for r in art[:8]
            ], ["articulation point", "extra fragments", "degree", "betweenness", "iter"]))
            lines.append("")
        if bridge_edges:
            lines.append(_md_table([
                (r["source"], r["target"], r.get("source_degree"), r.get("target_degree"))
                for r in bridge_edges[:8]
            ], ["bridge source", "bridge target", "source degree", "target degree"]))
            lines.append("")

    if semantic_outliers or distant_paths:
        lines += ["### Semantic Mining Snapshot", ""]
        if semantic_outliers:
            lines.append(_md_table([
                (r["label"], r.get("distance_from_centroid"), r.get("module"), r.get("iter"))
                for r in semantic_outliers[:8]
            ], ["semantic outlier", "distance", "module", "iter"]))
            lines.append("")
        if distant_paths:
            for i, path in enumerate(distant_paths[:5], 1):
                lines.append(f"{i}. distance={path.get('distance', 0):.4g}: {path.get('text', '')}")
            lines.append("")

    if source_questions or high_token_zero:
        lines += ["### Provenance And Search-Dynamics Snapshot", ""]
        if source_questions:
            lines.append(_md_table(source_questions[:8], ["source question", "node count"]))
            lines.append("")
        if high_token_zero:
            lines.append(_md_table([
                (r.get("iter"), r.get("tokens"), r.get("question"), r.get("answer_excerpt"))
                for r in high_token_zero[:5]
            ], ["iter", "tokens", "question", "answer excerpt"]))
            lines.append("")

    return lines


def _write_markdown(profile: Dict[str, Any], out_dir: Path) -> Path:
    path = out_dir / "report.md"
    gs = profile["global_stats"]
    lines = [
        f"# Graph Profile Report - {_short(Path(gs['source'].get('graph_path', 'graph')).name, 80)}",
        "",
        f"Generated: {profile['generated_at']}",
        "",
        "## Executive Summary",
        "",
    ]
    llm = profile.get("llm_summaries") or {}
    if llm.get("overview"):
        lines += [llm["overview"], ""]
    else:
        lines += [
            f"- The graph has {gs['nodes']} nodes and {gs['edges']} edges across "
            f"{gs['weak_or_connected_components']} weak/connected components.",
            f"- The largest component contains {gs['largest_component_frac']:.1%} of nodes.",
            f"- Community detection found {len(profile['communities'])} reported modules "
            f"(modularity Q={gs.get('modularity'):.3f}).",
            f"- {profile['quality']['n_flagged_labels']} labels were flagged for possible quality issues.",
            "",
        ]

    lines += _mined_discovery_summary(profile)
    lines += _deep_mining_summary(profile)

    if llm:
        lines += ["## Paper-Level Graph Interpretation", ""]
        if llm.get("deep_dive"):
            lines += [llm["deep_dive"], ""]
            residual_missing = llm.get("deep_dive_missing_headings_after_completion",
                                       llm.get("deep_dive_missing_headings", []))
            if residual_missing:
                lines += [
                    "> **LLM completeness warning:** the final deep-dive text is still missing "
                    f"these requested headings: {', '.join(residual_missing)}. "
                    "Increase `--deep-dive-tokens` and rerun if you need a fully structured report.",
                    "",
                ]
        else:
            lines += [
                "LLM summaries were requested, but no paper-level deep-dive text was returned. "
                "Check `profile.json` under `llm_summaries` and rerun with a larger "
                "`--deep-dive-tokens` value if the model truncated or returned an empty answer.",
                "",
            ]
        final_review = llm.get("final_report_review") or {}
        if final_review.get("synthesis"):
            lines += [
                "## Final LLM Report Review - Comprehensive Insight Synthesis",
                "",
                final_review["synthesis"],
                "",
            ]
        if llm.get("deep_dive_passes"):
            lines += ["## LLM Supporting Evidence Passes", ""]
            for name, text in llm["deep_dive_passes"].items():
                title = name.replace("_", " ").title()
                lines += [f"### {title}", "", text, ""]

    lines += [
        "## Source",
        "",
        _md_table([
            ("inferred topic", profile.get("topic")),
            ("input kind", gs["source"].get("kind")),
            ("graph path", gs["source"].get("graph_path")),
            ("run dir", gs["source"].get("run_dir")),
            ("input multigraph", gs["source"].get("input_multigraph")),
            ("input directed", gs["source"].get("input_directed")),
            ("llm model", llm.get("model") if llm else None),
            ("llm backend", llm.get("backend") if llm else None),
            ("llm reasoning effort", llm.get("reasoning_effort") if llm else None),
            ("llm deep passes", len(llm.get("deep_dive_passes", {})) if llm else None),
        ], ["field", "value"]),
        "",
        "## Global Statistics",
        "",
        _md_table([
            ("nodes", gs["nodes"]),
            ("edges", gs["edges"]),
            ("density", gs["density"]),
            ("average degree", gs["avg_degree"]),
            ("isolates", gs["isolates"]),
            ("self loops", gs["self_loops"]),
            ("weak/connected components", gs["weak_or_connected_components"]),
            ("strong components", gs.get("strong_components")),
            ("largest component fraction", gs["largest_component_frac"]),
            ("node iter range", gs["iter"]["nodes"]),
            ("edge iter range", gs["iter"]["edges"]),
            ("node depth range", gs["depth"]),
        ], ["metric", "value"]),
        "",
        "## Figures",
        "",
    ]
    for fig in profile.get("figures", []):
        lines.append(f"![{Path(fig).stem}]({fig})")
        lines.append("")

    lines += ["## Relation Audit", ""]
    lines.append(_md_table([(r, c) for r, c in gs["relations"][:25]], ["relation", "edge count"]))
    lines.append("")

    lines += ["## Top Nodes", ""]
    for key, title in [("degree", "Top By Degree"), ("pagerank", "Top By PageRank"),
                       ("betweenness", "Top By Betweenness")]:
        lines += [f"### {title}", ""]
        rows = [(r["label"], r["value"], r.get("iter"), _short(r.get("question", ""), 90))
                for r in profile["top_nodes"][key][:20]]
        lines.append(_md_table(rows, ["label", key, "iter", "source question"]))
        lines.append("")

    lines += ["## Components", ""]
    rows = []
    for c in profile["components"][:20]:
        terms = ", ".join(t for t, _ in c["top_terms"][:8])
        tops = "; ".join(n["label"] for n in c["top_nodes"][:5])
        rows.append((c["id"], c["size"], c["edges"], c["external_edges"], terms, tops))
    lines.append(_md_table(rows, ["id", "nodes", "edges", "external", "top terms", "top nodes"]))
    lines.append("")

    lines += ["## Module Atlas", ""]
    for m in profile["communities"][:profile["report_limits"]["modules_in_markdown"]]:
        lines += [f"### Module {m['id']} - {m['size']} nodes", ""]
        if llm.get("modules", {}).get(str(m["id"])):
            lines += [llm["modules"][str(m["id"])], ""]
        terms = ", ".join(f"{t} ({c})" for t, c in m["top_terms"][:12])
        rels = ", ".join(f"{r} ({c})" for r, c in m["relations"][:10])
        lines += [
            f"- Iter range: {m['iter']}; depth range: {m['depth']}",
            f"- Internal/external edges: {m['internal_edges']} / {m['external_edges']}",
            f"- Top terms: {terms or '(none)'}",
            f"- Top relations: {rels or '(none)'}",
            "",
            _md_table([(n["label"], n["degree"], n["pagerank"], n.get("iter"),
                        _short(n.get("question", ""), 100)) for n in m["top_nodes"][:10]],
                      ["top node", "degree", "pagerank", "iter", "source question"]),
            "",
        ]
        if m["boundary_nodes"]:
            lines += ["Boundary nodes:", ""]
            lines.append(_md_table([(n["label"], n["external_neighbors"]) for n in m["boundary_nodes"][:8]],
                                   ["label", "external neighbors"]))
            lines.append("")

    mined = profile.get("mined_insights") or {}
    if mined:
        lines += ["## Mined Graph Insights", ""]
        if mined.get("method_notes"):
            lines += ["### Mining Methods", ""]
            for note in mined["method_notes"]:
                lines.append(f"- {note}")
            lines.append("")

        if mined.get("long_range_transitive_paths"):
            lines += ["### Long-Range Transitive Connections", ""]
            lines.append(_md_table([
                (r["score"], r["hops"], r["endpoint_topic_distance"], r["module_diversity"], r["text"])
                for r in mined["long_range_transitive_paths"][:20]
            ], ["score", "hops", "topic distance", "modules", "path"]))
            lines.append("")

        if mined.get("short_cross_module_bridges"):
            lines += ["### Short Cross-Module Bridges", ""]
            lines.append(_md_table([
                (r["score"], r["hops"], r["endpoint_topic_distance"],
                 "; ".join(n["label"] for n in r.get("bridge_nodes", [])) or "(direct)",
                 r["text"])
                for r in mined["short_cross_module_bridges"][:20]
            ], ["score", "hops", "topic distance", "bridge nodes", "path"]))
            lines.append("")

        if mined.get("structural_analogies"):
            lines += ["### Structural Analogies / Role Isomorphism Candidates", ""]
            lines.append(_md_table([
                (r["score"], r["signature_similarity"], r["topic_distance"],
                 r["node_a"]["label"], r["node_a"].get("module"),
                 r["node_b"]["label"], r["node_b"].get("module"),
                 "; ".join(x["role"] for x in r.get("shared_relation_roles", [])[:5]))
                for r in mined["structural_analogies"][:20]
            ], ["score", "role sim", "topic distance", "node A", "module A",
                "node B", "module B", "shared relation roles"]))
            lines.append("")
        else:
            lines += [
                "### Structural Analogies / Role Isomorphism Candidates",
                "",
                "No strong local role-equivalence candidates met the current threshold. "
                "Inspect recurring motifs for weaker analogy patterns.",
                "",
            ]

        iso = mined.get("isomorphism_analysis") or {}
        if iso:
            lines += ["### Isomorphism Analysis", ""]
            for note in iso.get("scope_notes", []):
                lines.append(f"- {note}")
            lines.append("")

            auto = iso.get("whole_graph_automorphism") or {}
            if auto:
                if auto.get("attempted"):
                    auto_value = auto.get("automorphism_count", auto.get("count_lower_bound", "(unknown)"))
                    capped = "yes" if auto.get("capped") else "no"
                    lines.append(f"- Exact whole-graph automorphism enumeration: count={auto_value}; capped={capped}.")
                    if auto.get("error"):
                        lines.append(f"- Automorphism enumeration error: `{auto['error']}`")
                else:
                    lines.append(f"- Exact whole-graph automorphism enumeration: {auto.get('reason')}")
                lines.append("")

            wl = iso.get("wl_orbit_candidates") or {}
            if wl.get("candidate_classes"):
                lines += ["#### WL Orbit Candidate Classes", ""]
                lines.append(
                    f"WL refinement found {wl.get('candidate_class_count')} candidate classes covering "
                    f"{wl.get('candidate_node_count')} nodes after {wl.get('iterations')} iterations."
                )
                lines.append("")
                lines.append(_md_table([
                    (r["score"], r["class_size"], ", ".join(map(str, r["modules"])),
                     ", ".join(map(str, r["degree_values"])),
                     "; ".join(n["label"] for n in r.get("nodes", [])[:8]))
                    for r in wl["candidate_classes"][:15]
                ], ["score", "class size", "modules", "degree values", "example nodes"]))
                lines.append("")
            else:
                lines += ["#### WL Orbit Candidate Classes", "", "No repeated WL node classes met the reporting threshold.", ""]

            egos = iso.get("rooted_ego_isomorphism_classes") or []
            if egos:
                lines += ["#### Exact Rooted Ego-Net Isomorphism Classes", ""]
                lines.append(_md_table([
                    (r["score"], r["radius"], r["class_size"], r["ego_nodes"], r["ego_edges"],
                     ", ".join(map(str, r["modules"])),
                     "; ".join(n["label"] for n in r.get("roots", [])[:8]),
                     "; ".join(f"{rel} ({count})" for rel, count in r.get("relation_counts", [])[:5]))
                    for r in egos[:15]
                ], ["score", "radius", "class size", "ego nodes", "ego edges",
                    "modules", "root examples", "relation counts"]))
                lines.append("")
            else:
                lines += ["#### Exact Rooted Ego-Net Isomorphism Classes", "", "No repeated rooted ego-net isomorphism classes met the reporting threshold.", ""]

            module_iso = iso.get("small_module_isomorphism_classes") or []
            if module_iso:
                lines += ["#### Exact Small-Module Isomorphism Classes", ""]
                lines.append(_md_table([
                    (r["score"], r["class_size"], r["module_nodes"], r["module_edges"],
                     "; ".join(f"module {m['module']} ({m['size']} nodes)"
                               for m in r.get("modules", [])[:6]),
                     "; ".join(f"{rel} ({count})" for rel, count in r.get("relation_counts", [])[:5]))
                    for r in module_iso[:12]
                ], ["score", "class size", "nodes", "edges", "modules", "relation counts"]))
                lines.append("")
            else:
                lines += ["#### Exact Small-Module Isomorphism Classes", "", "No exact small-module isomorphism classes met the reporting threshold.", ""]

        if mined.get("relational_motifs"):
            lines += ["### Recurring Cross-Module Relational Motifs", ""]
            lines.append(_md_table([
                (r["count"], r["relation_chain"], r["role_pattern"],
                 "; ".join(f"{x['label']} ({x['count']})" for x in r.get("top_middle_nodes", [])[:4]),
                 r.get("examples", [{}])[0].get("text", ""))
                for r in mined["relational_motifs"][:20]
            ], ["count", "relation chain", "module role pattern", "top middle nodes", "example"]))
            lines.append("")

        if mined.get("brokerage_nodes"):
            lines += ["### Non-Hub Brokerage Nodes", ""]
            lines.append(_md_table([
                (r["score"], r["label"], r.get("module"), r["external_module_count"],
                 r["relation_diversity"], "; ".join(r.get("context", [])[:3]))
                for r in mined["brokerage_nodes"][:20]
            ], ["score", "label", "module", "external modules", "relation diversity", "context"]))
            lines.append("")

    lines += ["## Inter-Module Edges", ""]
    lines.append(_md_table([(e["module_a"], e["module_b"], e["edge_count"],
                             f"{e['example']['source']} --{e['example']['relation']}--> {e['example']['target']}")
                            for e in profile["module_edges"][:30]],
                           ["module A", "module B", "edges", "example"]))
    lines.append("")

    lines += ["## Critical Connectors And Paths", ""]
    ap = profile["critical_connectors"]["articulation_points"]
    if ap:
        lines += ["### Articulation Points", ""]
        lines.append(_md_table([(r["label"], r["extra_fragments"], r["degree"], r["betweenness"],
                                 r.get("iter"), _short(r.get("question", ""), 90)) for r in ap[:20]],
                               ["label", "extra fragments", "degree", "betweenness", "iter", "source question"]))
        lines.append("")
    bridges = profile["critical_connectors"]["bridge_edges"]
    if bridges:
        lines += ["### Bridge Edges", ""]
        lines.append(_md_table([(r["source"], r["target"], r["source_degree"], r["target_degree"])
                                for r in bridges[:20]], ["source", "target", "source degree", "target degree"]))
        lines.append("")
    if profile["representative_paths"]:
        lines += ["### Representative Cross-Module Paths", ""]
        for i, p in enumerate(profile["representative_paths"][:20], 1):
            lines.append(f"{i}. {p['text']}")
        lines.append("")

    sem = profile.get("semantic_audit") or {}
    if sem:
        lines += ["## Semantic Audit", ""]
        lines.append(f"- Embedding model: {profile.get('embedding_model')}")
        lines.append(f"- Embedding spread: {sem.get('embedding_spread'):.4g}")
        lines.append("")
        if sem.get("semantic_outliers"):
            lines += ["### Semantic Outliers", ""]
            lines.append(_md_table([(r["label"], r["distance_from_centroid"], r.get("module"), r.get("iter"))
                                    for r in sem["semantic_outliers"][:20]],
                                   ["label", "distance from centroid", "module", "iter"]))
            lines.append("")
        if sem.get("distant_connected_paths"):
            lines += ["### Distant Connected Paths", ""]
            for i, p in enumerate(sem["distant_connected_paths"][:15], 1):
                lines.append(f"{i}. distance={p['distance']:.3f}: {p['text']}")
            lines.append("")
    elif profile.get("embedding_error"):
        lines += ["## Semantic Audit", "", f"Embedding skipped/failed: `{profile['embedding_error']}`", ""]

    lines += ["## Provenance", ""]
    prov = profile["provenance"]
    lines.append(f"- Transcript rows: {prov['n_transcript_rows']}")
    lines.append(f"- Zero-yield iterations recorded: {len(prov['zero_yield_iterations'])}")
    lines.append(f"- High-token zero-yield iterations shown: {len(prov['high_token_zero_yield_iterations'])}")
    lines.append("")
    if prov["source_questions_by_new_nodes"]:
        lines += ["### Source Questions By Nodes Introduced", ""]
        lines.append(_md_table([(q, c) for q, c in prov["source_questions_by_new_nodes"][:20]],
                               ["question", "node count"]))
        lines.append("")
    if prov["high_token_zero_yield_iterations"]:
        lines += ["### High-Token Zero-Yield Iterations", ""]
        lines.append(_md_table([(r.get("iter"), r.get("tokens"), r.get("question"), r.get("answer_excerpt"))
                                for r in prov["high_token_zero_yield_iterations"][:15]],
                               ["iter", "tokens", "question", "answer excerpt"]))
        lines.append("")

    lines += ["## Data Quality Flags", ""]
    q = profile["quality"]
    lines.append(f"Flagged labels: {q['n_flagged_labels']}")
    lines.append("")
    if q["flagged_labels"]:
        lines.append(_md_table([(r["label"], ", ".join(r["flags"]), r.get("iter"),
                                 _short(r.get("question", ""), 90)) for r in q["flagged_labels"][:80]],
                               ["label", "flags", "iter", "source question"]))
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_pdf(report_path: Path, out_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        return None, "pandoc not found"
    engine = shutil.which("xelatex") or shutil.which("lualatex") or shutil.which("pdflatex")
    if not engine:
        return None, "no LaTeX PDF engine found (expected xelatex, lualatex, or pdflatex)"
    pdf_path = out_dir / "report.pdf"
    cmd = [
        pandoc,
        report_path.name,
        "-o", pdf_path.name,
        f"--pdf-engine={Path(engine).name}",
        "--toc",
        "-V", "geometry:margin=0.75in",
        "-V", "colorlinks=true",
    ]
    try:
        r = subprocess.run(cmd, cwd=out_dir, text=True, capture_output=True, check=False)
    except Exception as exc:
        return None, str(exc)
    if r.returncode != 0:
        msg = "\n".join(x for x in (r.stderr.strip(), r.stdout.strip()) if x)
        return None, msg or f"pandoc exited with code {r.returncode}"
    return pdf_path, None


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, set):
        return sorted(str(v) for v in x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def profile_graph(graph_path: Optional[str] = None, run_dir: Optional[str] = None,
                  out: Optional[str] = None, embed_model: Optional[str] = None,
                  top_nodes: int = 25, max_modules: int = 30, llm_options: Optional[LLMOptions] = None,
                  llm_modules: int = 12, verbose: bool = False, pdf: bool = True) -> Dict[str, Any]:
    total_steps = 12
    if embed_model:
        total_steps += 1
    if llm_options and llm_options.enabled:
        total_steps += 1
        if llm_options.report_review:
            total_steps += 1
    if pdf:
        total_steps += 1
    progress = _Progress(verbose, total_steps)

    progress.step("Loading graph")
    G, source = _load_graph(graph_path, run_dir)
    run_path = Path(source["run_dir"]) if source.get("run_dir") else None
    out_dir = Path(out or (str(run_path / "profile") if run_path else "graph_profile")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress.detail(f"graph: {source.get('graph_path')}")
    progress.detail(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} directed={G.is_directed()}")

    progress.step("Reading run transcript and preparing output directory")
    run_summary = _read_run_summary(run_path)
    transcript = _read_transcript(run_path)
    topic = _infer_topic(G, run_summary, transcript)
    progress.detail(f"out: {out_dir}")
    if topic:
        progress.detail(f"topic: {topic}")
    progress.detail(f"transcript rows: {len(transcript)}")

    progress.step("Computing centrality metrics")
    cents = _centralities(G)

    progress.step("Profiling connected components")
    components = _component_profiles(G, cents, top=30)

    progress.step("Detecting graph communities/modules")
    communities, module_of, modularity = _community_profiles(G, cents, max_modules=max_modules)
    progress.detail(f"reported modules: {len(communities)}; modularity={modularity:.4g}")

    progress.step("Compiling global statistics and data-quality flags")
    global_stats = _global_stats(G, source, components, communities, modularity)
    quality = _quality_report(G)
    progress.detail(f"flagged labels: {quality['n_flagged_labels']}")

    progress.step("Analyzing provenance")
    provenance = _provenance_report(G, run_path, transcript)
    progress.detail(f"zero-yield iterations: {len(provenance['zero_yield_iterations'])}")

    progress.step("Finding critical connectors, module edges, and representative paths")
    critical = _critical_connectors(G, cents, max_rows=30)
    module_edges = _module_edges(G, module_of, max_edges=60)
    representative_paths = _representative_paths(G, cents, module_of, max_paths=30)

    vecs, resolved_model, embed_error = None, None, None
    if embed_model:
        progress.step(f"Running semantic audit embeddings ({embed_model})")
        vecs, resolved_model, embed_error = _try_embed(G, run_path, embed_model)
        if embed_error:
            progress.detail(f"embedding skipped/failed: {embed_error}")
        else:
            progress.detail(f"embedding model: {resolved_model}")
    semantic = _semantic_audit(G, vecs, module_of) if vecs else {}

    progress.step("Mining transitive paths, motifs, analogies, and broker nodes")
    mined_insights = _mine_graph_insights(G, cents, module_of, vecs)
    iso_counts = mined_insights.get("isomorphism_analysis", {})
    progress.detail(
        "mined: "
        f"{len(mined_insights['long_range_transitive_paths'])} long paths, "
        f"{len(mined_insights['short_cross_module_bridges'])} short bridges, "
        f"{len(mined_insights['relational_motifs'])} motifs, "
        f"{len(mined_insights['structural_analogies'])} analogies, "
        f"{len(iso_counts.get('rooted_ego_isomorphism_classes', []))} ego-isomorphism classes"
    )

    profile: Dict[str, Any] = {
        "generated_at": _now(),
        "topic": topic,
        "run_summary": run_summary,
        "global_stats": global_stats,
        "components": components,
        "communities": communities,
        "module_edges": module_edges,
        "top_nodes": {
            "degree": _top_items(cents["degree"], G, top_nodes),
            "pagerank": _top_items(cents["pagerank"], G, top_nodes),
            "betweenness": _top_items(cents["betweenness"], G, top_nodes),
            "in_degree": _top_items(cents["in_degree"], G, top_nodes) if cents["in_degree"] else [],
            "out_degree": _top_items(cents["out_degree"], G, top_nodes) if cents["out_degree"] else [],
        },
        "critical_connectors": critical,
        "representative_paths": representative_paths,
        "quality": quality,
        "provenance": provenance,
        "semantic_audit": semantic,
        "mined_insights": mined_insights,
        "embedding_model": resolved_model,
        "embedding_error": embed_error,
        "report_limits": {"modules_in_markdown": min(max_modules, 18)},
        "_centrality_values": cents,
    }
    profile["deep_evidence"] = _deep_graph_evidence(G, cents, module_of, profile)

    progress.step("Writing diagnostic figures")
    profile["figures"] = _save_figures(profile, G, vecs, module_of, out_dir)
    if llm_options and llm_options.enabled:
        progress.step(f"Running LLM summaries ({llm_options.model})")
        profile["llm_summaries"] = _llm_summaries(profile, llm_options, llm_modules,
                                                  progress=progress.detail)

    progress.step("Writing Markdown report")
    report = _write_markdown(profile, out_dir)
    profile["report_path"] = str(report)

    if llm_options and llm_options.enabled and llm_options.report_review:
        progress.step("Running final LLM report review")
        profile.setdefault("llm_summaries", {})
        profile["llm_summaries"]["final_report_review"] = _llm_final_report_review(
            profile, report, llm_options, progress=progress.detail)
        report = _write_markdown(profile, out_dir)
        profile["report_path"] = str(report)

    if pdf:
        progress.step("Rendering PDF report")
        pdf_path, pdf_error = _write_pdf(report, out_dir)
        profile["pdf_path"] = str(pdf_path) if pdf_path else None
        profile["pdf_error"] = pdf_error
        if pdf_path:
            progress.detail(f"pdf: {pdf_path}")
        elif pdf_error:
            progress.detail(f"PDF skipped/failed: {pdf_error}")

    progress.step("Writing JSON profile")
    profile["json_path"] = str(out_dir / "profile.json")
    (out_dir / "profile.json").write_text(json.dumps(_json_safe(profile), indent=2), encoding="utf-8")
    progress.finish()
    return profile


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="Graph-PRefLexOR run directory containing graph.graphml")
    src.add_argument("--graph", help="Any GraphML file to profile")
    p.add_argument("--out", help="output directory (default: <run>/profile or ./graph_profile)")
    p.add_argument("--embed-model", default=None,
                   help="optional sentence-transformers model for semantic audit; use 'auto' for run/default")
    p.add_argument("--top-nodes", type=int, default=25)
    p.add_argument("--max-modules", type=int, default=30)
    p.add_argument("--llm", action="store_true",
                   help="ask an LLM for module summaries, overview, and a paper-level deep dive")
    p.add_argument("--llm-modules", type=int, default=12, help="number of largest modules to summarize")
    p.add_argument("--backend", choices=["responses", "openai", "chat", "hf"], default="responses",
                   help="LLM backend. responses/openai use OpenAI Responses API; chat is for local OpenAI-compatible servers without Responses.")
    p.add_argument("--model", default="gpt-5.5", help="summary model id")
    p.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    p.add_argument("--api-key", help="API key, else $OPENAI_API_KEY or 'x' for local servers")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-summary-tokens", type=int, default=1200)
    p.add_argument("--deep-pass-tokens", type=int, default=5000,
                   help="output-token budget for each deep evidence-pass LLM call")
    p.add_argument("--deep-dive-tokens", type=int, default=10000,
                   help="output-token budget for the paper-level LLM deep dive")
    p.add_argument("--reasoning-effort", default="high", choices=["minimal", "low", "medium", "high"],
                   help="[responses/openai] reasoning effort for the LLM analysis calls")
    p.add_argument("--llm-deep-passes", type=int, default=4,
                   help="number of extra LLM evidence passes before the final deep dive; 0 disables")
    p.add_argument("--llm-report-review", dest="llm_report_review", action="store_true", default=True,
                   help="after writing the draft report, run a final LLM review/synthesis pass (default with --llm)")
    p.add_argument("--no-llm-report-review", dest="llm_report_review", action="store_false",
                   help="skip the final report-level LLM review pass")
    p.add_argument("--report-review-tokens", type=int, default=8000,
                   help="output-token budget for the final report-level synthesis")
    p.add_argument("--report-review-max-chunks", type=int, default=0,
                   help="maximum report sections to review before final synthesis; 0 means all major sections")
    p.add_argument("--report-review-chunk-chars", type=int, default=0,
                   help="maximum characters per report section sent to the final review pass; 0 means no clipping")
    p.add_argument("--report-review-memo-chars", type=int, default=0,
                   help="maximum characters per section-review memo sent to final synthesis; 0 means no clipping")
    p.add_argument("--device", help="[hf] device_map")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--pdf", dest="pdf", action="store_true", default=True,
                   help="render report.pdf with pandoc after writing report.md (default)")
    p.add_argument("--no-pdf", dest="pdf", action="store_false", help="skip report.pdf rendering")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = p.parse_args()

    opts = LLMOptions(enabled=args.llm, backend=args.backend, model=args.model,
                      base_url=args.base_url, api_key=args.api_key,
                      temperature=args.temperature, max_tokens=args.max_summary_tokens,
                      deep_pass_tokens=args.deep_pass_tokens, deep_tokens=args.deep_dive_tokens,
                      reasoning_effort=args.reasoning_effort,
                      deep_passes=args.llm_deep_passes,
                      report_review=args.llm_report_review,
                      report_review_tokens=args.report_review_tokens,
                      report_review_max_chunks=args.report_review_max_chunks,
                      report_review_chunk_chars=args.report_review_chunk_chars,
                      report_review_memo_chars=args.report_review_memo_chars,
                      device=args.device, dtype=args.dtype)
    prof = profile_graph(graph_path=args.graph, run_dir=args.run, out=args.out,
                         embed_model=args.embed_model, top_nodes=args.top_nodes,
                         max_modules=args.max_modules, llm_options=opts,
                         llm_modules=args.llm_modules, verbose=not args.quiet, pdf=args.pdf)
    if not args.quiet:
        print(f"wrote {prof['report_path']}")
        if prof.get("pdf_path"):
            print(f"wrote {prof['pdf_path']}")
        elif prof.get("pdf_error"):
            print(f"PDF not written: {prof['pdf_error']}")
        print(f"wrote {prof['json_path']}")


if __name__ == "__main__":
    main()
