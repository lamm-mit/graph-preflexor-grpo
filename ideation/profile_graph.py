#!/usr/bin/env python
"""Deep graph profile / audit report for Graph-PRefLexOR or any GraphML file.

The normal plotting scripts answer "how well did the run do?" This script answers
"what is actually in this graph?" It builds a graph atlas with global statistics,
components, modularity communities, top nodes, critical connectors, relation
patterns, provenance, data-quality warnings, figures, and optional LLM-written
module summaries.

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
        --reasoning-effort high --deep-pass-tokens 5000 --deep-dive-tokens 12000

    # Local servers without Responses API can use the chat backend.
    python profile_graph.py --graph graph.graphml --llm --backend chat \
        --model meta-llama/Llama-3.2-3B-Instruct --base-url http://localhost:8000/v1 \
        --deep-pass-tokens 5000 --deep-dive-tokens 12000
"""
from __future__ import annotations

import argparse
import csv
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
}

GENERIC_LABELS = {
    "a", "b", "c", "d", "e", "f", "g", "h", "node", "concept", "idea", "thing",
    "entity", "mechanism", "process", "system", "property", "properties", "effect",
    "effects", "response", "interaction", "interactions", "structure", "function",
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
    deep_passes: int = 3
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
    if len(norm) <= 2:
        flags.append("too_short")
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
        rels = []
        for u, v in zip(p, p[1:]):
            d = G.get_edge_data(u, v) or G.get_edge_data(v, u) or {}
            rels.append(_edge_relation(d if isinstance(d, dict) else {}))
        out.append({
            "score": float(score),
            "nodes": [_label(G, n) for n in p],
            "node_ids": [str(n) for n in p],
            "modules": [module_of.get(n) for n in p],
            "relations": rels,
            "text": _path_text(G, p),
        })
    return out


def _path_text(G: nx.Graph, path: Sequence[Any]) -> str:
    bits = []
    for i, n in enumerate(path):
        bits.append(_label(G, n))
        if i < len(path) - 1:
            d = G.get_edge_data(path[i], path[i + 1]) or G.get_edge_data(path[i + 1], path[i]) or {}
            bits.append(f"--{_edge_relation(d if isinstance(d, dict) else {})}-->")
    return " ".join(bits)


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
        "5. Novel Or Speculative Hypotheses Worth Human Review\n"
        "6. What Looks Mature Versus What Looks Like Frontier Drift\n"
        "7. Provenance And Search Dynamics\n"
        "8. Reliability, Noise, And Failure Modes\n"
        "9. Concrete Next Analyses Or Experiments\n"
        "10. Short Abstract Suitable For A Paper Or Lab Notebook\n\n"
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
        }
        task = (
            "Analyze bridge concepts and graph-generated hypotheses. Focus on non-obvious bridges "
            "after obvious hubs are removed. Extract candidate hypotheses that a researcher should "
            "inspect, and cite the exact paths, relation verbs, module ids, and node labels that "
            "support each hypothesis."
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
    names = ["mechanistic_programs", "bridges_and_hypotheses", "search_dynamics_reliability"]
    return names[:max(0, min(n, len(names)))]


DEEP_DIVE_HEADINGS = [
    "Central Thesis",
    "What The Graph Discovered About The Topic",
    "Major Mechanistic Programs",
    "Cross-Module Bridges And Critical Concepts",
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
        return str(x).replace("\n", " ")
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(cell(x) for x in row) + " |")
    return "\n".join(out)


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
    total_steps = 11
    if embed_model:
        total_steps += 1
    if llm_options and llm_options.enabled:
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
    p.add_argument("--llm-deep-passes", type=int, default=3,
                   help="number of extra LLM evidence passes before the final deep dive; 0 disables")
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
