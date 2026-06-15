#!/usr/bin/env python
"""Local browser explorer for Graph-PRefLexOR graphml runs.

This is intentionally dependency-light: stdlib HTTP server plus NetworkX. The
browser does the interactive rendering; this process loads GraphML, runs graph
queries, optionally starts ideate.py jobs, and calls a user-selected LLM backend.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import urllib.error
import urllib.request

import networkx as nx


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
REACT_DIST_DIR = ROOT / "frontend" / "dist"
LOGO_PATH = ROOT / "logo.gif"
IDEATION_DIR = ROOT.parent
PROJECT_DIR = IDEATION_DIR.parent
EMBEDDING_CACHE_VERSION = "graph-explorer-embeddings-v1"
EMBEDDING_CACHE_DIR = IDEATION_DIR / ".cache" / "graph_explorer" / "embeddings"
DEFAULT_CHAT_MAX_OUTPUT_TOKENS = 20000
TOKEN_LIMIT_BACKOFF = (20000, 16384, 8192, 4096, 2048)
LLM_HTTP_TIMEOUT_SECONDS = 600
GRAPH_PREFLEXOR_ASSISTANT_PROMPT = (
    "You are Graph-PRefLexOR Assistant, a graph-aware research copilot for "
    "scientific ideation. Help the user inspect generated concept graphs, reason "
    "over selected nodes, neighborhoods, bridges, paths, communities, and attached "
    "reports, and propose concrete next analyses. Treat graph context as exploratory "
    "evidence rather than verified fact. If graph or report context is absent or "
    "insufficient, say so and answer from general reasoning without inventing graph "
    "evidence."
)

STATE = {
    "graph": None,
    "graph_id": None,
    "graph_name": "",
    "graph_path": "",
    "topic": "",
    "jobs": {},
    "profile_jobs": {},
    "embedding_job": None,
    "embedding_index": None,
    "embedding_models": {},
    "hf_cache": {},
}
LOCK = threading.RLock()


def _clean(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    return str(value)


def _int_attr(attrs, name, default=0):
    try:
        return int(float(attrs.get(name, default)))
    except Exception:
        return default


def _edge_iter(G):
    if G.is_multigraph():
        for u, v, k, d in G.edges(keys=True, data=True):
            yield str(u), str(v), str(k), dict(d)
    else:
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            yield str(u), str(v), str(i), dict(d)


def _normalize_graph(G):
    """Return a simple graph while preserving useful edge attributes."""
    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    for n, d in G.nodes(data=True):
        H.add_node(str(n), **{str(k): _clean(v) for k, v in dict(d).items()})
    for u, v, _, d in _edge_iter(G):
        attrs = {str(k): _clean(vv) for k, vv in d.items()}
        if H.has_edge(u, v):
            old = H[u][v]
            old_rel = str(old.get("relation", "related_to"))
            new_rel = str(attrs.get("relation", "related_to"))
            if new_rel not in old_rel.split(" | "):
                old["relation"] = old_rel + " | " + new_rel
            old["multiplicity"] = int(old.get("multiplicity", 1)) + 1
        else:
            attrs.setdefault("relation", "related_to")
            attrs.setdefault("multiplicity", 1)
            H.add_edge(u, v, **attrs)
    return H


def _load_topic(run_dir, G=None):
    if not run_dir:
        return ""
    summary = Path(run_dir) / "summary.json"
    if summary.exists():
        try:
            return json.load(open(summary)).get("topic", "")
        except Exception:
            pass
    if G is not None:
        topics = []
        for _, d in list(G.nodes(data=True))[:100]:
            t = d.get("topic")
            if t:
                topics.append(str(t))
        if topics:
            return max(set(topics), key=topics.count)
    return ""


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _chat_max_tokens(cfg, default=DEFAULT_CHAT_MAX_OUTPUT_TOKENS):
    return max(256, _safe_int((cfg or {}).get("max_tokens"), default))


def _is_token_limit_error(message):
    msg = str(message or "").lower()
    return any(
        token in msg
        for token in (
            "max_output_tokens",
            "max_completion_tokens",
            "max_tokens",
            "maximum",
            "less than or equal",
            "greater than",
            "too high",
            "context length",
            "context window",
        )
    )


def _is_unsupported_param_error(message):
    msg = str(message or "").lower()
    return any(token in msg for token in ("unsupported", "unexpected", "not supported", "unknown", "unrecognized"))


def _lower_token_limit(kwargs, key):
    current = _safe_int(kwargs.get(key), DEFAULT_CHAT_MAX_OUTPUT_TOKENS)
    for limit in TOKEN_LIMIT_BACKOFF:
        if current > limit:
            kwargs[key] = limit
            return True
    if key in kwargs:
        kwargs.pop(key, None)
        return True
    return False


def _component_map(G):
    U = G.to_undirected()
    comps = sorted(nx.connected_components(U), key=len, reverse=True)
    comp = {}
    sizes = []
    for i, nodes in enumerate(comps):
        sizes.append(len(nodes))
        for n in nodes:
            comp[str(n)] = i
    return comp, sizes


def _community_map(G):
    signature = (bool(G.is_directed()), G.number_of_nodes(), G.number_of_edges())
    cached = G.graph.get("_explorer_community_cache")
    if cached and cached.get("signature") == signature:
        return cached["community"], cached["sizes"]

    U = G.to_undirected()
    if U.number_of_nodes() == 0:
        community, sizes = {}, []
    elif U.number_of_edges() == 0 or U.number_of_nodes() > 3000:
        community = {str(n): i for i, n in enumerate(U.nodes)}
        sizes = [1 for _ in U.nodes]
    else:
        try:
            groups = sorted(nx.community.greedy_modularity_communities(U), key=len, reverse=True)
            community = {}
            sizes = []
            for i, nodes in enumerate(groups):
                sizes.append(len(nodes))
                for n in nodes:
                    community[str(n)] = i
        except Exception:
            community, sizes = _component_map(G)
    G.graph["_explorer_community_cache"] = {
        "signature": signature,
        "community": community,
        "sizes": sizes,
    }
    return community, sizes


def _centrality(G):
    signature = (bool(G.is_directed()), G.number_of_nodes(), G.number_of_edges())
    cached = G.graph.get("_explorer_centrality_cache")
    if cached and cached.get("signature") == signature:
        return cached["metrics"]

    U = G.to_undirected()
    degree = dict(U.degree())
    n = U.number_of_nodes()
    metrics = {
        "degree": degree,
        "pagerank": {},
        "core": {},
        "closeness": {},
        "betweenness": {},
        "clustering": {},
        "eigenvector": {},
    }
    try:
        metrics["pagerank"] = nx.pagerank(G if G.number_of_edges() else U, max_iter=100)
    except Exception:
        total = max(1, sum(degree.values()))
        metrics["pagerank"] = {node: degree.get(node, 0) / total for node in U.nodes}
    try:
        metrics["core"] = nx.core_number(U) if U.number_of_edges() else {node: 0 for node in U.nodes}
    except Exception:
        metrics["core"] = {node: 0 for node in U.nodes}
    try:
        metrics["clustering"] = nx.clustering(U) if U.number_of_edges() else {node: 0.0 for node in U.nodes}
    except Exception:
        pass
    try:
        if n <= 900:
            metrics["closeness"] = nx.closeness_centrality(U)
    except Exception:
        pass
    try:
        if U.number_of_edges():
            if n <= 220:
                metrics["betweenness"] = nx.betweenness_centrality(U, normalized=True)
            else:
                metrics["betweenness"] = nx.betweenness_centrality(
                    U, k=min(96, n), normalized=True, seed=7
                )
    except Exception:
        pass
    try:
        if U.number_of_edges() and n <= 1200:
            metrics["eigenvector"] = nx.eigenvector_centrality(U, max_iter=300, tol=1e-04)
    except Exception:
        metrics["eigenvector"] = metrics["pagerank"]
    G.graph["_explorer_centrality_cache"] = {"signature": signature, "metrics": metrics}
    return metrics


def graph_payload(G, *, name="", path="", topic="", node_subset=None, edge_subset=None, include_attrs=True):
    comp, comp_sizes = _component_map(G)
    community, community_sizes = _community_map(G)
    metrics = _centrality(G)
    degree = metrics["degree"]
    pagerank = metrics["pagerank"]
    core = metrics["core"]
    closeness = metrics["closeness"]
    betweenness = metrics["betweenness"]
    clustering = metrics["clustering"]
    eigenvector = metrics["eigenvector"]
    node_filter = set(map(str, node_subset)) if node_subset is not None else None
    edge_filter = set(edge_subset or []) if edge_subset is not None else None

    nodes = []
    for n, d in G.nodes(data=True):
        sid = str(n)
        if node_filter is not None and sid not in node_filter:
            continue
        attrs = {str(k): _clean(v) for k, v in dict(d).items()}
        label = str(attrs.get("label") or sid)
        nodes.append({
            "id": sid,
            "label": label,
            "degree": int(degree.get(n, 0)),
            "pagerank": float(pagerank.get(n, 0.0)),
            "core": int(core.get(n, 0)),
            "closeness": float(closeness.get(n, 0.0)),
            "betweenness": float(betweenness.get(n, 0.0)),
            "clustering": float(clustering.get(n, 0.0)),
            "eigenvector": float(eigenvector.get(n, 0.0)),
            "component": int(comp.get(sid, -1)),
            "community": int(community.get(sid, -1)),
            "iter": _int_attr(attrs, "iter", 0),
            "depth": _int_attr(attrs, "depth", 0),
            "attrs": attrs if include_attrs else {},
        })

    edges = []
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        su, sv = str(u), str(v)
        eid = f"e{i}"
        if node_filter is not None and (su not in node_filter or sv not in node_filter):
            continue
        if edge_filter is not None and eid not in edge_filter:
            continue
        attrs = {str(k): _clean(vv) for k, vv in dict(d).items()}
        edges.append({
            "id": eid,
            "source": su,
            "target": sv,
            "relation": str(attrs.get("relation") or "related_to"),
            "iter": _int_attr(attrs, "iter", 0),
            "depth": _int_attr(attrs, "depth", 0),
            "attrs": attrs if include_attrs else {},
        })

    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "directed": bool(G.is_directed()),
        "components": len(comp_sizes),
        "largest_component": comp_sizes[0] if comp_sizes else 0,
        "density": float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
        "max_degree": max(degree.values()) if degree else 0,
        "avg_degree": float(sum(degree.values()) / max(1, len(degree))),
        "max_iter": max((_int_attr(d, "iter", 0) for _, d in G.nodes(data=True)), default=0),
        "component_sizes": comp_sizes[:40],
        "communities": len(community_sizes),
        "community_sizes": community_sizes[:40],
    }
    return {
        "graph_id": STATE.get("graph_id"),
        "name": name,
        "path": path,
        "topic": topic,
        "stats": stats,
        "nodes": nodes,
        "edges": edges,
    }


def _read_graphml_text(text):
    try:
        return _normalize_graph(nx.parse_graphml(text))
    except Exception:
        with tempfile.NamedTemporaryFile("w", suffix=".graphml", delete=False) as f:
            f.write(text)
            tmp = f.name
        try:
            return _normalize_graph(nx.read_graphml(tmp))
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _available_runs_message(limit=20):
    runs_dir = IDEATION_DIR / "runs"
    if not runs_dir.exists():
        return f"no runs directory at {runs_dir}"
    runs = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    if not runs:
        return f"no run directories under {runs_dir}"
    shown = ", ".join(runs[:limit])
    if len(runs) > limit:
        shown += f", ... (+{len(runs) - limit} more)"
    return f"available runs: {shown}"


def _latest_existing_mtime(paths, fallback):
    mtimes = []
    for path in paths:
        try:
            if path and Path(path).exists():
                mtimes.append(Path(path).stat().st_mtime)
        except OSError:
            pass
    return max(mtimes) if mtimes else fallback


def _read_summary(run_dir):
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        with path.open(errors="replace") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _run_summary(run_dir):
    summary = _read_summary(run_dir)
    metrics = dict(summary.get("metrics") or {})
    stats = dict(summary.get("stats") or {})
    config = dict(summary.get("config") or {})
    snapshot = _snapshot_meta(run_dir)
    progress = _progress_payload(run_dir)
    rel_path = str(run_dir)
    try:
        rel_path = str(run_dir.relative_to(IDEATION_DIR))
    except ValueError:
        pass
    updated_at = _latest_existing_mtime(
        [
            run_dir,
            run_dir / "summary.json",
            run_dir / "growth.csv",
            snapshot.get("graph_path") or "",
        ],
        time.time(),
    )
    nodes = progress.get("nodes") or _safe_int(metrics.get("nodes"))
    edges = progress.get("edges") or _safe_int(metrics.get("edges"))
    calls = progress.get("calls") or _safe_int(stats.get("calls"))
    iters = progress.get("iter")
    if iters is None or iters < 0:
        iters = _safe_int(stats.get("iters"))
    return {
        "name": run_dir.name,
        "path": rel_path,
        "absolute_path": str(run_dir),
        "topic": str(summary.get("topic") or ""),
        "strategy": str(config.get("strategy") or ""),
        "updated_at": updated_at,
        "graph_ready": bool(snapshot.get("graph_ready")),
        "graph_path": snapshot.get("graph_path") or "",
        "snapshot_count": snapshot.get("snapshot_count") or 0,
        "snapshot_iter": snapshot.get("snapshot_iter"),
        "nodes": nodes,
        "edges": edges,
        "calls": calls,
        "iters": iters,
        "status": "graph ready" if snapshot.get("graph_ready") else "waiting for graph",
    }


def _runs_payload():
    runs_dir = IDEATION_DIR / "runs"
    if not runs_dir.exists():
        return {"root": str(runs_dir), "runs": []}
    runs = [_run_summary(path) for path in runs_dir.iterdir() if path.is_dir()]
    runs.sort(key=lambda item: item.get("updated_at") or 0, reverse=True)
    return {"root": str(runs_dir), "runs": runs}


def _resolve_run_path(run_value):
    raw = str(run_value or "").strip()
    if not raw:
        raise ValueError("run path is required")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path

    candidates = []
    if path.parts and path.parts[0] == IDEATION_DIR.name:
        candidates.append(PROJECT_DIR / path)
    candidates.extend([IDEATION_DIR / path, Path.cwd() / path])
    candidates = list(dict.fromkeys(candidates))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in candidates:
        if candidate.parent.exists():
            return candidate
    return candidates[0]


def _relative_to_ideation(path):
    try:
        return str(Path(path).resolve().relative_to(IDEATION_DIR))
    except Exception:
        return str(path)


def _safe_workspace_path(raw, *, default_base=IDEATION_DIR, require_safe_root=True):
    value = str(raw or "").strip()
    if not value:
        raise ValueError("path is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        if path.parts and path.parts[0] == IDEATION_DIR.name:
            path = PROJECT_DIR / path
        else:
            path = default_base / path
    resolved = path.resolve()
    if require_safe_root:
        roots = [PROJECT_DIR.resolve(), Path(tempfile.gettempdir()).resolve()]
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise ValueError(f"path is outside the project workspace: {resolved}")
    return resolved


def _resolve_profile_out(out_value):
    return _safe_workspace_path(out_value, default_base=IDEATION_DIR, require_safe_root=True)


def _clear_ideate_artifacts(out_value):
    """Remove stale graph outputs for a new Explorer-launched run.

    This is intentionally narrow: only generated ideate.py artifacts under
    ideation/runs are removed. Profile/report subdirectories are preserved.
    """
    run_dir = _safe_workspace_path(out_value, default_base=IDEATION_DIR, require_safe_root=True)
    runs_root = (IDEATION_DIR / "runs").resolve()
    if not (run_dir == runs_root or runs_root in run_dir.parents):
        return []
    removed = []
    for name in ("graph.graphml", "transcript.jsonl", "growth.csv", "summary.json"):
        path = run_dir / name
        if path.exists() and path.is_file():
            path.unlink()
            removed.append(_relative_to_ideation(path))
    snapshot_dir = run_dir / "graphml"
    if snapshot_dir.exists() and snapshot_dir.is_dir():
        shutil.rmtree(snapshot_dir)
        removed.append(_relative_to_ideation(snapshot_dir))
    return removed


def _graphml_candidates(run_dir):
    graph_path = run_dir / "graph.graphml"
    candidates = []
    if graph_path.exists() and graph_path.stat().st_size > 0:
        candidates.append(graph_path)
    snapshot_dir = run_dir / "graphml"
    if snapshot_dir.exists():
        candidates.extend(
            p for p in sorted(snapshot_dir.glob("iter_*.graphml"), reverse=True)
            if p.is_file() and p.stat().st_size > 0
        )

    def sort_key(path):
        try:
            stat = path.stat()
            mtime = stat.st_mtime_ns
        except OSError:
            mtime = 0
        match = re.search(r"iter_(\d+)\.graphml$", path.name)
        iteration = int(match.group(1)) if match else -1
        return (mtime, iteration, 1 if path.name == "graph.graphml" else 0)

    return sorted(candidates, key=sort_key, reverse=True)


def _iter_from_snapshot(path):
    match = re.search(r"iter_(\d+)\.graphml$", path.name)
    return int(match.group(1)) if match else None


def _snapshot_meta(run_dir):
    candidates = _graphml_candidates(run_dir) if run_dir.exists() and run_dir.is_dir() else []
    latest = candidates[0] if candidates else run_dir / "graph.graphml"
    snapshot_count = 0
    if run_dir.exists() and run_dir.is_dir():
        snapshot_dir = run_dir / "graphml"
        if snapshot_dir.exists():
            snapshot_count = len([p for p in snapshot_dir.glob("iter_*.graphml") if p.is_file()])
    try:
        stat = latest.stat()
        signature = f"{latest}:{stat.st_mtime_ns}:{stat.st_size}"
        mtime = stat.st_mtime
        size = stat.st_size
    except OSError:
        signature = ""
        mtime = None
        size = 0
    return {
        "graph_ready": bool(candidates),
        "graph_path": str(latest),
        "snapshot_id": signature,
        "snapshot_count": snapshot_count,
        "snapshot_iter": _iter_from_snapshot(latest),
        "snapshot_mtime": mtime,
        "snapshot_size": size,
    }


def _graph_file_item(path, *, run_dir=None, latest_path=None):
    path = Path(path)
    try:
        stat = path.stat()
        updated_at = stat.st_mtime
        size = stat.st_size
    except OSError:
        updated_at = 0
        size = 0
    run_dir = Path(run_dir) if run_dir else None
    run_rel = _relative_to_ideation(run_dir) if run_dir else ""
    return {
        "name": path.name,
        "path": _relative_to_ideation(path),
        "absolute_path": str(path),
        "run": run_rel,
        "run_name": run_dir.name if run_dir else "",
        "iter": _iter_from_snapshot(path),
        "updated_at": updated_at,
        "size": size,
        "is_latest": bool(latest_path and Path(latest_path) == path),
    }


def _run_graphs_payload(run_value):
    target = _resolve_run_path(run_value)
    if target.is_file():
        run_dir = target.parent.parent if target.parent.name == "graphml" else target.parent
        candidates = [target]
    elif target.is_dir():
        run_dir = target
        candidates = _graphml_candidates(run_dir)
    else:
        raise ValueError(f"run path is neither a file nor a directory: {target}")
    latest = candidates[0] if candidates else None
    return {
        "run": _relative_to_ideation(run_dir),
        "graphs": [_graph_file_item(path, run_dir=run_dir, latest_path=latest) for path in candidates],
    }


def _graphml_files_payload(limit=500):
    items = []
    runs_dir = IDEATION_DIR / "runs"
    if runs_dir.exists():
        for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
            candidates = _graphml_candidates(run_dir)
            latest = candidates[0] if candidates else None
            for path in candidates:
                items.append(_graph_file_item(path, run_dir=run_dir, latest_path=latest))
                if len(items) >= limit:
                    break
            if len(items) >= limit:
                break
    current_path = STATE.get("graph_path")
    if current_path:
        path = Path(current_path)
        if path.exists() and path.is_file() and path.suffix.lower() in {".graphml", ".xml"}:
            seen = {item["absolute_path"] for item in items}
            if str(path) not in seen:
                items.insert(0, _graph_file_item(path, run_dir=path.parent, latest_path=path))
    items.sort(key=lambda item: (item.get("updated_at") or 0, item.get("iter") or -1), reverse=True)
    return {"graphs": items[:limit]}


def _read_growth(run_dir, limit=80):
    path = run_dir / "growth.csv"
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        with path.open(newline="", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    out = []
    for row in rows[-limit:]:
        out.append({
            "iter": _safe_int(row.get("iter")),
            "depth": _safe_int(row.get("depth")),
            "nodes": _safe_int(row.get("n_nodes")),
            "edges": _safe_int(row.get("n_edges")),
            "new_nodes": _safe_int(row.get("new_nodes")),
            "tokens": _safe_int(row.get("tokens")),
            "cum_tokens": _safe_int(row.get("cum_tokens")),
            "diversity": _safe_float(row.get("diversity")),
        })
    return out


def _progress_payload(run_dir, *, budget_calls=None, max_iters=None, max_tokens=None):
    rows = _read_growth(run_dir)
    last = rows[-1] if rows else {}
    calls = len(rows)
    current_iter = last.get("iter", -1)
    total_calls = _safe_int(budget_calls, 0)
    total_iters = _safe_int(max_iters, 0)
    token_budget = _safe_int(max_tokens, 0)
    ratios = []
    if total_calls > 0:
        ratios.append(calls / total_calls)
    if total_iters > 0 and current_iter >= 0:
        ratios.append((current_iter + 1) / total_iters)
    if token_budget > 0:
        ratios.append(_safe_int(last.get("cum_tokens"), 0) / token_budget)
    percent = max(ratios) if ratios else (1.0 if rows else 0.0)
    return {
        "percent": max(0.0, min(1.0, percent)),
        "calls": calls,
        "total_calls": total_calls,
        "iter": current_iter,
        "total_iters": total_iters,
        "nodes": _safe_int(last.get("nodes"), 0),
        "edges": _safe_int(last.get("edges"), 0),
        "new_nodes": _safe_int(last.get("new_nodes"), 0),
        "tokens": _safe_int(last.get("tokens"), 0),
        "cum_tokens": _safe_int(last.get("cum_tokens"), 0),
        "max_tokens": token_budget,
        "diversity": _safe_float(last.get("diversity"), 0.0),
        "growth_tail": rows,
    }


def _load_run_graph(run_value):
    target = _resolve_run_path(run_value)
    if not target.exists():
        raise ValueError(f"run directory not found: {target} ({_available_runs_message()})")
    if target.is_file():
        candidates = [target]
        run_dir = target.parent.parent if target.parent.name == "graphml" else target.parent
    elif target.is_dir():
        run_dir = target
        candidates = _graphml_candidates(run_dir)
        if not candidates:
            raise ValueError(
                f"no graphml found in {run_dir}; expected graph.graphml or graphml/iter_*.graphml "
                f"({_available_runs_message()})"
            )
    else:
        raise ValueError(f"run path is neither a file nor a directory: {target}")

    errors = []
    for graph_path in candidates:
        try:
            return run_dir, graph_path, _normalize_graph(nx.read_graphml(graph_path))
        except Exception as exc:
            errors.append(f"{graph_path}: {exc}")
    detail = errors[-1] if errors else "no candidates"
    raise ValueError(f"no readable graphml in {target}; last error: {detail}")


def _load_config():
    path = IDEATION_DIR / "config.yaml"
    if not path.exists():
        return {}, path
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read config.yaml") from exc
    with open(path) as f:
        return yaml.safe_load(f) or {}, path


def _normalize_generation_backend(value, provider="openai"):
    provider = str(provider or "openai").strip().lower()
    if provider == "hf":
        return "hf"
    backend = str(value or "responses").strip().lower().replace("_", "-")
    if backend in {"chat", "chat-completions", "chat.completions"}:
        return "chat"
    if backend in {"prompt", "completion", "completions", "text"}:
        return "prompt"
    return "responses"


def _role_from_config(role, cfg):
    if role == "embedder":
        return {
            "role": role,
            "provider": "embedding",
            "model": str(cfg.get("embed_model") or ""),
            "base_url": "",
            "api_key_env": "",
            "temperature": "",
            "max_tokens": "",
            "reasoning_effort": "",
        }
    data = dict(cfg.get(role) or {})
    if role == "graph_qa" and not data.get("model"):
        data = dict(cfg.get("questioner") or {})
    if role == "chat" and not data.get("model"):
        data = dict(cfg.get("questioner") or {})
    provider = data.get("provider") or "openai"
    backend = _normalize_generation_backend(data.get("backend"), provider)
    return {
        "role": role,
        "provider": provider,
        "model": str(data.get("model") or ""),
        "base_url": str(data.get("base_url") or ""),
        "backend": backend,
        "api_key_env": str(data.get("api_key_env") or data.get("api_key") or ""),
        "temperature": data.get("temperature", ""),
        "max_tokens": data.get("max_tokens", ""),
        "reasoning_effort": str(data.get("reasoning_effort") or ""),
    }


def _config_payload():
    cfg, path = _load_config()
    roles = ["chat", "generator", "questioner", "graph_qa", "judge", "baseline", "embedder"]
    return {
        "path": str(path),
        "exists": path.exists(),
        "roles": {role: _role_from_config(role, cfg) for role in roles},
    }


def _coerce_role(raw):
    role = dict(raw or {})
    for key in ("role", "provider", "model", "base_url", "backend", "api_key_env", "reasoning_effort"):
        role[key] = str(role.get(key) or "").strip()
    provider = role.get("provider") or "openai"
    role["backend"] = _normalize_generation_backend(role.get("backend"), provider)
    for key in ("temperature", "max_tokens"):
        value = role.get(key)
        if value in (None, ""):
            role[key] = ""
            continue
        try:
            role[key] = float(value) if key == "temperature" else int(float(value))
        except Exception:
            role[key] = value
    return role


def _roles_to_config_text(roles):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to export config.yaml") from exc
    cfg, _ = _load_config()
    cfg = dict(cfg)
    for name, raw in dict(roles or {}).items():
        role = _coerce_role(raw)
        if name == "embedder":
            if role.get("model"):
                cfg["embed_model"] = role["model"]
            continue
        block = dict(cfg.get(name) or {})
        for key in ("provider", "model", "base_url", "backend", "api_key_env", "temperature", "max_tokens", "reasoning_effort"):
            value = role.get(key)
            if value not in (None, ""):
                block[key] = value
            elif key in block:
                block.pop(key, None)
        cfg[name] = block
    return yaml.safe_dump(cfg, sort_keys=False, width=100)


def _model_status(body):
    role = _coerce_role(body.get("role") or body)
    provider = role.get("provider") or "openai"
    if provider in ("hf", "embedding"):
        return {
            "ok": True,
            "provider": provider,
            "message": "No HTTP server check is needed for this provider.",
            "models": [role.get("model")] if role.get("model") else [],
        }
    base_url = (role.get("base_url") or "").rstrip("/")
    if not base_url:
        return {"ok": False, "message": "base_url is required for OpenAI-compatible health checks.", "models": []}
    url = f"{base_url}/models"
    headers = {"Accept": "application/json"}
    api_key_env = role.get("api_key_env")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=float(body.get("timeout", 2.0))) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "{}")
        models = []
        for item in payload.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
        return {"ok": True, "url": url, "status": 200, "models": models[:80], "message": f"{len(models)} models"}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "url": url, "status": exc.code, "models": [], "message": str(exc)}
    except Exception as exc:
        return {"ok": False, "url": url, "models": [], "message": str(exc)}


def _is_local_base_url(base_url):
    if not base_url:
        return False
    host = urlparse(base_url).hostname or ""
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _model_probe(body):
    role = _coerce_role(body.get("role") or body)
    provider = role.get("provider") or "openai"
    model = role.get("model") or ""
    base_url = (role.get("base_url") or "").rstrip("/")
    api_key_env = role.get("api_key_env") or ("OPENAI_API_KEY" if not base_url or "api.openai.com" in base_url else "")
    local = _is_local_base_url(base_url)

    if not model:
        return {
            "ok": False,
            "category": "missing_model",
            "stage": "config",
            "message": "Model id is required.",
            "model": model,
            "base_url": base_url,
            "local": local,
        }
    if provider in ("hf", "embedding"):
        return {
            "ok": True,
            "category": "not_http",
            "stage": "config",
            "message": f"{provider} roles do not use an OpenAI-compatible HTTP probe.",
            "model": model,
            "base_url": base_url,
            "local": local,
        }
    if not base_url:
        if api_key_env and not os.environ.get(api_key_env):
            return {
                "ok": False,
                "category": "api_key",
                "stage": "config",
                "message": f"Set {api_key_env} before using {model}.",
                "model": model,
                "base_url": base_url,
                "api_key_env": api_key_env,
                "local": False,
            }
    elif not local and "api.openai.com" in base_url and api_key_env and not os.environ.get(api_key_env):
        return {
            "ok": False,
            "category": "api_key",
            "stage": "config",
            "message": f"Set {api_key_env} before using {model}.",
            "model": model,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "local": False,
        }

    status = _model_status({"role": role, "timeout": body.get("timeout", 2.0)})
    models = status.get("models") or []
    if not status.get("ok"):
        message = status.get("message") or "model endpoint is unavailable"
        category = "connection" if local else ("api_key" if status.get("status") in (401, 403) else "status_error")
        return {
            "ok": False,
            "category": category,
            "stage": "models",
            "message": message,
            "model": model,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "local": local,
            "url": status.get("url"),
            "models": models,
        }
    if models and model not in models:
        return {
            "ok": False,
            "category": "model_missing",
            "stage": "models",
            "message": f"{model} was not listed by the server.",
            "model": model,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "local": local,
            "url": status.get("url"),
            "models": models[:40],
        }

    try:
        cfg = dict(role)
        cfg["max_tokens"] = 8
        cfg["temperature"] = 0
        answer = _call_openai_compatible(
            cfg,
            [
                {"role": "system", "content": "You are a health check."},
                {"role": "user", "content": "Reply exactly with: ok"},
            ],
        )
        return {
            "ok": True,
            "category": "ok",
            "stage": "completion",
            "message": "Completion probe succeeded.",
            "model": model,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "local": local,
            "models": models[:40],
            "sample": answer[:80],
        }
    except Exception as exc:
        msg = str(exc)
        lower = msg.lower()
        if "api key" in lower or "unauthorized" in lower or "authentication" in lower or "401" in lower:
            category = "api_key"
        elif local and any(text in lower for text in ("connection refused", "failed to establish", "no route", "connection error")):
            category = "connection"
        elif "model" in lower and any(text in lower for text in ("not found", "does not exist", "not served", "unknown")):
            category = "model_missing"
        else:
            category = "completion_error"
        return {
            "ok": False,
            "category": category,
            "stage": "completion",
            "message": msg,
            "model": model,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "local": local,
            "models": models[:40],
        }


def _save_config(body):
    text = _roles_to_config_text(body.get("roles") or {})
    path = IDEATION_DIR / "config.yaml"
    path.write_text(text)
    return {"ok": True, "path": str(path), "config": text}


def _edge_key(u, v, directed=False):
    su, sv = str(u), str(v)
    return (su, sv) if directed or su <= sv else (sv, su)


def _compare_run(body):
    G = _require_graph()
    run, graph_path, H = _load_run_graph(body.get("run") or "")
    g_nodes = {str(n) for n in G.nodes}
    h_nodes = {str(n) for n in H.nodes}
    g_edges = {_edge_key(u, v, G.is_directed()) for u, v in G.edges}
    h_edges = {_edge_key(u, v, H.is_directed()) for u, v in H.edges}
    h_degree = dict(H.to_undirected().degree())
    g_degree = dict(G.to_undirected().degree())

    def top(nodes, degree, graph, limit=20):
        rows = []
        for n in sorted(nodes, key=lambda x: degree.get(x, 0), reverse=True)[:limit]:
            label = graph.nodes[n].get("label", n) if n in graph else n
            rows.append({"id": str(n), "label": str(label), "degree": int(degree.get(n, 0))})
        return rows

    return {
        "run": run.name,
        "path": str(graph_path),
        "current": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "name": STATE.get("graph_name", "")},
        "other": {"nodes": H.number_of_nodes(), "edges": H.number_of_edges(), "name": run.name},
        "shared": {"nodes": len(g_nodes & h_nodes), "edges": len(g_edges & h_edges)},
        "current_only": {
            "nodes": len(g_nodes - h_nodes),
            "edges": len(g_edges - h_edges),
            "top_nodes": top(g_nodes - h_nodes, g_degree, G),
        },
        "other_only": {
            "nodes": len(h_nodes - g_nodes),
            "edges": len(h_edges - g_edges),
            "top_nodes": top(h_nodes - g_nodes, h_degree, H),
        },
    }


def _set_graph(G, *, name, path="", topic=""):
    with LOCK:
        STATE["graph"] = G
        STATE["graph_id"] = uuid.uuid4().hex[:12]
        STATE["graph_name"] = name
        STATE["graph_path"] = path
        STATE["topic"] = topic
        return graph_payload(G, name=name, path=path, topic=topic, include_attrs=True)


def _clear_graph():
    with LOCK:
        STATE["graph"] = None
        STATE["graph_id"] = None
        STATE["graph_name"] = ""
        STATE["graph_path"] = ""
        STATE["topic"] = ""
        STATE["embedding_index"] = None
        STATE["embedding_job"] = None
    return {"ok": True}


def _require_graph():
    with LOCK:
        G = STATE.get("graph")
        if G is None:
            raise ValueError("No graph is loaded yet.")
        return G


def _resolve_embedding_model():
    if str(IDEATION_DIR) not in sys.path:
        sys.path.insert(0, str(IDEATION_DIR))
    try:
        from graphstore import DEFAULT_EMBED_MODEL, resolve_embed_model
    except Exception:
        DEFAULT_EMBED_MODEL = "google/embeddinggemma-300m"

        def resolve_embed_model(run_dir=None, cli=None, default=DEFAULT_EMBED_MODEL):
            return cli or default

    cfg, _ = _load_config()
    configured = cfg.get("embed_model")
    graph_path = str(STATE.get("graph_path") or "")
    run_dir = ""
    if graph_path:
        path = Path(graph_path)
        if path.name.endswith(".graphml"):
            run_dir = str(path.parent)
    return resolve_embed_model(run_dir or None, configured, DEFAULT_EMBED_MODEL)


def _node_embedding_text(node_id, attrs):
    attrs = dict(attrs or {})
    label = str(attrs.get("label") or node_id)
    parts = [label]
    for key in ("question", "answer", "topic", "relation", "source", "response_id"):
        value = attrs.get(key)
        if value not in (None, ""):
            parts.append(f"{key}: {value}")
    text = " | ".join(str(part) for part in parts)
    return text[:1200]


def _embedding_status_locked():
    job = STATE.get("embedding_job")
    index = STATE.get("embedding_index")
    if job:
        public = {k: v for k, v in dict(job).items() if k not in ("thread",)}
    else:
        public = {
            "id": "",
            "graph_id": STATE.get("graph_id") or "",
            "model": "",
            "status": "idle",
            "started_at": None,
            "ended_at": None,
            "error": "",
            "progress": {"percent": 0.0, "current": 0, "total": 0, "message": "Idle", "detail": ""},
        }
    if index and index.get("graph_id") == STATE.get("graph_id"):
        public.update({
            "ready": True,
            "nodes": len(index.get("ids") or []),
            "dimension": int(index.get("dimension") or 0),
            "model": index.get("model") or public.get("model") or "",
        })
    else:
        public.setdefault("ready", False)
        public.setdefault("nodes", 0)
        public.setdefault("dimension", 0)
    return public


def _embedding_status():
    with LOCK:
        return _embedding_status_locked()


def _embedding_model(model_name):
    from sentence_transformers import SentenceTransformer

    with LOCK:
        model = STATE["embedding_models"].get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        with LOCK:
            STATE["embedding_models"][model_name] = model
    return model


def _encode_embedding_texts(texts, model_name, *, batch_size=64):
    import numpy as np

    model = _embedding_model(model_name)
    prompt_name = "STS" if "embeddinggemma" in model_name.lower() else None
    prompts = getattr(model, "prompts", None) or {}
    use_prompt = prompt_name if prompt_name in prompts else None
    kw = {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "batch_size": batch_size,
        "show_progress_bar": False,
    }
    if use_prompt:
        kw["prompt_name"] = use_prompt
    vectors = model.encode(list(texts), **kw)
    return vectors.astype(np.float32)


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_hash_value(h, value):
    h.update(json.dumps(_clean(value), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))


def _graph_content_signature(G, path=""):
    path = Path(path) if path else None
    if path and path.exists() and path.is_file():
        try:
            stat = path.stat()
            return f"file:{_hash_file(path)}:{stat.st_size}"
        except OSError:
            pass

    h = hashlib.sha256()
    h.update(f"directed:{int(G.is_directed())}\n".encode("utf-8"))
    for node_id, attrs in sorted(G.nodes(data=True), key=lambda item: str(item[0])):
        h.update(b"node\0")
        h.update(str(node_id).encode("utf-8", errors="replace"))
        h.update(b"\0")
        _stable_hash_value(h, attrs)
        h.update(b"\n")
    edge_rows = []
    for u, v, attrs in G.edges(data=True):
        if not G.is_directed() and str(v) < str(u):
            u, v = v, u
        edge_rows.append((str(u), str(v), dict(attrs)))
    for u, v, attrs in sorted(edge_rows):
        h.update(b"edge\0")
        h.update(u.encode("utf-8", errors="replace"))
        h.update(b"\0")
        h.update(v.encode("utf-8", errors="replace"))
        h.update(b"\0")
        _stable_hash_value(h, attrs)
        h.update(b"\n")
    return f"graph:{h.hexdigest()}:{G.number_of_nodes()}:{G.number_of_edges()}"


def _embedding_cache_key(signature, model_name):
    raw = f"{EMBEDDING_CACHE_VERSION}\n{model_name}\n{signature}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _embedding_cache_path(cache_key):
    return EMBEDDING_CACHE_DIR / cache_key[:2] / cache_key


def _load_embedding_cache(signature, model_name, graph_id, graph_name):
    cache_key = _embedding_cache_key(signature, model_name)
    cache_path = _embedding_cache_path(cache_key)
    meta_path = cache_path / "meta.json"
    vectors_path = cache_path / "vectors.npy"
    if not meta_path.exists() or not vectors_path.exists():
        return None
    try:
        import numpy as np

        with meta_path.open() as f:
            meta = json.load(f)
        if (
            meta.get("version") != EMBEDDING_CACHE_VERSION
            or meta.get("signature") != signature
            or meta.get("model") != model_name
        ):
            return None
        ids = [str(x) for x in meta.get("ids") or []]
        labels = [str(x) for x in meta.get("labels") or []]
        texts = [str(x) for x in meta.get("texts") or []]
        vectors = np.load(vectors_path, allow_pickle=False)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            return None
    except Exception:
        return None
    return {
        "graph_id": graph_id,
        "graph_name": graph_name,
        "graph_signature": signature,
        "model": model_name,
        "ids": ids,
        "labels": labels,
        "texts": texts,
        "vectors": vectors.astype(np.float32, copy=False),
        "dimension": int(vectors.shape[1]) if vectors.shape[0] else 0,
        "created_at": float(meta.get("created_at") or time.time()),
        "cache_key": cache_key,
        "cache_path": str(cache_path),
        "cached": True,
    }


def _save_embedding_cache(signature, model_name, index):
    import numpy as np

    cache_key = _embedding_cache_key(signature, model_name)
    cache_path = _embedding_cache_path(cache_key)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{cache_key}.", dir=str(cache_path.parent)))
    try:
        vectors = index.get("vectors")
        np.save(tmp / "vectors.npy", vectors)
        meta = {
            "version": EMBEDDING_CACHE_VERSION,
            "signature": signature,
            "model": model_name,
            "ids": index.get("ids") or [],
            "labels": index.get("labels") or [],
            "texts": index.get("texts") or [],
            "dimension": int(index.get("dimension") or 0),
            "created_at": float(index.get("created_at") or time.time()),
        }
        with (tmp / "meta.json").open("w") as f:
            json.dump(meta, f, ensure_ascii=True)
        if cache_path.exists():
            shutil.rmtree(cache_path)
        os.replace(tmp, cache_path)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return cache_key, cache_path


def _start_embedding_index(body=None):
    body = dict(body or {})
    force = bool(body.get("force"))
    requested = str(body.get("model") or "auto").strip()
    model_name = _resolve_embedding_model() if requested in ("", "auto") else requested
    with LOCK:
        G = STATE.get("graph")
        graph_id = STATE.get("graph_id")
        graph_name = STATE.get("graph_name") or ""
        if G is None or not graph_id:
            raise ValueError("Load a graph before building an embedding index.")
        graph_path = STATE.get("graph_path") or ""
        index = STATE.get("embedding_index")
        if (
            not force
            and index
            and index.get("graph_id") == graph_id
            and index.get("model") == model_name
        ):
            return _embedding_status_locked()
        job = STATE.get("embedding_job")
        if (
            not force
            and job
            and job.get("graph_id") == graph_id
            and job.get("model") == model_name
            and job.get("status") == "running"
        ):
            return _embedding_status_locked()
        nodes = [(str(n), dict(d)) for n, d in G.nodes(data=True)]
        signature = _graph_content_signature(G, graph_path)
        if not force:
            cached = _load_embedding_cache(signature, model_name, graph_id, graph_name)
            if cached:
                STATE["embedding_index"] = cached
                STATE["embedding_job"] = {
                    "id": cached.get("cache_key", ""),
                    "graph_id": graph_id,
                    "graph_name": graph_name,
                    "model": model_name,
                    "status": "done",
                    "started_at": time.time(),
                    "ended_at": time.time(),
                    "error": "",
                    "cached": True,
                    "cache_key": cached.get("cache_key", ""),
                    "progress": {
                        "percent": 1.0,
                        "current": len(cached.get("ids") or []),
                        "total": len(cached.get("ids") or []),
                        "message": "Loaded cached embedding index",
                        "detail": f"{len(cached.get('ids') or [])} nodes | {model_name}",
                    },
                }
                return _embedding_status_locked()
        job_id = uuid.uuid4().hex[:10]
        STATE["embedding_index"] = None
        STATE["embedding_job"] = {
            "id": job_id,
            "graph_id": graph_id,
            "graph_name": graph_name,
            "model": model_name,
            "status": "running",
            "started_at": time.time(),
            "ended_at": None,
            "error": "",
            "cached": False,
            "progress": {
                "percent": 0.02,
                "current": 0,
                "total": len(nodes),
                "message": "Preparing node text",
                "detail": graph_name,
            },
        }

    def worker():
        import numpy as np

        try:
            ids = [node_id for node_id, _ in nodes]
            labels = [str(attrs.get("label") or node_id) for node_id, attrs in nodes]
            texts = [_node_embedding_text(node_id, attrs) for node_id, attrs in nodes]
            total = len(texts)
            batches = []
            batch_size = max(16, min(96, _safe_int(body.get("batch_size"), 64)))
            with LOCK:
                if STATE.get("embedding_job", {}).get("id") == job_id:
                    STATE["embedding_job"]["progress"] = {
                        "percent": 0.08,
                        "current": 0,
                        "total": total,
                        "message": "Loading embedding model",
                        "detail": model_name,
                    }
            _embedding_model(model_name)
            for start in range(0, total, batch_size):
                with LOCK:
                    if STATE.get("graph_id") != graph_id or STATE.get("embedding_job", {}).get("id") != job_id:
                        return
                stop = min(total, start + batch_size)
                batch = _encode_embedding_texts(texts[start:stop], model_name, batch_size=batch_size)
                batches.append(batch)
                with LOCK:
                    current = stop
                    if STATE.get("embedding_job", {}).get("id") == job_id:
                        STATE["embedding_job"]["progress"] = {
                            "percent": 0.08 + 0.9 * (current / max(1, total)),
                            "current": current,
                            "total": total,
                            "message": "Embedding graph nodes",
                            "detail": f"{current}/{total} nodes",
                        }
            vectors = np.vstack(batches) if batches else np.zeros((0, 0), dtype=np.float32)
            index = {
                "graph_id": graph_id,
                "graph_name": graph_name,
                "graph_signature": signature,
                "model": model_name,
                "ids": ids,
                "labels": labels,
                "texts": texts,
                "vectors": vectors,
                "dimension": int(vectors.shape[1]) if len(vectors.shape) == 2 and vectors.shape[0] else 0,
                "created_at": time.time(),
                "cached": False,
            }
            try:
                cache_key, cache_path = _save_embedding_cache(signature, model_name, index)
                index["cache_key"] = cache_key
                index["cache_path"] = str(cache_path)
            except Exception as cache_exc:
                index["cache_error"] = str(cache_exc)
            with LOCK:
                if STATE.get("graph_id") == graph_id and STATE.get("embedding_job", {}).get("id") == job_id:
                    STATE["embedding_index"] = index
                    STATE["embedding_job"]["status"] = "done"
                    STATE["embedding_job"]["ended_at"] = time.time()
                    STATE["embedding_job"]["cached"] = False
                    STATE["embedding_job"]["cache_key"] = index.get("cache_key", "")
                    STATE["embedding_job"]["progress"] = {
                        "percent": 1.0,
                        "current": total,
                        "total": total,
                        "message": "Embedding index ready",
                        "detail": f"{total} nodes | {model_name}",
                    }
        except Exception as exc:
            with LOCK:
                if STATE.get("embedding_job", {}).get("id") == job_id:
                    STATE["embedding_job"]["status"] = "failed"
                    STATE["embedding_job"]["ended_at"] = time.time()
                    STATE["embedding_job"]["error"] = str(exc)
                    progress = dict(STATE["embedding_job"].get("progress") or {})
                    progress.update({"message": "Embedding failed", "detail": str(exc)})
                    STATE["embedding_job"]["progress"] = progress

    thread = threading.Thread(target=worker, daemon=True)
    with LOCK:
        if STATE.get("embedding_job", {}).get("id") == job_id:
            STATE["embedding_job"]["thread"] = thread
    thread.start()
    return _embedding_status()


def _semantic_search_nodes(query, limit=8):
    q = str(query or "").strip()
    if not q:
        return []
    with LOCK:
        index = STATE.get("embedding_index")
        graph_id = STATE.get("graph_id")
        if not index or index.get("graph_id") != graph_id:
            return []
        model_name = index.get("model")
        ids = list(index.get("ids") or [])
        labels = list(index.get("labels") or [])
        vectors = index.get("vectors")
    if vectors is None or not ids or not model_name:
        return []
    try:
        query_vec = _encode_embedding_texts([q], model_name, batch_size=1)[0]
        sims = vectors @ query_vec
        order = sims.argsort()[::-1][:max(1, limit)]
    except Exception:
        return []
    G = _require_graph()
    metrics = _centrality(G)
    degree = metrics["degree"]
    pr = metrics["pagerank"]
    core = metrics["core"]
    out = []
    for idx in order:
        node_id = ids[int(idx)]
        if node_id not in G:
            continue
        d = G.nodes[node_id]
        sim = float(sims[int(idx)])
        out.append({
            "id": str(node_id),
            "label": labels[int(idx)] if int(idx) < len(labels) else str(d.get("label") or node_id),
            "score": float(100.0 + sim * 25.0 + math.log1p(degree.get(node_id, 0))),
            "degree": int(degree.get(node_id, 0)),
            "pagerank": float(pr.get(node_id, 0.0)),
            "core": int(core.get(node_id, 0)),
            "iter": _int_attr(d, "iter", 0),
            "semantic_score": sim,
        })
    return out


def _search_nodes(G, query, limit=50):
    q = " ".join(str(query or "").lower().split())
    if not q:
        return []
    terms = [t for t in q.split() if t]
    metrics = _centrality(G)
    degree = metrics["degree"]
    pr = metrics["pagerank"]
    core = metrics["core"]
    out = []
    by_id = {}
    for n, d in G.nodes(data=True):
        attrs = {str(k): str(v) for k, v in dict(d).items()}
        label = str(attrs.get("label") or n)
        hay = " ".join([label, str(n)] + [f"{k} {v}" for k, v in attrs.items()]).lower()
        hits = sum(1 for t in terms if t in hay)
        if q in hay:
            hits += 2
        if hits:
            item = {
                "id": str(n),
                "label": label,
                "score": float(hits + math.log1p(degree.get(n, 0)) + pr.get(n, 0.0) * 10.0),
                "degree": int(degree.get(n, 0)),
                "pagerank": float(pr.get(n, 0.0)),
                "core": int(core.get(n, 0)),
                "iter": _int_attr(d, "iter", 0),
            }
            out.append(item)
            by_id[item["id"]] = item
    for item in _semantic_search_nodes(query, limit=max(12, limit)):
        existing = by_id.get(item["id"])
        if existing:
            existing["score"] = max(float(existing.get("score", 0.0)), float(item.get("score", 0.0)))
            existing["semantic_score"] = item.get("semantic_score")
        else:
            out.append(item)
            by_id[item["id"]] = item
    out.sort(key=lambda x: (x["score"], x["degree"]), reverse=True)
    return out[:limit]


def _neighborhood_nodes(G, seeds, depth=1, limit=400):
    if not seeds:
        return set()
    U = G.to_undirected()
    seen = set()
    frontier = {s for s in seeds if s in U}
    for _ in range(max(0, int(depth)) + 1):
        if not frontier:
            break
        seen.update(frontier)
        if len(seen) >= limit:
            break
        nxt = set()
        for n in frontier:
            nxt.update(U.neighbors(n))
        frontier = nxt - seen
    if len(seen) > limit:
        degree = dict(U.degree(seen))
        keep = set(seeds)
        rest = sorted((n for n in seen if n not in keep), key=lambda n: degree.get(n, 0), reverse=True)
        seen = set(list(keep) + rest[:max(0, limit - len(keep))])
    return seen


def _subgraph_payload(nodes, *, name=None):
    G = _require_graph()
    H = G.subgraph(nodes).copy()
    return graph_payload(
        H,
        name=name or STATE.get("graph_name", ""),
        path=STATE.get("graph_path", ""),
        topic=STATE.get("topic", ""),
        include_attrs=True,
    )


def _path_payload(source, target, k=5, cutoff=6):
    G = _require_graph()
    U = G.to_undirected()
    resolved = _resolve_concept_nodes(G, [source, target], limit=2)
    if len(resolved) >= 2:
        source, target = resolved[:2]
    if source not in U or target not in U:
        raise ValueError("Source or target is not in the graph.")
    paths = []
    try:
        for p in nx.shortest_simple_paths(U, source, target):
            if len(p) - 1 <= cutoff:
                paths.append([str(x) for x in p])
            if len(paths) >= k:
                break
            if len(p) - 1 > cutoff and paths:
                break
    except nx.NetworkXNoPath:
        paths = []
    nodes = set(x for p in paths for x in p)
    payload = _subgraph_payload(nodes, name=f"path {source} to {target}") if nodes else _subgraph_payload([], name="empty path")
    payload["paths"] = paths
    payload["resolved_source"] = str(source)
    payload["resolved_target"] = str(target)
    return payload


def _resolve_concept_nodes(G, concepts, *, limit=8):
    found = []
    seen = set()
    labels = {str(d.get("label") or n).lower(): str(n) for n, d in G.nodes(data=True)}
    for raw in concepts:
        query = str(raw or "").strip()
        if not query:
            continue
        node_id = query if query in G else labels.get(query.lower())
        if not node_id:
            hits = _search_nodes(G, query, limit=1)
            node_id = hits[0]["id"] if hits else None
        if node_id and node_id not in seen:
            seen.add(node_id)
            found.append(str(node_id))
        if len(found) >= limit:
            break
    return found


def _multipath_payload(body):
    G = _require_graph()
    raw = body.get("nodes") or []
    raw_nodes = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    raw_query = str(body.get("query") or "")
    if raw_query:
        raw_nodes.extend([part.strip() for part in re.split(r"[,;\n]+", raw_query) if part.strip()])
    anchors = _resolve_concept_nodes(G, raw_nodes, limit=_safe_int(body.get("anchor_limit"), 8))
    if len(anchors) < 2:
        raise ValueError("At least two resolvable concepts are required.")

    U = G.to_undirected()
    degree = dict(U.degree())
    metrics = _centrality(G)
    cutoff = _safe_int(body.get("cutoff"), 8)
    mode = str(body.get("mode") or "pairwise")
    pairs = []
    if mode == "sequence":
        pairs = list(zip(anchors, anchors[1:]))
    elif mode == "stochastic":
        base_pairs = [(a, b) for i, a in enumerate(anchors) for b in anchors[i + 1:] if a != b]
        rng = random.Random(str(body.get("seed") or "|".join(anchors)))
        sample_count = max(1, min(80, _safe_int(body.get("sample_count"), 28)))
        pairs = [rng.choice(base_pairs) for _ in range(sample_count)] if base_pairs else []
    else:
        for i, a in enumerate(anchors):
            for b in anchors[i + 1:]:
                pairs.append((a, b))

    paths = []
    seen_paths = set()
    pair_limit = max(1, min(80, _safe_int(body.get("sample_count"), 28))) if mode == "stochastic" else 24
    for a, b in pairs[:pair_limit]:
        if a not in U or b not in U:
            continue
        try:
            if mode == "stochastic":
                edge_weights = {}
                rng = random.Random(f"{a}|{b}|{len(paths)}|{body.get('seed') or ''}")
                for u, v in U.edges():
                    key = tuple(sorted((str(u), str(v))))
                    # Jitter plus a mild hub penalty gives alternative plausible routes without
                    # ignoring graph distance.
                    edge_weights[key] = 1.0 + rng.random() * 1.75 + math.log1p(max(degree.get(u, 0), degree.get(v, 0))) * 0.025

                def weight(u, v, _attrs):
                    return edge_weights.get(tuple(sorted((str(u), str(v)))), 1.0)

                p = nx.shortest_path(U, a, b, weight=weight)
            else:
                p = nx.shortest_path(U, a, b)
        except nx.NetworkXNoPath:
            continue
        if cutoff <= 0 or len(p) - 1 <= cutoff:
            key = tuple(str(x) for x in p)
            if key not in seen_paths:
                seen_paths.add(key)
                paths.append(list(key))

    path_nodes = set(x for p in paths for x in p)
    if not path_nodes:
        payload = _subgraph_payload([], name="empty bridge network")
        payload.update({"anchors": anchors, "paths": [], "connectors": []})
        return payload

    connector_counts = {}
    for p in paths:
        for n in p[1:-1]:
            if n not in anchors:
                connector_counts[n] = connector_counts.get(n, 0) + 1
    connectors = []
    for n, count in sorted(connector_counts.items(), key=lambda item: (item[1], degree.get(item[0], 0)), reverse=True)[:18]:
        d = G.nodes[n]
        connectors.append({
            "id": str(n),
            "label": str(d.get("label") or n),
            "count": int(count),
            "degree": int(degree.get(n, 0)),
            "pagerank": float(metrics.get("pagerank", {}).get(n, 0.0)),
            "core": int(metrics.get("core", {}).get(n, 0)),
        })

    payload = _subgraph_payload(path_nodes, name="bridge network")
    payload["anchors"] = anchors
    payload["paths"] = paths
    payload["connectors"] = connectors
    return payload


def _bridge_suggestions_payload(body):
    G = _require_graph()
    U = G.to_undirected()
    degree = dict(U.degree())
    metrics = _centrality(G)
    raw_selected = body.get("selected_nodes") or []
    if isinstance(raw_selected, str):
        raw_selected = [raw_selected]
    selected = [str(n) for n in raw_selected if n in G]

    def score(n):
        return float(metrics.get("pagerank", {}).get(n, 0.0)) * 100000.0 + float(degree.get(n, 0))

    def label(n):
        d = G.nodes[n]
        return str(d.get("label") or n)

    ranked = [n for n in sorted(U.nodes, key=score, reverse=True) if degree.get(n, 0) > 0]
    anchors = []
    seen_groups = set()
    for n in selected + ranked:
        group = (G.nodes[n].get("component", 0), G.nodes[n].get("community", 0))
        if selected and n in selected:
            anchors.append(n)
        elif group not in seen_groups:
            anchors.append(n)
            seen_groups.add(group)
        if len(anchors) >= 10:
            break
    if len(anchors) < 4:
        anchors = list(dict.fromkeys(anchors + ranked[:10]))

    ideas = []
    seen_pairs = set()
    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            try:
                path = nx.shortest_path(U, a, b)
            except nx.NetworkXNoPath:
                continue
            hops = len(path) - 1
            if hops < 2 or hops > 8:
                continue
            interior = [n for n in path[1:-1] if n in G]
            connectors = sorted(interior, key=score, reverse=True)[:3]
            connector_text = ", ".join(label(n) for n in connectors) or "direct neighborhood"
            ideas.append({
                "title": f"{label(a)} to {label(b)}",
                "concepts": [label(a), label(b)],
                "query": f"{label(a)}, {label(b)}",
                "rationale": f"{hops} hops; likely connector: {connector_text}",
                "connectors": [label(n) for n in connectors],
            })
            if len(ideas) >= _safe_int(body.get("limit"), 5):
                return {"ideas": ideas}

    if not ideas:
        for a, b in zip(ranked[::2], ranked[1::2]):
            ideas.append({
                "title": f"{label(a)} to {label(b)}",
                "concepts": [label(a), label(b)],
                "query": f"{label(a)}, {label(b)}",
                "rationale": "High-centrality seed pair; run Find Bridges to test whether a route exists.",
                "connectors": [],
            })
            if len(ideas) >= _safe_int(body.get("limit"), 5):
                break
    return {"ideas": ideas}


def _format_node(G, n):
    d = dict(G.nodes[n])
    label = str(d.get("label") or n)
    bits = [f"id={n}", f"label={label}"]
    for key in ("iter", "depth", "response_id", "question"):
        if key in d:
            bits.append(f"{key}={d.get(key)}")
    return "; ".join(bits)


def _context_node_payload(G, nodes, scores=None, limit=80):
    metrics = _centrality(G)
    degree = metrics["degree"]
    pr = metrics["pagerank"]
    core = metrics["core"]
    scores = scores or {}
    out = []
    for n in nodes[:limit]:
        if n not in G:
            continue
        d = dict(G.nodes[n])
        out.append({
            "id": str(n),
            "label": str(d.get("label") or n),
            "degree": int(degree.get(n, 0)),
            "pagerank": float(pr.get(n, 0.0)),
            "core": int(core.get(n, 0)),
            "iter": _int_attr(d, "iter", 0),
            "score": float(scores.get(n, 0.0)),
        })
    return out


def _add_score(scores, sources, node, amount, source):
    node = str(node)
    scores[node] = scores.get(node, 0.0) + float(amount)
    if source:
        sources.setdefault(node, set()).add(source)


def _graph_rag_ranked_nodes(G, question, selected_nodes=None, query="", max_nodes=240):
    selected_nodes = [str(n) for n in (selected_nodes or []) if str(n) in G]
    retrieval_query = " ".join(x for x in [str(query or "").strip(), str(question or "").strip()] if x)
    metrics = _centrality(G)
    degree = metrics["degree"]
    pr = metrics["pagerank"]
    core = metrics["core"]
    scores = {}
    sources = {}

    for n in selected_nodes:
        _add_score(scores, sources, n, 500.0, "selected")

    if retrieval_query:
        for rank, item in enumerate(_search_nodes(G, retrieval_query, limit=max(60, min(180, max_nodes)))):
            node = str(item["id"])
            _add_score(scores, sources, node, max(20.0, float(item.get("score", 0.0))) + max(0, 60 - rank) * 0.35, "semantic/text")

    central = sorted(G.nodes, key=lambda n: (pr.get(n, 0.0), degree.get(n, 0), core.get(n, 0)), reverse=True)
    for rank, n in enumerate(central[: max(16, min(80, max_nodes // 4))]):
        _add_score(scores, sources, n, 12.0 + max(0, 40 - rank) * 0.15, "centrality")

    seed_nodes = [n for n, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:24] if n in G]
    if selected_nodes:
        seed_nodes = list(dict.fromkeys(selected_nodes + seed_nodes))

    if seed_nodes:
        neighborhood = _neighborhood_nodes(G, seed_nodes[:16], depth=2, limit=max_nodes * 3)
        for n in neighborhood:
            _add_score(scores, sources, n, 4.0 + math.log1p(degree.get(n, 0)), "neighborhood")

    path_rows = []
    if len(seed_nodes) >= 2:
        U = G.to_undirected()
        path_pairs = []
        if len(selected_nodes) >= 2:
            path_pairs.extend(zip(selected_nodes, selected_nodes[1:]))
        path_pairs.extend(zip(seed_nodes[:8:2], seed_nodes[1:8:2]))
        seen_pairs = set()
        for a, b in path_pairs[:8]:
            key = tuple(sorted((str(a), str(b))))
            if key in seen_pairs or a not in U or b not in U:
                continue
            seen_pairs.add(key)
            try:
                path = nx.shortest_path(U, a, b)
            except Exception:
                continue
            if len(path) > 14:
                continue
            path_rows.append([str(x) for x in path])
            for i, n in enumerate(path):
                _add_score(scores, sources, n, 35.0 if 0 < i < len(path) - 1 else 12.0, "path")

    ranked = [n for n, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True) if n in G]
    if not ranked:
        ranked = central[:max_nodes]
    return ranked[:max_nodes], scores, sources, path_rows, retrieval_query


def _normalize_context_mode(value):
    mode = str(value or "").strip().lower().replace("-", "_")
    if mode in {"graph_rag", "rag", "graphrag"}:
        return "graph_rag"
    if mode in {"focused", "focus"}:
        return "focused"
    if mode in {"none", "off", "regular", "chat", "selection", "selected"}:
        return "none"
    return "none"


def _empty_graph_context(question, selected_nodes=None, mode="none"):
    selected_nodes = [str(n) for n in (selected_nodes or [])]
    return {
        "mode": mode,
        "selected": selected_nodes,
        "query": "",
        "node_count": 0,
        "edge_count": 0,
        "node_ids": [],
        "nodes": [],
        "text": "",
    }


def _context_for_llm(G, question, selected_nodes=None, query="", depth=1, max_nodes=90, max_edges=160, context_mode="none"):
    selected_nodes = [str(n) for n in (selected_nodes or []) if str(n) in G]
    context_mode = _normalize_context_mode(context_mode)
    source_map = {}
    path_rows = []
    retrieval_query = query
    scores = {}
    if context_mode == "graph_rag":
        max_nodes = min(max(_safe_int(max_nodes, 240), 120), 900)
        max_edges = min(max(_safe_int(max_edges, 420), 220), 1400)
        ranked_nodes, scores, source_map, path_rows, retrieval_query = _graph_rag_ranked_nodes(
            G,
            question,
            selected_nodes=selected_nodes,
            query=query,
            max_nodes=max_nodes,
        )
        nodes = set(ranked_nodes)
        ranked_seed = ranked_nodes
    elif context_mode == "none":
        nodes = set(selected_nodes)
        ranked_seed = []
        retrieval_query = ""
        max_edges = min(max(_safe_int(max_edges), 0), 80)
    else:
        seeds = list(selected_nodes)
        if query:
            seeds.extend([r["id"] for r in _search_nodes(G, query, limit=12)])
        if seeds:
            nodes = _neighborhood_nodes(G, seeds, depth=depth, limit=max_nodes)
        else:
            degree = dict(G.to_undirected().degree())
            nodes = set(n for n, _ in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:max_nodes])
        ranked_seed = []

    H = G.subgraph(nodes).copy()
    degree = dict(G.to_undirected().degree())
    if context_mode == "graph_rag" and ranked_seed:
        ranked_nodes = [n for n in ranked_seed if n in H][:max_nodes]
    else:
        ranked_nodes = sorted(H.nodes, key=lambda n: degree.get(n, 0), reverse=True)[:max_nodes]
    edge_rows = []
    if context_mode != "none" or selected_nodes:
        for u, v, d in H.edges(data=True):
            edge_rows.append((degree.get(u, 0) + degree.get(v, 0), str(u), str(v), str(d.get("relation", "related_to"))))
        edge_rows.sort(reverse=True)
        edge_rows = edge_rows[:max_edges]

    shortest_paths = []
    if context_mode != "none" and len(selected_nodes) >= 2:
        U = G.to_undirected()
        for a, b in zip(selected_nodes, selected_nodes[1:]):
            try:
                p = nx.shortest_path(U, a, b)
                if len(p) <= 10:
                    shortest_paths.append(" -> ".join(str(x) for x in p))
            except Exception:
                pass
    for p in path_rows:
        text = " -> ".join(str(x) for x in p)
        if text not in shortest_paths:
            shortest_paths.append(text)

    if context_mode == "graph_rag":
        node_lines = []
        for n in ranked_nodes:
            sources = ",".join(sorted(source_map.get(str(n), []))) or "ranked"
            node_lines.append(f"- {_format_node(G, n)}; degree={degree.get(n, 0)}; retrieval={sources}; score={scores.get(str(n), 0):.2f}")
        node_text = "\n".join(node_lines)
    else:
        node_text = "\n".join(f"- {_format_node(G, n)}; degree={degree.get(n, 0)}" for n in ranked_nodes)
    edge_text = "\n".join(f"- {u} -[{rel}]- {v}" for _, u, v, rel in edge_rows)
    path_text = "\n".join(f"- {p}" for p in shortest_paths) if shortest_paths else "(none requested or no short path found)"
    node_heading = "Selected nodes" if context_mode == "none" else "Key nodes in focus context"
    edge_heading = "Edges among selected nodes" if context_mode == "none" else "Edges in focus context"
    return {
        "mode": context_mode,
        "selected": selected_nodes,
        "query": retrieval_query,
        "node_count": len(ranked_nodes),
        "edge_count": len(edge_rows),
        "node_ids": [str(n) for n in ranked_nodes],
        "nodes": _context_node_payload(G, ranked_nodes, scores=scores, limit=80),
        "text": (
            f"Graph: {STATE.get('graph_name') or '(loaded graph)'}\n"
            f"Topic: {STATE.get('topic') or '(not recorded)'}\n"
            f"Question: {question}\n"
            f"Context mode: {context_mode}\n"
            f"Selected nodes: {', '.join(selected_nodes) if selected_nodes else '(none)'}\n"
            f"Retrieval query: {retrieval_query or '(none)'}\n\n"
            f"{node_heading}:\n{node_text or '(none)'}\n\n"
            f"{edge_heading}:\n{edge_text or '(none)'}\n\n"
            f"Short selected paths:\n{path_text}\n"
        ),
    }


def _assistant_instruction_role(cfg):
    provider = str((cfg or {}).get("provider") or "openai").strip().lower()
    if provider != "openai":
        return "system"
    model = str((cfg or {}).get("model") or "").strip().lower()
    base_url = str((cfg or {}).get("base_url") or "").strip().lower()
    openai_hosted = not base_url or "api.openai.com" in base_url
    openai_reasoning_model = model.startswith(("gpt-5", "gpt-4.1", "o1", "o3", "o4"))
    return "developer" if openai_hosted or openai_reasoning_model else "system"


def _messages_to_prompt(messages):
    parts = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role in {"system", "developer"}:
            prefix = "Developer" if role == "developer" else "System"
            parts.append(f"{prefix}:\n{content}")
        elif role == "assistant":
            parts.append(f"Assistant:\n{content}")
        else:
            parts.append(f"User:\n{content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _completion_text(response):
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    choice = choices[0]
    text = getattr(choice, "text", None)
    if text:
        return text
    message = getattr(choice, "message", None)
    if message is not None:
        return getattr(message, "content", None) or ""
    if isinstance(choice, dict):
        return choice.get("text") or (choice.get("message") or {}).get("content") or ""
    return ""


def _completion_text_from_payload(payload):
    if not isinstance(payload, dict):
        return ""
    text = payload.get("output_text")
    if text:
        return str(text)
    choices = payload.get("choices") or []
    if choices and isinstance(choices[0], dict):
        choice = choices[0]
        return str(choice.get("text") or (choice.get("message") or {}).get("content") or "")
    output = payload.get("output") or []
    chunks = []
    for item in output:
        for content in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(content, dict) and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()


def _messages_to_responses_parts(messages, cfg=None):
    instruction_role = _assistant_instruction_role(cfg)
    inputs = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "developer":
            inputs.append({"role": "developer", "content": content})
        elif role == "system":
            inputs.append({"role": instruction_role, "content": content})
        else:
            inputs.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    return inputs


def _response_text(response):
    text = getattr(response, "output_text", None)
    if text:
        return text
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(value)
            elif isinstance(content, dict) and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()


def _response_result(response):
    return {
        "text": _response_text(response),
        "response_id": str(getattr(response, "id", "") or ""),
    }


def _response_result_from_payload(payload):
    return {
        "text": _completion_text_from_payload(payload),
        "response_id": str((payload or {}).get("id") or ""),
    }


def _is_no_chat_template_error(message):
    msg = str(message or "").lower()
    return (
        "chat template" in msg
        or "single string as the prompt" in msg
        or ("received messages" in msg and "prompt" in msg)
    )


def _is_previous_response_error(message):
    msg = str(message or "").lower()
    return (
        "previous_response_id" in msg
        or "previous response" in msg
        or ("unknown" in msg and "parameter" in msg)
        or ("unexpected" in msg and "parameter" in msg)
        or ("not found" in msg and "response" in msg)
        or ("expired" in msg and "response" in msg)
    )


def _prompt_completion_payload(cfg, messages):
    payload = {
        "model": cfg["model"],
        "prompt": _messages_to_prompt(messages),
        "temperature": _safe_float(cfg.get("temperature"), 0.3),
        "max_tokens": _chat_max_tokens(cfg),
    }
    return {k: v for k, v in payload.items() if v not in (None, "")}


def _responses_payload(cfg, messages, previous_response_id=None):
    input_items = _messages_to_responses_parts(messages, cfg)
    payload = {
        "model": cfg["model"],
        "input": input_items or _messages_to_prompt(messages),
        "store": True,
        "temperature": _safe_float(cfg.get("temperature"), 0.3),
        "max_output_tokens": _chat_max_tokens(cfg),
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    if cfg.get("reasoning_effort"):
        payload["reasoning"] = {"effort": cfg["reasoning_effort"]}
    return {k: v for k, v in payload.items() if v not in (None, "")}


def _call_responses_http(cfg, messages, previous_response_id=None, previous_error=None, return_metadata=False, fallback_messages=None, payload_override=None):
    base_url = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    api_key_env = cfg.get("api_key_env") or "OPENAI_API_KEY"
    api_key = cfg.get("api_key") or os.environ.get(api_key_env) or "x"
    payload = dict(payload_override or _responses_payload(cfg, messages, previous_response_id=previous_response_id))
    last = previous_error
    for _ in range(6):
        req = urllib.request.Request(
            f"{base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=LLM_HTTP_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8") or "{}")
            result = _response_result_from_payload(data)
            return result if return_metadata else result["text"]
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = str(exc)
            last = RuntimeError(f"HTTP {exc.code} from /responses: {detail}")
            msg = detail.lower()
            if "previous_response_id" in payload and _is_previous_response_error(msg):
                payload = _responses_payload(cfg, fallback_messages or messages)
            elif "store" in payload and "store" in msg:
                payload.pop("store", None)
            elif isinstance(payload.get("input"), list) and any(s in msg for s in ("input", "array", "list", "string", "messages")):
                payload["input"] = _messages_to_prompt(fallback_messages or messages)
            elif "reasoning" in payload and "reasoning" in msg:
                payload.pop("reasoning", None)
            elif "max_output_tokens" in payload and _is_unsupported_param_error(msg):
                payload["max_tokens"] = payload.pop("max_output_tokens")
            elif "max_output_tokens" in payload and _is_token_limit_error(msg):
                if not _lower_token_limit(payload, "max_output_tokens"):
                    break
            elif "max_tokens" in payload and _is_unsupported_param_error(msg):
                payload.pop("max_tokens", None)
            elif "max_tokens" in payload and _is_token_limit_error(msg):
                if not _lower_token_limit(payload, "max_tokens"):
                    break
            elif "temperature" in payload and any(s in msg for s in ("temperature", "not support")):
                payload.pop("temperature", None)
            else:
                break
        except Exception as exc:
            last = exc
            break
    raise last


def _call_responses(client, cfg, messages, previous_response_id=None, return_metadata=False, fallback_messages=None):
    kwargs = _responses_payload(cfg, messages, previous_response_id=previous_response_id)
    last = None
    for _ in range(7):
        try:
            result = _response_result(client.responses.create(**kwargs))
            return result if return_metadata else result["text"]
        except Exception as e:
            last = e
            msg = str(e).lower()
            if "previous_response_id" in kwargs and _is_previous_response_error(msg):
                kwargs = _responses_payload(cfg, fallback_messages or messages)
            elif "store" in kwargs and "store" in msg:
                kwargs.pop("store", None)
            elif isinstance(kwargs.get("input"), list) and any(s in msg for s in ("input", "array", "list", "string", "messages")):
                kwargs["input"] = _messages_to_prompt(fallback_messages or messages)
            elif "reasoning" in kwargs and "reasoning" in msg:
                kwargs.pop("reasoning", None)
            elif "max_output_tokens" in kwargs and _is_unsupported_param_error(msg):
                kwargs["max_tokens"] = kwargs.pop("max_output_tokens")
            elif "max_output_tokens" in kwargs and _is_token_limit_error(msg):
                if not _lower_token_limit(kwargs, "max_output_tokens"):
                    break
            elif "max_tokens" in kwargs and _is_unsupported_param_error(msg):
                kwargs.pop("max_tokens", None)
            elif "max_tokens" in kwargs and _is_token_limit_error(msg):
                if not _lower_token_limit(kwargs, "max_tokens"):
                    break
            elif "temperature" in kwargs and any(s in msg for s in ("temperature", "not support")):
                kwargs.pop("temperature", None)
            else:
                return _call_responses_http(
                    cfg,
                    messages,
                    previous_response_id=previous_response_id,
                    previous_error=e,
                    return_metadata=return_metadata,
                    fallback_messages=fallback_messages,
                    payload_override=kwargs,
                )
    return _call_responses_http(
        cfg,
        messages,
        previous_response_id=previous_response_id,
        previous_error=last,
        return_metadata=return_metadata,
        fallback_messages=fallback_messages,
        payload_override=kwargs,
    )


def _call_prompt_completion_http(cfg, messages, previous_error=None, payload_override=None):
    base_url = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    api_key_env = cfg.get("api_key_env") or "OPENAI_API_KEY"
    api_key = cfg.get("api_key") or os.environ.get(api_key_env) or "x"
    payload = dict(payload_override or _prompt_completion_payload(cfg, messages))
    last = previous_error
    for suffix in ("/completions", "/chat/completions"):
        req = urllib.request.Request(
            f"{base_url}{suffix}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=LLM_HTTP_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8") or "{}")
            return _completion_text_from_payload(data)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = str(exc)
            last = RuntimeError(f"HTTP {exc.code} from {suffix}: {detail}")
        except Exception as exc:
            last = exc
    raise last


def _call_prompt_completion(client, cfg, messages):
    kwargs = _prompt_completion_payload(cfg, messages)
    last = None
    for _ in range(4):
        try:
            return _completion_text(client.completions.create(**kwargs))
        except Exception as e:
            last = e
            msg = str(e).lower()
            if "temperature" in kwargs and any(s in msg for s in ("temperature", "not support")):
                kwargs.pop("temperature", None)
            elif "max_tokens" in kwargs and _is_unsupported_param_error(msg):
                kwargs.pop("max_tokens", None)
            elif "max_tokens" in kwargs and _is_token_limit_error(msg):
                if not _lower_token_limit(kwargs, "max_tokens"):
                    break
            else:
                return _call_prompt_completion_http(cfg, messages, previous_error=e, payload_override=kwargs)
    return _call_prompt_completion_http(cfg, messages, previous_error=last, payload_override=kwargs)


def _call_openai_compatible(cfg, messages, previous_response_id=None, return_metadata=False, fallback_messages=None):
    from openai import OpenAI

    api_key_env = cfg.get("api_key_env") or "OPENAI_API_KEY"
    api_key = cfg.get("api_key") or os.environ.get(api_key_env) or "x"
    base_url = cfg.get("base_url") or None
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=LLM_HTTP_TIMEOUT_SECONDS)
    backend = _normalize_generation_backend(cfg.get("backend"), cfg.get("provider") or "openai")
    if backend == "prompt":
        return _call_prompt_completion(client, cfg, messages)
    use_responses = backend == "responses"
    if use_responses:
        if hasattr(client, "responses"):
            return _call_responses(
                client,
                cfg,
                messages,
                previous_response_id=previous_response_id,
                return_metadata=return_metadata,
                fallback_messages=fallback_messages,
            )
        return _call_responses_http(
            cfg,
            messages,
            previous_response_id=previous_response_id,
            return_metadata=return_metadata,
            fallback_messages=fallback_messages,
        )
    kwargs = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": _safe_float(cfg.get("temperature"), 0.3),
        "max_completion_tokens": _chat_max_tokens(cfg),
    }
    if cfg.get("reasoning_effort"):
        kwargs["reasoning_effort"] = cfg["reasoning_effort"]
    last = None
    for _ in range(5):
        try:
            r = client.chat.completions.create(**kwargs)
            text = r.choices[0].message.content or ""
            return {"text": text, "response_id": ""} if return_metadata else text
        except Exception as e:
            last = e
            msg = str(e).lower()
            if _is_no_chat_template_error(msg):
                try:
                    return _call_responses(
                        client,
                        cfg,
                        messages,
                        previous_response_id=previous_response_id,
                        return_metadata=return_metadata,
                        fallback_messages=fallback_messages,
                    )
                except Exception:
                    text = _call_prompt_completion(client, cfg, messages)
                    return {"text": text, "response_id": ""} if return_metadata else text
            if "reasoning_effort" in kwargs and "reasoning_effort" in msg:
                kwargs.pop("reasoning_effort")
            elif "max_completion_tokens" in kwargs and _is_unsupported_param_error(msg):
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            elif "max_completion_tokens" in kwargs and _is_token_limit_error(msg):
                if not _lower_token_limit(kwargs, "max_completion_tokens"):
                    raise
            elif "max_tokens" in kwargs and _is_unsupported_param_error(msg):
                kwargs.pop("max_tokens", None)
            elif "max_tokens" in kwargs and _is_token_limit_error(msg):
                if not _lower_token_limit(kwargs, "max_tokens"):
                    raise
            elif "temperature" in kwargs and any(s in msg for s in ("temperature", "not support")):
                kwargs.pop("temperature", None)
            else:
                raise
    if return_metadata:
        raise last
    raise last


def _call_hf(cfg, messages):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = cfg["model"]
    cache_key = (model, cfg.get("dtype", "auto"), cfg.get("device") or "auto")
    with LOCK:
        pair = STATE["hf_cache"].get(cache_key)
    if pair is None:
        tok = AutoTokenizer.from_pretrained(model)
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        lm = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype_map.get(cfg.get("dtype", "auto"), "auto"),
            device_map=cfg.get("device") or "auto",
        )
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        pair = (tok, lm)
        with LOCK:
            STATE["hf_cache"][cache_key] = pair
    tok, lm = pair
    if getattr(tok, "chat_template", None):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    else:
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"
        enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(lm.device) for k, v in dict(enc).items()}
    input_len = enc["input_ids"].shape[-1]
    temp = _safe_float(cfg.get("temperature"), 0.3)
    gen = lm.generate(
        **enc,
        max_new_tokens=_chat_max_tokens(cfg, 8192),
        do_sample=temp > 0,
        temperature=max(temp, 1e-5),
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(gen[0][input_len:], skip_special_tokens=True).strip()


def _prepare_chat_request(body):
    question = str(body.get("question") or "").strip()
    if not question:
        raise ValueError("Question is required.")
    cfg = body.get("model_config") or {}
    if not cfg.get("model"):
        raise ValueError("Model is required.")
    context_mode = _normalize_context_mode(body.get("context_mode") or "none")
    selected_nodes = [str(x) for x in (body.get("selected_nodes") or []) if str(x)]
    if context_mode == "none" and not selected_nodes:
        context = _empty_graph_context(question, mode="none")
    else:
        G = _require_graph()
        context = _context_for_llm(
            G,
            question,
            selected_nodes=selected_nodes,
            query=body.get("query") or "",
            depth=_safe_int(body.get("depth"), 1),
            max_nodes=_safe_int(body.get("max_nodes"), 90),
            max_edges=_safe_int(body.get("max_edges"), 160),
            context_mode=context_mode,
        )
    report_context = _report_context_payload(body.get("report_context"))
    if context.get("mode") == "graph_rag":
        mode_instructions = (
            "Use the selected nodes, semantic/text retrieval hits, "
            "neighborhoods, path connectors, and centrality anchors in the graph context as exploratory "
            "evidence, not verified facts. Surface the most relevant nodes and neighborhoods, explain why "
            "they were retrieved, identify bridge concepts and structural gaps, and propose concrete next "
            "queries or experiments. If retrieval is weak or ambiguous, say exactly what more context is needed."
        )
    elif context.get("mode") == "none" and not context.get("node_count"):
        mode_instructions = (
            "Answer the user directly. "
            "Do not claim graph evidence was provided unless an attached report or selected-node context is present."
        )
    else:
        mode_instructions = (
            "Use the graph context as exploratory leads, "
            "not as verified facts. Explain what the selected neighborhood or paths suggest, name "
            "specific mechanisms, identify structural gaps, and propose concrete next queries or "
            "experiments when useful. If the graph context is insufficient, say what is missing."
        )
    assistant_instruction = f"{GRAPH_PREFLEXOR_ASSISTANT_PROMPT}\n\n{mode_instructions}"
    report_text = f"# Attached user-selected report context\n{report_context['text']}\n\n" if report_context else ""
    if context.get("text"):
        user = (
            f"# User question\n{question}\n\n"
            f"# Graph context packet\n{context['text']}\n\n"
            f"{report_text}"
            "Answer directly. Refer to node labels and path structure when they matter."
        )
    else:
        user = (
            f"# User question\n{question}\n\n"
            f"{report_text}"
            "Answer directly."
        )
    provider = cfg.get("provider", "openai")
    backend = _normalize_generation_backend(cfg.get("backend"), cfg.get("provider") or "openai")
    previous_response_id = str(body.get("previous_response_id") or "").strip()
    native_responses_state = backend == "responses" and bool(previous_response_id)
    instruction_role = _assistant_instruction_role(cfg)
    messages = [{"role": instruction_role, "content": assistant_instruction}]
    fallback_messages = [{"role": instruction_role, "content": assistant_instruction}]
    raw_history = body.get("history") or []
    history_turns = 0
    if isinstance(raw_history, list):
        for item in raw_history[-8:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = str(item.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                history_turns += 1
                history_message = {"role": role, "content": content[:5000]}
                fallback_messages.append(history_message)
                if not native_responses_state:
                    messages.append(history_message)
    messages.append({"role": "user", "content": user})
    fallback_messages.append({"role": "user", "content": user})
    return {
        "question": question,
        "cfg": cfg,
        "context": context,
        "report_context": report_context,
        "provider": provider,
        "backend": backend,
        "previous_response_id": previous_response_id,
        "native_responses_state": native_responses_state,
        "instruction_role": instruction_role,
        "assistant_instruction": assistant_instruction,
        "user_prompt": user,
        "messages": messages,
        "fallback_messages": fallback_messages,
        "history_turns": history_turns,
    }


def _chat_state_mode(backend, response_id="", history_turns=0, native_responses_state=False):
    if backend == "responses" and (response_id or native_responses_state):
        return "responses_previous_response_id"
    if history_turns:
        return "history_replay"
    return "ready_for_multiturn"


def _answer_question(body):
    prepared = _prepare_chat_request(body)
    cfg = prepared["cfg"]
    provider = prepared["provider"]
    backend = prepared["backend"]
    if provider == "hf":
        answer = _call_hf(cfg, prepared["messages"])
        response_id = ""
    else:
        result = _call_openai_compatible(
            cfg,
            prepared["messages"],
            previous_response_id=prepared["previous_response_id"] if prepared["native_responses_state"] else None,
            return_metadata=True,
            fallback_messages=prepared["fallback_messages"],
        )
        answer = str(result.get("text") or "") if isinstance(result, dict) else str(result or "")
        response_id = str(result.get("response_id") or "") if isinstance(result, dict) else ""
    context = prepared["context"]
    report_context = prepared["report_context"]
    public_context = {k: v for k, v in context.items() if k != "text"}
    if report_context:
        public_context["report_context"] = {k: v for k, v in report_context.items() if k != "text"}
    state_mode = _chat_state_mode(
        backend,
        response_id=response_id,
        history_turns=prepared["history_turns"],
        native_responses_state=prepared["native_responses_state"],
    )
    return {
        "answer": answer,
        "context": public_context,
        "response_id": response_id,
        "stateful": backend in {"responses", "chat", "hf"} and (bool(response_id) or bool(prepared["history_turns"]) or backend == "chat"),
        "state_mode": state_mode,
        "backend": backend,
    }


def _chat_context_preview(body):
    prepared = _prepare_chat_request(body)
    context = prepared["context"]
    report_context = prepared["report_context"]
    public_context = {k: v for k, v in context.items() if k != "text"}
    if report_context:
        public_context["report_context"] = {k: v for k, v in report_context.items() if k != "text"}
    sanitized_cfg = {
        key: value
        for key, value in dict(prepared["cfg"]).items()
        if key not in {"api_key"} and value not in (None, "")
    }
    return {
        "backend": prepared["backend"],
        "state_mode": _chat_state_mode(
            prepared["backend"],
            history_turns=prepared["history_turns"],
            native_responses_state=prepared["native_responses_state"],
        ),
        "instruction_role": prepared["instruction_role"],
        "assistant_instruction": prepared["assistant_instruction"],
        "user_prompt": prepared["user_prompt"],
        "messages": prepared["messages"],
        "fallback_messages": prepared["fallback_messages"] if prepared["native_responses_state"] else [],
        "context": public_context,
        "request": {
            "question": prepared["question"],
            "selected_nodes": body.get("selected_nodes") or [],
            "query": body.get("query") or "",
            "depth": _safe_int(body.get("depth"), 1),
            "max_nodes": _safe_int(body.get("max_nodes"), 90),
            "max_edges": _safe_int(body.get("max_edges"), 160),
            "context_mode": _normalize_context_mode(body.get("context_mode") or "none"),
            "report_context": body.get("report_context") or None,
            "model_config": sanitized_cfg,
            "history_turns": prepared["history_turns"],
            "previous_response_id": prepared["previous_response_id"] or "",
        },
    }


def _graph_rag_context_payload(body):
    G = _require_graph()
    question = str(body.get("question") or body.get("query") or "").strip()
    context = _context_for_llm(
        G,
        question,
        selected_nodes=body.get("selected_nodes") or [],
        query=body.get("query") or "",
        depth=_safe_int(body.get("depth"), 1),
        max_nodes=_safe_int(body.get("max_nodes"), 240),
        max_edges=_safe_int(body.get("max_edges"), 520),
        context_mode="graph_rag",
    )
    return {"context": {k: v for k, v in context.items() if k != "text"}}


def _report_context_payload(raw):
    if not isinstance(raw, dict):
        return None
    out = str(raw.get("out") or "").strip()
    if not out:
        return None
    max_chars = max(1000, min(50000, _safe_int(raw.get("max_chars"), 12000)))
    include_report = raw.get("include_report", True) is not False
    include_profile = bool(raw.get("include_profile"))
    if not include_report and not include_profile:
        return None
    artifacts = _profile_artifacts(out)
    summary = dict(artifacts.get("summary") or {})
    title = summary.get("topic") or Path(out).name
    parts = []
    included = []
    total_chars = 0
    used_chars = 0
    truncated = False

    def add_part(label, path_value):
        nonlocal total_chars, used_chars, truncated
        path = Path(path_value or "")
        if not path.exists() or not path.is_file():
            raise ValueError(f"selected {label} context is not available: {out}")
        text = path.read_text(errors="replace")
        total_chars += len(text)
        remaining = max_chars - used_chars
        if remaining <= 0:
            truncated = True
            return
        excerpt = text[:remaining]
        used_chars += len(excerpt)
        truncated = truncated or len(text) > len(excerpt)
        included.append(label)
        parts.append(
            f"## {label}\n"
            f"Source: {path.name}\n"
            f"Excerpt chars: {len(excerpt)} of {len(text)}"
            f"{' (truncated)' if len(text) > len(excerpt) else ''}\n\n"
            f"{excerpt}"
        )

    if include_report:
        add_part("report.md", artifacts.get("report_path") or "")
    if include_profile:
        add_part("profile.json", artifacts.get("profile_path") or "")

    context_text = "\n\n".join(parts)
    return {
        "out": artifacts.get("out") or out,
        "title": str(title),
        "chars": used_chars,
        "total_chars": total_chars,
        "truncated": truncated,
        "included": included,
        "text": (
            f"Profile context: {artifacts.get('out') or out}\n"
            f"Title/topic: {title}\n"
            f"Nodes: {summary.get('nodes', '')}; Edges: {summary.get('edges', '')}; Modules: {summary.get('modules', '')}\n"
            f"Included: {', '.join(included)}\n"
            f"Excerpt chars: {used_chars} of {total_chars}"
            f"{' (truncated)' if truncated else ''}\n\n"
            f"{context_text}"
        ),
    }


def _start_ideate_job(body):
    topic = str(body.get("topic") or "").strip()
    if not topic:
        raise ValueError("Topic is required.")
    out = str(body.get("out") or "").strip()
    if not out:
        safe = "".join(c if c.isalnum() else "_" for c in topic.lower())[:60].strip("_")
        out = f"runs/explorer_{safe or 'run'}"
    cmd = ["python", "ideate.py", "--topic", topic, "--out", out]
    for arg in ("strategy", "context_mode"):
        if body.get(arg):
            cmd.extend([f"--{arg.replace('_', '-')}", str(body[arg])])
    for arg in ("budget_calls", "budget_tokens", "max_iters", "fanout", "dedup_threshold"):
        value = body.get(arg)
        if value not in (None, ""):
            cmd.extend([f"--{arg.replace('_', '-')}", str(value)])
    if body.get("config"):
        cmd.extend(["--config", str(body["config"])])

    cleared_artifacts = _clear_ideate_artifacts(out) if body.get("clear_output", True) else []
    job_id = uuid.uuid4().hex[:10]
    log_path = ROOT / f"job_{job_id}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=IDEATION_DIR,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    job = {
        "id": job_id,
        "cmd": cmd,
        "cwd": str(IDEATION_DIR),
        "out": out,
        "cleared_artifacts": cleared_artifacts,
        "log_path": str(log_path),
        "status": "running",
        "returncode": None,
        "started_at": time.time(),
        "ended_at": None,
        "budget_calls": body.get("budget_calls"),
        "max_iters": body.get("max_iters"),
        "budget_tokens": body.get("budget_tokens"),
        "stop_requested": False,
        "proc": proc,
    }
    with LOCK:
        STATE["jobs"][job_id] = job

    def wait():
        rc = proc.wait()
        log.close()
        with LOCK:
            if job.get("stop_requested"):
                job["status"] = "stopped"
            else:
                job["status"] = "done" if rc == 0 else "failed"
            job["returncode"] = rc
            job["ended_at"] = time.time()

    threading.Thread(target=wait, daemon=True).start()
    return _job_public(job)


def _job_public(job):
    return {k: v for k, v in dict(job).items() if k != "proc"}


def _job_status(job_id):
    with LOCK:
        job = STATE["jobs"].get(job_id)
        if not job:
            raise ValueError("Unknown job id.")
        out = _job_public(job)
    try:
        out["log_tail"] = Path(out["log_path"]).read_text(errors="replace")[-8000:]
    except Exception:
        out["log_tail"] = ""
    run_dir = _resolve_run_path(out["out"])
    out.update(_snapshot_meta(run_dir))
    out["progress"] = _progress_payload(
        run_dir,
        budget_calls=out.get("budget_calls"),
        max_iters=out.get("max_iters"),
        max_tokens=out.get("budget_tokens"),
    )
    return out


def _stop_job(body):
    job_id = str(body.get("id") or "").strip()
    if not job_id:
        raise ValueError("Job id is required.")
    with LOCK:
        job = STATE["jobs"].get(job_id)
        if not job:
            raise ValueError("Unknown job id.")
        proc = job.get("proc")
        if job.get("status") not in ("running", "stopping"):
            return _job_status(job_id)
        job["stop_requested"] = True
        job["status"] = "stopping"
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, 15)
        except Exception:
            proc.terminate()
    return _job_status(job_id)


def _profile_summary_from_json(path):
    if not path.exists():
        return {}
    try:
        with path.open(errors="replace") as f:
            profile = json.load(f) or {}
    except Exception:
        return {}
    stats = dict(profile.get("global_stats") or {})
    source = dict(stats.get("source") or {})
    llm = dict(profile.get("llm_summaries") or {})
    return {
        "topic": str(profile.get("topic") or ""),
        "generated_at": str(profile.get("generated_at") or ""),
        "nodes": _safe_int(stats.get("nodes")),
        "edges": _safe_int(stats.get("edges")),
        "density": _safe_float(stats.get("density")),
        "components": _safe_int(stats.get("components")),
        "modules": len(profile.get("communities") or []),
        "modularity": _safe_float(stats.get("modularity")),
        "embed_model": str(profile.get("embedding_model") or ""),
        "embed_error": str(profile.get("embedding_error") or ""),
        "llm_model": str(llm.get("model") or ""),
        "llm_backend": str(llm.get("backend") or ""),
        "pdf_error": str(profile.get("pdf_error") or ""),
        "source": str(source.get("graph_path") or source.get("run_dir") or ""),
    }


def _profile_artifacts(out_value):
    out_dir = _resolve_profile_out(out_value)
    report = out_dir / "report.md"
    profile_json = out_dir / "profile.json"
    pdf = out_dir / "report.pdf"
    figures_dir = out_dir / "figures"
    figures = []
    if figures_dir.exists():
        for path in sorted(figures_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"):
                figures.append(str(path.relative_to(out_dir)))
    mtimes = []
    for path in [out_dir, report, profile_json, pdf, *[out_dir / fig for fig in figures]]:
        try:
            if path.exists():
                mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
    return {
        "out": _relative_to_ideation(out_dir),
        "absolute_out": str(out_dir),
        "ready": report.exists() or profile_json.exists(),
        "report_path": str(report) if report.exists() else "",
        "profile_path": str(profile_json) if profile_json.exists() else "",
        "pdf_path": str(pdf) if pdf.exists() else "",
        "figures": figures,
        "summary": _profile_summary_from_json(profile_json),
        "updated_at": max(mtimes) if mtimes else 0,
    }


def _profile_reports_payload(run_value):
    run_dir = _resolve_run_path(run_value)
    if run_dir.is_file():
        run_dir = run_dir.parent
    if not run_dir.exists():
        return {"run": _relative_to_ideation(run_dir), "reports": []}
    candidates = []
    for child in run_dir.iterdir():
        if child.is_dir() and (child.name.startswith("profile") or (child / "report.md").exists() or (child / "profile.json").exists()):
            candidates.append(child)
    direct = run_dir / "profile"
    if direct.exists() and direct not in candidates:
        candidates.append(direct)
    reports = []
    for path in sorted(candidates):
        try:
            artifacts = _profile_artifacts(str(path))
        except Exception:
            continue
        if artifacts.get("ready"):
            reports.append(artifacts)
    reports.sort(key=lambda item: item.get("updated_at") or 0, reverse=True)
    return {"run": _relative_to_ideation(run_dir), "reports": reports}


def _profile_progress(log_text, status):
    matches = re.findall(r"^\[(\d{1,2})/(\d{1,2})\]\s+(.+)$", log_text, flags=re.MULTILINE)
    current = total = 0
    message = ""
    if matches:
        current, total, message = matches[-1]
        current, total = int(current), int(total)
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    detail = lines[-1] if lines else ""
    percent = (current / total) if total else 0.0
    if status == "done":
        percent = 1.0
    return {
        "percent": max(0.0, min(1.0, percent)),
        "current": current,
        "total": total,
        "message": message,
        "detail": detail,
    }


def _profile_job_public(job):
    return {k: v for k, v in dict(job).items() if k != "proc"}


def _add_optional_arg(cmd, body, name, flag=None):
    value = body.get(name)
    if value not in (None, ""):
        cmd.extend([flag or f"--{name.replace('_', '-')}", str(value)])


def _start_profile_job(body):
    run = str(body.get("run") or "").strip()
    graph = str(body.get("graph") or "").strip()
    if bool(run) == bool(graph):
        raise ValueError("Choose exactly one source: run or graph.")

    out = str(body.get("out") or "").strip()
    if not out:
        if run:
            run_dir = _resolve_run_path(run)
            out = str(run_dir / "profile")
        else:
            out = "graph_profile"
    _resolve_profile_out(out)

    cmd = ["python", "profile_graph.py"]
    if run:
        cmd.extend(["--run", run])
    else:
        cmd.extend(["--graph", graph])
    cmd.extend(["--out", out])

    _add_optional_arg(cmd, body, "embed_model")
    _add_optional_arg(cmd, body, "top_nodes")
    _add_optional_arg(cmd, body, "max_modules")
    _add_optional_arg(cmd, body, "profile_preset")
    if bool(body.get("llm")):
        cmd.append("--llm")
    _add_optional_arg(cmd, body, "llm_modules")
    _add_optional_arg(cmd, body, "backend")
    _add_optional_arg(cmd, body, "model")
    _add_optional_arg(cmd, body, "base_url")
    _add_optional_arg(cmd, body, "temperature")
    _add_optional_arg(cmd, body, "max_summary_tokens")
    _add_optional_arg(cmd, body, "deep_pass_tokens")
    _add_optional_arg(cmd, body, "deep_dive_tokens")
    _add_optional_arg(cmd, body, "reasoning_effort")
    _add_optional_arg(cmd, body, "llm_deep_passes")
    _add_optional_arg(cmd, body, "report_review_tokens")
    _add_optional_arg(cmd, body, "report_review_max_chunks")
    _add_optional_arg(cmd, body, "report_review_chunk_chars")
    _add_optional_arg(cmd, body, "report_review_memo_chars")
    _add_optional_arg(cmd, body, "device")
    _add_optional_arg(cmd, body, "dtype")
    if body.get("llm_report_review") is False:
        cmd.append("--no-llm-report-review")
    if body.get("pdf") is False:
        cmd.append("--no-pdf")

    job_id = uuid.uuid4().hex[:10]
    log_path = ROOT / f"profile_job_{job_id}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=IDEATION_DIR,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    job = {
        "id": job_id,
        "cmd": cmd,
        "cwd": str(IDEATION_DIR),
        "run": run,
        "graph": graph,
        "out": out,
        "log_path": str(log_path),
        "status": "running",
        "returncode": None,
        "started_at": time.time(),
        "ended_at": None,
        "stop_requested": False,
        "proc": proc,
    }
    with LOCK:
        STATE["profile_jobs"][job_id] = job

    def wait():
        rc = proc.wait()
        log.close()
        with LOCK:
            if job.get("stop_requested"):
                job["status"] = "stopped"
            else:
                job["status"] = "done" if rc == 0 else "failed"
            job["returncode"] = rc
            job["ended_at"] = time.time()

    threading.Thread(target=wait, daemon=True).start()
    return _profile_job_status(job_id)


def _profile_job_status(job_id):
    with LOCK:
        job = STATE["profile_jobs"].get(job_id)
        if not job:
            raise ValueError("Unknown profile job id.")
        out = _profile_job_public(job)
    try:
        log_tail = Path(out["log_path"]).read_text(errors="replace")[-12000:]
    except Exception:
        log_tail = ""
    out["log_tail"] = log_tail
    out["progress"] = _profile_progress(log_tail, out.get("status"))
    try:
        out["artifacts"] = _profile_artifacts(out["out"])
    except Exception as exc:
        out["artifacts"] = {"out": out.get("out", ""), "ready": False, "error": str(exc)}
    return out


def _stop_profile_job(body):
    job_id = str(body.get("id") or "").strip()
    if not job_id:
        raise ValueError("Profile job id is required.")
    with LOCK:
        job = STATE["profile_jobs"].get(job_id)
        if not job:
            raise ValueError("Unknown profile job id.")
        proc = job.get("proc")
        if job.get("status") not in ("running", "stopping"):
            return _profile_job_status(job_id)
        job["stop_requested"] = True
        job["status"] = "stopping"
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, 15)
        except Exception:
            proc.terminate()
    return _profile_job_status(job_id)


def _profile_report_payload(body):
    artifacts = _profile_artifacts(body.get("out") or "")
    markdown = ""
    profile_json = ""
    report_value = artifacts.get("report_path") or ""
    report_path = Path(report_value) if report_value else None
    if report_path and report_path.exists() and report_path.is_file():
        markdown = report_path.read_text(errors="replace")
    profile_value = artifacts.get("profile_path") or ""
    profile_path = Path(profile_value) if profile_value else None
    if profile_path and profile_path.exists() and profile_path.is_file():
        profile_json = profile_path.read_text(errors="replace")
    return {
        "artifacts": artifacts,
        "markdown": markdown,
        "profile_json": profile_json,
    }


def _resolve_report_asset(out_value, file_value):
    out_dir = _resolve_profile_out(out_value)
    file_name = str(file_value or "").strip()
    if not file_name:
        raise ValueError("file is required")
    target = (out_dir / file_name).resolve()
    if not (target == out_dir or out_dir in target.parents):
        raise ValueError("report asset path escapes the output directory")
    if not target.exists() or not target.is_file():
        raise ValueError(f"report asset not found: {file_name}")
    return target


class Handler(SimpleHTTPRequestHandler):
    server_version = "GraphExplorer/0.1"

    def __init__(self, *args, **kwargs):
        directory = REACT_DIST_DIR if (REACT_DIST_DIR / "index.html").exists() else STATIC_DIR
        super().__init__(*args, directory=str(directory), **kwargs)

    def translate_path(self, path):
        root = Path(self.directory)
        translated = Path(super().translate_path(path))
        if root == REACT_DIST_DIR and not translated.exists() and not path.startswith("/api/"):
            return str(REACT_DIST_DIR / "index.html")
        return str(translated)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _json(self, obj, status=200):
        raw = json.dumps(obj, indent=None).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _error(self, exc, status=400):
        self._json({"error": str(exc), "trace": traceback.format_exc(limit=4)}, status=status)

    def _file(self, path):
        path = Path(path)
        raw = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            if parsed.path == "/":
                self.path = "/index.html"
            return super().do_GET()
        try:
            qs = parse_qs(parsed.query)
            if parsed.path == "/api/logo":
                self._file(LOGO_PATH)
            elif parsed.path == "/api/graph":
                G = _require_graph()
                with LOCK:
                    self._json(graph_payload(
                        G,
                        name=STATE.get("graph_name", ""),
                        path=STATE.get("graph_path", ""),
                        topic=STATE.get("topic", ""),
                    ))
            elif parsed.path == "/api/runs":
                self._json(_runs_payload())
            elif parsed.path == "/api/job":
                self._json(_job_status((qs.get("id") or [""])[0]))
            elif parsed.path == "/api/profile_job":
                self._json(_profile_job_status((qs.get("id") or [""])[0]))
            elif parsed.path == "/api/embedding_status":
                self._json(_embedding_status())
            elif parsed.path == "/api/report_asset":
                self._file(_resolve_report_asset((qs.get("out") or [""])[0], (qs.get("file") or [""])[0]))
            elif parsed.path == "/api/config":
                self._json(_config_payload())
            else:
                self._json({"error": "unknown endpoint"}, status=404)
        except Exception as exc:
            self._error(exc)

    def do_POST(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            return self._json({"error": "unknown endpoint"}, status=404)
        try:
            body = self._read_body()
            if parsed.path == "/api/load_graphml":
                text = body.get("graphml") or ""
                if not text.strip():
                    raise ValueError("graphml text is required")
                G = _read_graphml_text(text)
                self._json(_set_graph(G, name=body.get("name") or "uploaded.graphml", topic=body.get("topic") or ""))
            elif parsed.path == "/api/load_run":
                run, graph_path, G = _load_run_graph(body.get("run") or "")
                self._json(_set_graph(G, name=run.name, path=str(graph_path), topic=_load_topic(run, G)))
            elif parsed.path == "/api/run_graphs":
                self._json(_run_graphs_payload(body.get("run") or ""))
            elif parsed.path == "/api/graphml_files":
                self._json(_graphml_files_payload())
            elif parsed.path == "/api/search":
                G = _require_graph()
                self._json({"results": _search_nodes(G, body.get("query", ""), int(body.get("limit", 50)))})
            elif parsed.path == "/api/embedding_index":
                self._json(_start_embedding_index(body))
            elif parsed.path == "/api/neighborhood":
                G = _require_graph()
                seeds = [str(x) for x in body.get("nodes", [])]
                nodes = _neighborhood_nodes(G, seeds, int(body.get("depth", 1)), int(body.get("limit", 400)))
                payload = _subgraph_payload(nodes, name=f"neighborhood depth {body.get('depth', 1)}")
                payload["focus_nodes"] = list(nodes)
                self._json(payload)
            elif parsed.path == "/api/path":
                self._json(_path_payload(
                    str(body.get("source") or ""),
                    str(body.get("target") or ""),
                    int(body.get("k", 5)),
                    int(body.get("cutoff", 6)),
                ))
            elif parsed.path == "/api/multipath":
                self._json(_multipath_payload(body))
            elif parsed.path == "/api/bridge_suggestions":
                self._json(_bridge_suggestions_payload(body))
            elif parsed.path == "/api/ask":
                self._json(_answer_question(body))
            elif parsed.path == "/api/chat_context_preview":
                self._json(_chat_context_preview(body))
            elif parsed.path == "/api/graph_rag_context":
                self._json(_graph_rag_context_payload(body))
            elif parsed.path == "/api/ideate":
                self._json(_start_ideate_job(body))
            elif parsed.path == "/api/clear_graph":
                self._json(_clear_graph())
            elif parsed.path == "/api/stop_job":
                self._json(_stop_job(body))
            elif parsed.path == "/api/profile_graph":
                self._json(_start_profile_job(body))
            elif parsed.path == "/api/stop_profile_job":
                self._json(_stop_profile_job(body))
            elif parsed.path == "/api/profile_reports":
                self._json(_profile_reports_payload(body.get("run") or ""))
            elif parsed.path == "/api/profile_report":
                self._json(_profile_report_payload(body))
            elif parsed.path == "/api/model_status":
                self._json(_model_status(body))
            elif parsed.path == "/api/model_probe":
                self._json(_model_probe(body))
            elif parsed.path == "/api/config_preview":
                self._json({"config": _roles_to_config_text(body.get("roles") or {})})
            elif parsed.path == "/api/save_config":
                self._json(_save_config(body))
            elif parsed.path == "/api/compare_run":
                self._json(_compare_run(body))
            else:
                self._json({"error": "unknown endpoint"}, status=404)
        except Exception as exc:
            self._error(exc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--run", help="optional run directory to load on startup")
    args = p.parse_args()

    if args.run:
        try:
            run, graph_path, G = _load_run_graph(args.run)
        except ValueError as exc:
            print(f"[graph-explorer] warning: {exc}")
            print("[graph-explorer] starting without a preloaded graph")
        else:
            _set_graph(G, name=run.name, path=str(graph_path), topic=_load_topic(run, G))

    STATIC_DIR.mkdir(exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[graph-explorer] serving {url}")
    print("[graph-explorer] open the URL in a browser; Ctrl-C stops the server")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
