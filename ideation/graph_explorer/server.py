#!/usr/bin/env python
"""Local browser explorer for Graph-PRefLexOR graphml runs.

This is intentionally dependency-light: stdlib HTTP server plus NetworkX. The
browser does the interactive rendering; this process loads GraphML, runs graph
queries, optionally starts ideate.py jobs, and calls a user-selected LLM backend.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
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
IDEATION_DIR = ROOT.parent
PROJECT_DIR = IDEATION_DIR.parent

STATE = {
    "graph": None,
    "graph_id": None,
    "graph_name": "",
    "graph_path": "",
    "topic": "",
    "jobs": {},
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
    return candidates


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
        run_dir = target.parent
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
    return {
        "role": role,
        "provider": data.get("provider") or "openai",
        "model": str(data.get("model") or ""),
        "base_url": str(data.get("base_url") or ""),
        "api_key_env": str(data.get("api_key_env") or data.get("api_key") or ""),
        "temperature": data.get("temperature", ""),
        "max_tokens": data.get("max_tokens", ""),
        "reasoning_effort": str(data.get("reasoning_effort") or ""),
    }


def _config_payload():
    cfg, path = _load_config()
    roles = ["generator", "questioner", "graph_qa", "judge", "baseline", "embedder"]
    return {
        "path": str(path),
        "exists": path.exists(),
        "roles": {role: _role_from_config(role, cfg) for role in roles},
    }


def _coerce_role(raw):
    role = dict(raw or {})
    for key in ("role", "provider", "model", "base_url", "api_key_env", "reasoning_effort"):
        role[key] = str(role.get(key) or "").strip()
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
        for key in ("provider", "model", "base_url", "api_key_env", "temperature", "max_tokens", "reasoning_effort"):
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
    return {"ok": True}


def _require_graph():
    with LOCK:
        G = STATE.get("graph")
        if G is None:
            raise ValueError("No graph is loaded yet.")
        return G


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
    for n, d in G.nodes(data=True):
        attrs = {str(k): str(v) for k, v in dict(d).items()}
        label = str(attrs.get("label") or n)
        hay = " ".join([label, str(n)] + [f"{k} {v}" for k, v in attrs.items()]).lower()
        hits = sum(1 for t in terms if t in hay)
        if q in hay:
            hits += 2
        if hits:
            out.append({
                "id": str(n),
                "label": label,
                "score": float(hits + math.log1p(degree.get(n, 0)) + pr.get(n, 0.0) * 10.0),
                "degree": int(degree.get(n, 0)),
                "pagerank": float(pr.get(n, 0.0)),
                "core": int(core.get(n, 0)),
                "iter": _int_attr(d, "iter", 0),
            })
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
    else:
        for i, a in enumerate(anchors):
            for b in anchors[i + 1:]:
                pairs.append((a, b))

    paths = []
    for a, b in pairs[:24]:
        if a not in U or b not in U:
            continue
        try:
            p = nx.shortest_path(U, a, b)
        except nx.NetworkXNoPath:
            continue
        if cutoff <= 0 or len(p) - 1 <= cutoff:
            paths.append([str(x) for x in p])

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


def _context_for_llm(G, question, selected_nodes=None, query="", depth=1, max_nodes=90, max_edges=160):
    selected_nodes = [str(n) for n in (selected_nodes or []) if str(n) in G]
    seeds = list(selected_nodes)
    if query:
        seeds.extend([r["id"] for r in _search_nodes(G, query, limit=12)])
    if seeds:
        nodes = _neighborhood_nodes(G, seeds, depth=depth, limit=max_nodes)
    else:
        degree = dict(G.to_undirected().degree())
        nodes = set(n for n, _ in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:max_nodes])

    H = G.subgraph(nodes).copy()
    degree = dict(G.to_undirected().degree())
    ranked_nodes = sorted(H.nodes, key=lambda n: degree.get(n, 0), reverse=True)[:max_nodes]
    edge_rows = []
    for u, v, d in H.edges(data=True):
        edge_rows.append((degree.get(u, 0) + degree.get(v, 0), str(u), str(v), str(d.get("relation", "related_to"))))
    edge_rows.sort(reverse=True)
    edge_rows = edge_rows[:max_edges]

    shortest_paths = []
    if len(selected_nodes) >= 2:
        U = G.to_undirected()
        for a, b in zip(selected_nodes, selected_nodes[1:]):
            try:
                p = nx.shortest_path(U, a, b)
                if len(p) <= 10:
                    shortest_paths.append(" -> ".join(str(x) for x in p))
            except Exception:
                pass

    node_text = "\n".join(f"- {_format_node(G, n)}; degree={degree.get(n, 0)}" for n in ranked_nodes)
    edge_text = "\n".join(f"- {u} -[{rel}]- {v}" for _, u, v, rel in edge_rows)
    path_text = "\n".join(f"- {p}" for p in shortest_paths) if shortest_paths else "(none requested or no short path found)"
    return {
        "selected": selected_nodes,
        "query": query,
        "node_count": len(ranked_nodes),
        "edge_count": len(edge_rows),
        "text": (
            f"Graph: {STATE.get('graph_name') or '(loaded graph)'}\n"
            f"Topic: {STATE.get('topic') or '(not recorded)'}\n"
            f"Question: {question}\n"
            f"Selected nodes: {', '.join(selected_nodes) if selected_nodes else '(none)'}\n"
            f"Search query: {query or '(none)'}\n\n"
            f"Key nodes in focus context:\n{node_text or '(none)'}\n\n"
            f"Edges in focus context:\n{edge_text or '(none)'}\n\n"
            f"Short selected paths:\n{path_text}\n"
        ),
    }


def _call_openai_compatible(cfg, messages):
    from openai import OpenAI

    api_key_env = cfg.get("api_key_env") or "OPENAI_API_KEY"
    api_key = cfg.get("api_key") or os.environ.get(api_key_env) or "x"
    base_url = cfg.get("base_url") or None
    client = OpenAI(base_url=base_url, api_key=api_key)
    kwargs = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": _safe_float(cfg.get("temperature"), 0.3),
        "max_completion_tokens": _safe_int(cfg.get("max_tokens"), 1600),
    }
    if cfg.get("reasoning_effort"):
        kwargs["reasoning_effort"] = cfg["reasoning_effort"]
    last = None
    for _ in range(5):
        try:
            r = client.chat.completions.create(**kwargs)
            return r.choices[0].message.content or ""
        except Exception as e:
            last = e
            msg = str(e).lower()
            if "reasoning_effort" in kwargs and "reasoning_effort" in msg:
                kwargs.pop("reasoning_effort")
            elif "max_completion_tokens" in kwargs and any(s in msg for s in ("unsupported", "unexpected", "not supported")):
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            elif "temperature" in kwargs and any(s in msg for s in ("temperature", "not support")):
                kwargs.pop("temperature", None)
            else:
                raise
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
        max_new_tokens=_safe_int(cfg.get("max_tokens"), 1200),
        do_sample=temp > 0,
        temperature=max(temp, 1e-5),
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(gen[0][input_len:], skip_special_tokens=True).strip()


def _answer_question(body):
    G = _require_graph()
    question = str(body.get("question") or "").strip()
    if not question:
        raise ValueError("Question is required.")
    cfg = body.get("model_config") or {}
    if not cfg.get("model"):
        raise ValueError("Model is required.")
    context = _context_for_llm(
        G,
        question,
        selected_nodes=body.get("selected_nodes") or [],
        query=body.get("query") or "",
        depth=_safe_int(body.get("depth"), 1),
        max_nodes=_safe_int(body.get("max_nodes"), 90),
        max_edges=_safe_int(body.get("max_edges"), 160),
    )
    system = (
        "You are a graph-aware research assistant. Use the graph context as exploratory leads, "
        "not as verified facts. Explain what the selected neighborhood or paths suggest, name "
        "specific mechanisms, identify structural gaps, and propose concrete next queries or "
        "experiments when useful. If the graph context is insufficient, say what is missing."
    )
    user = (
        f"# User question\n{question}\n\n"
        f"# Graph context packet\n{context['text']}\n\n"
        "Answer directly. Refer to node labels and path structure when they matter."
    )
    provider = cfg.get("provider", "openai")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if provider == "hf":
        answer = _call_hf(cfg, messages)
    else:
        answer = _call_openai_compatible(cfg, messages)
    return {"answer": answer, "context": {k: v for k, v in context.items() if k != "text"}}


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

    def do_GET(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            if parsed.path == "/":
                self.path = "/index.html"
            return super().do_GET()
        try:
            qs = parse_qs(parsed.query)
            if parsed.path == "/api/graph":
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
            elif parsed.path == "/api/search":
                G = _require_graph()
                self._json({"results": _search_nodes(G, body.get("query", ""), int(body.get("limit", 50)))})
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
            elif parsed.path == "/api/ideate":
                self._json(_start_ideate_job(body))
            elif parsed.path == "/api/clear_graph":
                self._json(_clear_graph())
            elif parsed.path == "/api/stop_job":
                self._json(_stop_job(body))
            elif parsed.path == "/api/model_status":
                self._json(_model_status(body))
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
