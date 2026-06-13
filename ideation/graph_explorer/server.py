#!/usr/bin/env python
"""Local browser explorer for Graph-PRefLexOR graphml runs.

This is intentionally dependency-light: stdlib HTTP server plus NetworkX. The
browser does the interactive rendering; this process loads GraphML, runs graph
queries, optionally starts ideate.py jobs, and calls a user-selected LLM backend.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import networkx as nx


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
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


def _centrality(G):
    U = G.to_undirected()
    degree = dict(U.degree())
    try:
        pr = nx.pagerank(G if G.number_of_edges() else U, max_iter=100)
    except Exception:
        total = max(1, sum(degree.values()))
        pr = {n: degree.get(n, 0) / total for n in U.nodes}
    try:
        core = nx.core_number(U) if U.number_of_edges() else {n: 0 for n in U.nodes}
    except Exception:
        core = {n: 0 for n in U.nodes}
    return degree, pr, core


def graph_payload(G, *, name="", path="", topic="", node_subset=None, edge_subset=None, include_attrs=True):
    comp, comp_sizes = _component_map(G)
    degree, pagerank, core = _centrality(G)
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
            "component": int(comp.get(sid, -1)),
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


def _set_graph(G, *, name, path="", topic=""):
    with LOCK:
        STATE["graph"] = G
        STATE["graph_id"] = uuid.uuid4().hex[:12]
        STATE["graph_name"] = name
        STATE["graph_path"] = path
        STATE["topic"] = topic
        return graph_payload(G, name=name, path=path, topic=topic, include_attrs=True)


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
    degree, pr, core = _centrality(G)
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

    api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY") or "x"
    base_url = cfg.get("base_url") or None
    client = OpenAI(base_url=base_url, api_key=api_key)
    kwargs = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": float(cfg.get("temperature", 0.3)),
        "max_completion_tokens": int(cfg.get("max_tokens", 1600)),
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
    temp = float(cfg.get("temperature", 0.3))
    gen = lm.generate(
        **enc,
        max_new_tokens=int(cfg.get("max_tokens", 1200)),
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
        depth=int(body.get("depth", 1)),
        max_nodes=int(body.get("max_nodes", 90)),
        max_edges=int(body.get("max_edges", 160)),
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
    proc = subprocess.Popen(cmd, cwd=IDEATION_DIR, stdout=log, stderr=subprocess.STDOUT, text=True)
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
    }
    with LOCK:
        STATE["jobs"][job_id] = job

    def wait():
        rc = proc.wait()
        log.close()
        with LOCK:
            job["status"] = "done" if rc == 0 else "failed"
            job["returncode"] = rc
            job["ended_at"] = time.time()

    threading.Thread(target=wait, daemon=True).start()
    return job


def _job_status(job_id):
    with LOCK:
        job = STATE["jobs"].get(job_id)
        if not job:
            raise ValueError("Unknown job id.")
        out = dict(job)
    try:
        out["log_tail"] = Path(out["log_path"]).read_text(errors="replace")[-8000:]
    except Exception:
        out["log_tail"] = ""
    run_dir = _resolve_run_path(out["out"])
    candidates = _graphml_candidates(run_dir) if run_dir.exists() and run_dir.is_dir() else []
    graph_path = candidates[0] if candidates else run_dir / "graph.graphml"
    out["graph_ready"] = bool(candidates)
    out["graph_path"] = str(graph_path)
    return out


class Handler(SimpleHTTPRequestHandler):
    server_version = "GraphExplorer/0.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

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
            elif parsed.path == "/api/job":
                self._json(_job_status((qs.get("id") or [""])[0]))
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
            elif parsed.path == "/api/ask":
                self._json(_answer_question(body))
            elif parsed.path == "/api/ideate":
                self._json(_start_ideate_job(body))
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
            raise SystemExit(f"[graph-explorer] {exc}") from None
        _set_graph(G, name=run.name, path=str(graph_path), topic=_load_topic(run, G))

    STATIC_DIR.mkdir(exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[graph-explorer] serving {url}")
    print("[graph-explorer] open the URL in a browser; Ctrl-C stops the server")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
