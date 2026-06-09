"""Parse Graph-PRefLexOR output into (graph, patterns, answer).

Self-contained: mirrors the sentinel tags used by the reward function in
src/run_grpo_graph.py so parsing matches training. Works whether the model's
<think> block comes back inside `output_text` or a separate `reasoning` field
(we always parse the concatenated text).
"""
import json
import re
import networkx as nx

THINK_START, THINK_END = "<think>", "</think>"
GRAPH_JSON_START, GRAPH_JSON_END = "<graph_json>", "</graph_json>"
PATTERNS_START, PATTERNS_END = "<patterns>", "</patterns>"


def _between(s, a, b):
    i = s.find(a)
    if i == -1:
        return None
    j = s.find(b, i + len(a))
    if j == -1:
        return None
    return s[i + len(a):j]


def extract_graph_json(text):
    """Return the parsed graph_json dict ({'nodes':[...], 'edges':[...]}) or None."""
    region = _between(text, THINK_START, THINK_END)
    region = region if region is not None else text
    raw = _between(region, GRAPH_JSON_START, GRAPH_JSON_END) or \
        _between(text, GRAPH_JSON_START, GRAPH_JSON_END)
    if raw is None:
        return None
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    return obj if isinstance(obj, dict) and "nodes" in obj else None


def to_networkx(obj):
    G = nx.DiGraph()
    if not obj:
        return G
    for n in obj.get("nodes", []):
        nid = n.get("id") if isinstance(n, dict) else n
        if nid:
            G.add_node(str(nid).strip())
    for e in obj.get("edges", []):
        if not isinstance(e, dict):
            continue
        s = str(e.get("source", "")).strip()
        t = str(e.get("target", "")).strip()
        rel = str(e.get("relation", "related_to")).strip()
        if s and t:
            G.add_node(s)
            G.add_node(t)
            G.add_edge(s, t, relation=rel)
    return G


def parse_trace(text):
    """text = concatenated reasoning + output. Returns dict with graph, answer, patterns."""
    obj = extract_graph_json(text)
    answer = text.split(THINK_END)[-1].strip() if THINK_END in text else text.strip()
    patterns = _between(text, PATTERNS_START, PATTERNS_END) or ""
    return {
        "graph": to_networkx(obj),
        "graph_json": obj,
        "answer": answer.strip(),
        "patterns": patterns.strip(),
    }
