"""Question-generation strategies (the 'expansion policy').

Each strategy is f(ctx) -> list[Candidate], where
    Candidate = {"q": str, "anchor": canonical_node_id or None}
and ctx exposes: topic, question, parse (parsed trace), store (GraphStore),
new_nodes (list of new canonical ids this step), clients, cfg.

The loop turns each Candidate into a generator call; `anchor` is used by the
branched context-mode to chain off the response that introduced that node.
"""
import numpy as np
import networkx as nx


def _q(text, topic):
    return text if topic.lower() in text.lower() else f"In the context of {topic}: {text}"


# ---- breadth: one question per newly discovered node ----
def node_driven(ctx):
    return [{"q": _q(f"By what mechanism does '{n}' operate, and what does it depend on?", ctx["topic"]),
             "anchor": n} for n in ctx["new_nodes"]][: ctx["cfg"]["fanout"]]


# ---- depth: ask the questioner LLM for follow-ups to the latest answer ----
def answer_driven(ctx):
    k = ctx["cfg"]["fanout"]
    prompt = (f"Topic: {ctx['topic']}\n\nAnswer so far:\n{ctx['parse']['answer']}\n\n"
              f"Propose {k} specific, non-overlapping follow-up questions that would deepen or "
              f"expand this into new territory. One per line, no numbering.")
    text, _ = ctx["clients"].ask(prompt)
    qs = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()][:k]
    return [{"q": q, "anchor": None} for q in qs]


# ---- densify: probe likely-but-missing links (embedding-close, unconnected) ----
def edge_driven(ctx):
    G, vecs = ctx["store"].G, ctx["store"].node_vectors()
    ids = list(vecs)
    cands = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if G.has_edge(a, b) or G.has_edge(b, a):
                continue
            sim = float(np.dot(vecs[a], vecs[b]))
            if sim >= ctx["cfg"].get("link_sim", 0.55):
                cands.append((sim, a, b))
    cands.sort(reverse=True)
    return [{"q": _q(f"How are '{a}' and '{b}' related, and what connects them?", ctx["topic"]),
             "anchor": a} for _, a, b in cands[: ctx["cfg"]["fanout"]]]


# ---- graph-analytic: expand frontier leaves + deepen hubs (RECOMMENDED) ----
def frontier_driven(ctx):
    G = ctx["store"].G
    if G.number_of_nodes() == 0:
        return []
    deg = dict(G.degree())
    leaves = sorted(deg, key=deg.get)[:max(1, ctx["cfg"]["fanout"])]      # grow outward
    try:
        bc = nx.betweenness_centrality(G) if G.number_of_nodes() > 3 else {}
        hubs = sorted(bc, key=bc.get, reverse=True)[:1]                   # deepen a hub
    except Exception:
        hubs = []
    targets, seen = [], set()
    for n in leaves + hubs:
        if n not in seen:
            seen.add(n); targets.append(n)
    return [{"q": _q(f"What is not yet explained about '{t}'?", ctx["topic"]), "anchor": t}
            for t in targets[: ctx["cfg"]["fanout"]]]


# ---- novelty: steer toward under-explored embedding regions ----
def novelty_driven(ctx):
    vecs = ctx["store"].node_vectors()
    if len(vecs) < 3:
        return node_driven(ctx)
    ids = list(vecs)
    centroid = np.mean([vecs[i] for i in ids], axis=0)
    far = sorted(ids, key=lambda i: float(np.dot(vecs[i], centroid)))     # least central
    return [{"q": _q(f"Explore an unconventional angle on '{t}'.", ctx["topic"]), "anchor": t}
            for t in far[: ctx["cfg"]["fanout"]]]


_REGISTRY = {
    "node": node_driven, "answer": answer_driven, "edge": edge_driven,
    "frontier": frontier_driven, "novelty": novelty_driven,
}


def mixed(ctx):
    """Round-robin across all strategies by iteration."""
    order = ["frontier", "node", "edge", "novelty"]
    fn = _REGISTRY[order[ctx["iter"] % len(order)]]
    return fn(ctx)


def get_strategy(name):
    if name == "mixed":
        return mixed
    if name not in _REGISTRY:
        raise SystemExit(f"unknown strategy '{name}'. choices: {list(_REGISTRY)+['mixed']}")
    return _REGISTRY[name]
