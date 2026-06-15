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


def _quoted_topic(topic):
    quote = '"' if "'" in topic else "'"
    return f"{quote}{topic}{quote}"


def _q(text, topic):
    return text if topic.lower() in text.lower() else f"In the context of {_quoted_topic(topic)}: {text}"


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


# ---- conversational expansion: LLM questioner reasons over (question + answer) to open NEW
#      directions the graph-structure strategies can't reach (they only probe existing nodes,
#      so they saturate). The questioner can introduce concepts not yet in the graph. ----
def converse_driven(ctx):
    """Feed the ORIGINAL question + the model's prose answer to the questioner LLM (e.g. a
    Llama-instruct, set via `questioner.model`) and ask for genuinely NEW questions that open an
    unexplored direction — an implication, a tension/contradiction, a cross-domain analogy, or a
    deeper mechanism. Because the questioner reasons over content (not graph structure), it can
    leave a saturated region. Cost: 2 LLM calls/step (generator + questioner), like `answer`."""
    k = ctx["cfg"]["fanout"]
    topic = ctx["topic"]
    prompt = (f"Topic: {topic}\n\nOriginal question:\n{ctx['question']}\n\n"
              f"Answer:\n{ctx['parse']['answer']}\n\n"
              f"Pose {k} NEW, specific question(s) that open an UNEXPLORED direction this answer "
              f"hints at but did not address — an implication, a tension or contradiction, a "
              f"cross-domain analogy, or a deeper underlying mechanism. Do NOT restate or merely "
              f"narrow the original question; aim for new territory. **Every question must stay "
              f"directly relevant to '{topic}' and advance understanding of it — no tangents into "
              f"unrelated fields.** One per line, no numbering.")
    text, _ = ctx["clients"].ask(prompt)
    qs = [ln.strip("-•*1234567890. ").strip() for ln in text.splitlines() if ln.strip()][:k]
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
    queried = ctx.get("queried", set())
    deg = dict(G.degree())
    pool = [n for n in deg if n not in queried] or list(deg)              # prefer UNvisited
    leaves = sorted(pool, key=deg.get)[:max(1, ctx["cfg"]["fanout"])]     # lowest-degree first
    try:
        bc = nx.betweenness_centrality(G) if G.number_of_nodes() > 3 else {}
        hubs = [n for n in sorted(bc, key=bc.get, reverse=True) if n not in queried][:1]
    except Exception:
        hubs = []
    targets, seen = [], set()
    for n in leaves + hubs:
        if n not in seen:
            seen.add(n); targets.append(n)
    return [{"q": _q(f"What are the key unresolved questions and underlying mechanisms "
                     f"concerning '{t}'?", ctx["topic"]), "anchor": t}
            for t in targets[: ctx["cfg"]["fanout"]]]


# ---- novelty: steer toward under-explored embedding regions ----
def novelty_driven(ctx):
    vecs = ctx["store"].node_vectors()
    if len(vecs) < 3:
        return node_driven(ctx)
    queried = ctx.get("queried", set())
    centroid = np.mean([vecs[i] for i in vecs], axis=0)
    ids = [i for i in vecs if i not in queried] or list(vecs)             # prefer UNvisited
    far = sorted(ids, key=lambda i: float(np.dot(vecs[i], centroid)))     # least central
    return [{"q": _q(f"What is an unconventional or overlooked aspect of '{t}', and why "
                     f"might it matter?", ctx["topic"]), "anchor": t}
            for t in far[: ctx["cfg"]["fanout"]]]


# ---- leap: aggressive exploration — jump OUTSIDE the known idea space ----
def leap_driven(ctx):
    """Divergent search. Where `novelty` drifts to the edge of what's known (re-examining the
    single most-peripheral node), `leap` pushes *past* it, two ways:
      (1) distant recombination — pair the most embedding-DISSIMILAR concepts and force a
          mechanism linking them (a creative jump across the space, not within a neighborhood);
      (2) cross-domain transfer — import a principle from an unrelated field onto a peripheral
          concept, injecting genuinely external ideas the graph doesn't yet contain.
    Both are standalone, self-contained questions (in-distribution for the single-turn model)."""
    store = ctx["store"]
    vecs = store.node_vectors()
    topic, k = ctx["topic"], ctx["cfg"]["fanout"]
    queried = ctx.get("queried", set())
    if len(vecs) < 3:
        return novelty_driven(ctx)
    ids = list(vecs)
    centroid = np.mean([vecs[i] for i in ids], axis=0)
    far = sorted(ids, key=lambda i: float(np.dot(vecs[i], centroid)))      # least central first
    far = [n for n in far if n not in queried] or far                      # prefer UNvisited
    cands = []
    # (1) recombine each peripheral seed with its most dissimilar partner anywhere in the graph
    for a in far[: max(1, k // 2)]:
        b = min((j for j in ids if j != a),
                key=lambda j: float(np.dot(vecs[a], vecs[j])), default=None)
        if b is None:
            continue
        cands.append({"q": _q(f"What radically new approach to {topic} could emerge by combining "
                              f"'{a}' and '{b}' — two ideas that seem unrelated? Propose a concrete "
                              f"mechanism that links them.", topic), "anchor": a})
    # (2) import an outside principle onto the most peripheral concepts
    for t in far[:k]:
        cands.append({"q": _q(f"What principle or mechanism from a completely different field "
                              f"could transform how we approach '{t}', and what novel, untested "
                              f"idea would that suggest?", topic), "anchor": t})
    out, seen = [], set()
    for c in cands:                                                        # dedup, keep variety
        if c["q"] not in seen:
            seen.add(c["q"]); out.append(c)
    return out[:k]


_REGISTRY = {
    "node": node_driven, "answer": answer_driven, "edge": edge_driven,
    "frontier": frontier_driven, "novelty": novelty_driven, "leap": leap_driven,
    "converse": converse_driven,
}


def mixed(ctx):
    """Round-robin across all strategies by iteration."""
    order = ["frontier", "node", "edge", "novelty", "leap"]
    fn = _REGISTRY[order[ctx["iter"] % len(order)]]
    return fn(ctx)


def get_strategy(name):
    if name == "mixed":
        return mixed
    if name not in _REGISTRY:
        raise SystemExit(f"unknown strategy '{name}'. choices: {list(_REGISTRY)+['mixed']}")
    return _REGISTRY[name]
