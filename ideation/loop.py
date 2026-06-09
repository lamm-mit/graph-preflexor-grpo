"""The ideation loop: seed topic -> generate -> accumulate graph -> expand.

Budget = whichever of generator_calls / max_tokens / max_iters hits first, plus a
novelty-based early stop. Context modes (Responses API `previous_response_id`):
  fresh    -> every question independent
  chained  -> chain off the previous response
  branched -> chain off the response that introduced the anchor node
"""
import heapq
from collections import deque

from parse import parse_trace
from metrics import semantic_metrics


def _parent_id(mode, cand, node_origin, last_id):
    if mode == "fresh":
        return None
    if mode == "chained":
        return last_id
    if mode == "branched":
        a = cand.get("anchor")
        return node_origin.get(a) if a else last_id
    raise SystemExit(f"unknown context_mode '{mode}'")


def _score(cand, store):
    """Prioritize anchors that are novel / peripheral; deprioritize well-covered ones."""
    a = cand.get("anchor")
    if not a or a not in store.G:
        return 1.0
    deg = store.G.degree(a)
    return 1.0 / (1.0 + deg)            # low-degree (frontier) first


def run(clients, cfg, topic, strategy_fn, on_step=None):
    from graphstore import GraphStore, make_embedder
    store = GraphStore(make_embedder(cfg["embed_model"]), cfg["dedup_threshold"])

    budget, nstop = cfg["budget"], cfg["novelty_stop"]
    heap, counter, seen_q = [], 0, set()
    node_origin, last_id = {}, None
    recent_new = deque(maxlen=nstop["window"])
    transcript = []

    def push(cand, prio):
        nonlocal counter
        heapq.heappush(heap, (-prio, counter, cand)); counter += 1

    push({"q": topic, "anchor": None}, 1.0)
    seen_q.add(topic)
    calls = tokens = it = 0

    while heap and calls < budget["generator_calls"] and tokens < budget["max_tokens"] \
            and it < budget["max_iters"]:
        _, _, cand = heapq.heappop(heap)
        q = cand["q"]
        parent = _parent_id(cfg["context_mode"], cand, node_origin, last_id)
        out = clients.generate(q, previous_id=parent)
        calls += 1; tokens += out["usage"]; last_id = out["id"]

        p = parse_trace(out["full"])
        new_nodes = store.merge(p["graph"], prov={"question": q, "iter": it,
                                                  "response_id": out["id"]})
        for cid in new_nodes:
            node_origin.setdefault(cid, out["id"])
        recent_new.append(len(new_nodes))

        rec = {"iter": it, "question": q, "parent": parent, "response_id": out["id"],
               "answer": p["answer"], "new_nodes": new_nodes,
               "n_nodes": store.G.number_of_nodes(), "n_edges": store.G.number_of_edges(),
               "tokens": out["usage"], "cum_tokens": tokens,
               "diversity": semantic_metrics(store)["mean_pairwise_distance"]}
        transcript.append(rec)
        if on_step:
            on_step(rec)

        ctx = {"topic": topic, "question": q, "parse": p, "store": store,
               "new_nodes": new_nodes, "clients": clients, "cfg": cfg, "iter": it}
        for c in strategy_fn(ctx):
            if not c["q"] or c["q"] in seen_q:
                continue
            seen_q.add(c["q"])
            push(c, _score(c, store))

        it += 1
        if len(recent_new) == recent_new.maxlen and sum(recent_new) < nstop["min_new_nodes"] * recent_new.maxlen:
            break   # novelty converged

    return store, transcript, {"calls": calls, "tokens": tokens, "iters": it}
