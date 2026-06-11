#!/usr/bin/env python
"""Mine a finished ideation graph for *novel* insights.

`ideate.py` accumulates the model's `<graph_json>` traces into one knowledge graph
(`runs/<exp>/graph.graphml`). That graph is more than the sum of the answers: the
*structure* it forms — paths, recurring relational motifs, brokered clusters,
feedback loops, latent links — encodes hypotheses the model never stated in any
single turn. This script exploits that structure to surface them.

Seven miners, each emitting ranked, human-readable candidate insights:

  1. conceptual_bridges   shortest reasoning chains linking semantically *distant*
                          concepts — multi-step arguments the model implies but never wrote.
  2. latent_links         link prediction (structural Adamic-Adar x semantic cosine):
                          relationships the graph "wants" but is missing.
  3. open_triads          directed transitivity gaps A->B->C with no A->C: inferable relations.
  4. relational_analogies recurring relation-typed motifs across disjoint concepts:
                          "A is to B as C is to D" structural analogies.
  5. feedback_loops       directed cycles = candidate feedback / self-reinforcing mechanisms.
  6. semantic_dissonance  concept pairs that are embedding-similar but graph-distant —
                          the model treats them as related yet never linked them.
  7. broker_ideas         high-betweenness, multi-community, low-constraint nodes:
                          interdisciplinary connectors where novelty concentrates.

Embeddings are re-derived offline from node labels (they aren't stored in graphml),
mirroring plot_ideation.py. Structural miners (3,4-fallback,5,7) run without them.

    python insights.py --run runs/exp2 --top 12 --out runs/exp2/insights
    python insights.py --run runs/exp2 --llm        # expand top insights via the generator

Writes <out>.json (structured), <out>.md (ranked report), and (unless --no-fig)
<out>_map.png/svg/pdf (semantic map with the top bridges drawn).
"""
import argparse
import itertools
import json
import os

import networkx as nx
import numpy as np


# --------------------------------------------------------------------------- #
#  Loading + embedding
# --------------------------------------------------------------------------- #
# Module-level iteration cap (set from --max-iter); truncates the mined graph to iter <= this
# so insights/novelty match plot_ideation when making fair cross-run figures. None = no cap.
MAX_ITER = None


def load_graph(run_dir):
    """Read <run>/graph.graphml, tolerating a still-running ideate.py (which truncates and
    rewrites it each step) by falling back to the newest stable graphml/iter_*.graphml snapshot.
    Truncates to MAX_ITER when set."""
    import glob
    from graphstore import cap_graph
    candidates = []
    gp = os.path.join(run_dir, "graph.graphml")
    if os.path.exists(gp) and os.path.getsize(gp) > 0:
        candidates.append(gp)
    candidates += sorted(glob.glob(os.path.join(run_dir, "graphml", "iter_*.graphml")))[::-1]
    G = None
    for path in candidates:
        try:
            if os.path.getsize(path) > 0:
                G = nx.read_graphml(path)
                break
        except Exception:
            continue
    if G is None:
        raise SystemExit(f"{gp}: no readable graph (empty/locked, and no graphml/ snapshot). "
                         f"Is the run still starting up?")
    G = cap_graph(G, MAX_ITER)
    if G.number_of_nodes() < 3:
        raise SystemExit(f"graph too small ({G.number_of_nodes()} nodes) to mine"
                         + (f" at iter <= {MAX_ITER}" if MAX_ITER is not None else "") + ".")
    return G


def lbl(G, n, k=64):
    s = str(G.nodes[n].get("label", n))
    return s if len(s) <= k else s[: k - 1] + "…"


def embed_nodes(G, model=None):
    """Re-embed node labels offline (batched: single model load + batched forward passes, with a
    tqdm bar for big graphs). Returns {node: unit-vec} or None. `model` selects the
    sentence-transformers id (default: graphstore.DEFAULT_EMBED_MODEL = embeddinggemma)."""
    try:
        from graphstore import embed_texts, DEFAULT_EMBED_MODEL
        ns = list(G)
        print(f"[insights] embedding {len(ns)} concepts (batched)…", flush=True)
        V = embed_texts([str(G.nodes[n].get("label", n)) for n in ns], model or DEFAULT_EMBED_MODEL)
        return {n: V[i] for i, n in enumerate(ns)}
    except Exception as e:
        print(f"[insights] embeddings unavailable ({e}); running structural miners only.")
        return None


def _cos(vecs, a, b):
    return float(np.dot(vecs[a], vecs[b]))


# --------------------------------------------------------------------------- #
#  Miners — each returns list[dict] with at least {kind, score, title, detail}
# --------------------------------------------------------------------------- #
def conceptual_bridges(G, vecs, top=10, min_len=2):
    """Shortest paths between semantically distant *connected* concepts: a path is a
    multi-hop chain of relations the model implies but never stated in one turn."""
    if not vecs:
        return []
    U = G.to_undirected()
    out = []
    # candidate endpoint pairs: most embedding-distant pairs that are still connected.
    # Sample random index pairs (vectorized) instead of materializing all O(N^2) combinations.
    nodes = list(G.nodes)
    N = len(nodes)
    X = np.stack([vecs[n] for n in nodes])
    rng = np.random.default_rng(0)
    M = min(20000, N * (N - 1) // 2)
    ii = rng.integers(0, N, 2 * M); jj = rng.integers(0, N, 2 * M)
    m = ii != jj
    ii, jj = ii[m][:M], jj[m][:M]
    dist_all = 1.0 - np.einsum("ij,ij->i", X[ii], X[jj])     # vectorized cosine distance
    order = np.argsort(dist_all)[::-1]                       # most distant first
    seen_pairs = 0
    for k in order:
        if seen_pairs >= top * 6:
            break
        seen_pairs += 1
        a, b, dist = nodes[ii[k]], nodes[jj[k]], float(dist_all[k])
        if not nx.has_path(U, a, b):
            continue
        path = nx.shortest_path(U, a, b)
        if len(path) - 1 < min_len:
            continue
        rels = []
        for u, v in zip(path, path[1:]):
            r = (G.get_edge_data(u, v) or G.get_edge_data(v, u) or {}).get("relation", "related_to")
            rels.append(r)
        chain = "  ".join(f"[{lbl(G, n, 28)}]" + (f" --{rels[i]}-->" if i < len(rels) else "")
                          for i, n in enumerate(path))
        out.append({
            "kind": "conceptual_bridge",
            "score": float(dist * (len(path) - 1)),    # distant AND non-trivial chain
            "title": f"{lbl(G, a, 32)}  ⇝  {lbl(G, b, 32)}",
            "detail": f"distance={dist:.2f}, hops={len(path)-1}\n    {chain}",
            "endpoints": [a, b], "path": list(path), "relations": rels,
            "embed_distance": float(dist), "hops": len(path) - 1,
        })
        if len(out) >= top:
            break
    return out


def latent_links(G, vecs, top=12, alpha=0.5):
    """Link prediction over non-edges: combine a structural score (Adamic-Adar, common
    neighbours) with semantic cosine. High = a relationship the graph implies but lacks."""
    U = nx.Graph(G.to_undirected())
    U.remove_edges_from(nx.selfloop_edges(U))
    eset = {frozenset(e) for e in U.edges}
    # Only pairs sharing >=1 neighbour have nonzero Adamic-Adar, so enumerate candidates from
    # each node's neighbour pairs (O(sum deg^2)) instead of all O(N^2) non-edges. Cap hub fan-out
    # and the total candidate set to stay fast on large graphs.
    cand, CAP, HUB = set(), 200000, 200
    for w in U:
        nb = list(U[w])
        if len(nb) > HUB:
            nb = nb[:HUB]
        for a, b in itertools.combinations(nb, 2):
            fs = frozenset((a, b))
            if a != b and fs not in eset:
                cand.add(fs)
        if len(cand) >= CAP:
            break
    try:
        aa = {frozenset((u, v)): p
              for u, v, p in nx.adamic_adar_index(U, [tuple(fs) for fs in cand])}
    except Exception:
        aa = {}
    if not aa:
        return []
    aa_max = max(aa.values()) or 1.0
    out = []
    for fs, a_struct in aa.items():
        u, v = tuple(fs)
        s_struct = a_struct / aa_max
        s_sem = (_cos(vecs, u, v) if vecs else 0.0)
        score = alpha * s_struct + (1 - alpha) * max(0.0, s_sem) if vecs else s_struct
        cn = len(list(nx.common_neighbors(U, u, v)))
        out.append({
            "kind": "latent_link",
            "score": float(score),
            "title": f"{lbl(G, u, 32)}  ⟷  {lbl(G, v, 32)}",
            "detail": f"structural={s_struct:.2f} (AA, {cn} shared), semantic={s_sem:.2f}"
                      + ("" if vecs else "  [structural only]"),
            "pair": [u, v], "common_neighbors": cn,
            "struct": float(s_struct), "semantic": float(s_sem),
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top]


def open_triads(G, vecs, top=12):
    """Directed transitivity gaps: A--r1-->B--r2-->C with no A-->C. The graph 'wants'
    a relation A-->C. Rank by semantic plausibility of the inferred endpoint pair."""
    out = []
    seen = set()
    for b in G.nodes:
        preds = list(G.predecessors(b))
        succs = list(G.successors(b))
        for a in preds:
            for c in succs:
                if a == c or G.has_edge(a, c) or (a, c) in seen:
                    continue
                seen.add((a, c))
                r1 = G[a][b].get("relation", "related_to")
                r2 = G[b][c].get("relation", "related_to")
                sem = (max(0.0, _cos(vecs, a, c)) if vecs else 0.0)
                out.append({
                    "kind": "open_triad",
                    "score": float((sem if vecs else 0.5) + 0.0),
                    "title": f"infer:  {lbl(G, a, 26)}  ⟶?  {lbl(G, c, 26)}",
                    "detail": f"via [{lbl(G, b, 24)}]:  --{r1}--> · --{r2}-->"
                              + (f"   (endpoint sim={sem:.2f})" if vecs else ""),
                    "chain": [a, b, c], "relations": [r1, r2], "semantic": float(sem),
                })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top]


def relational_analogies(G, vecs, top=10):
    """Recurring relation-typed 2-paths (A--r1-->B--r2-->C) shared by node-disjoint
    instances => 'A:B::C:D'-style structural analogies. Scored by embedding parallelism
    of the two relation steps when embeddings are available, else by motif frequency."""
    # collect 2-path instances keyed by (r1, r2) relation signature
    sig = {}
    for b in G.nodes:
        for a in G.predecessors(b):
            r1 = G[a][b].get("relation", "related_to")
            for c in G.successors(b):
                if c == a:
                    continue
                r2 = G[b][c].get("relation", "related_to")
                sig.setdefault((r1, r2), []).append((a, b, c))
    out = []
    MAX_INST = 60                                   # cap per-signature instances (pairs are O(n^2))
    for (r1, r2), insts in sig.items():
        if len(insts) < 2:
            continue
        if len(insts) > MAX_INST:
            insts = insts[:MAX_INST]
        # compare disjoint instance pairs
        for i in range(len(insts)):
            for j in range(i + 1, len(insts)):
                A, B, C = insts[i]
                D, E, F = insts[j]
                if len({A, B, C} & {D, E, F}) > 0:
                    continue                       # require node-disjoint analogues
                if vecs:
                    p1 = vecs[B] - vecs[A]; q1 = vecs[E] - vecs[D]
                    p2 = vecs[C] - vecs[B]; q2 = vecs[F] - vecs[E]
                    def u(x):
                        n = np.linalg.norm(x)
                        return x / n if n > 1e-9 else x
                    par = 0.5 * (float(np.dot(u(p1), u(q1))) + float(np.dot(u(p2), u(q2))))
                    novelty = 1.0 - max(_cos(vecs, A, D), _cos(vecs, C, F))
                    score = par * max(0.0, novelty)
                else:
                    par, novelty, score = float("nan"), float("nan"), float(len(insts))
                out.append({
                    "kind": "relational_analogy",
                    "score": float(score),
                    "title": f"{lbl(G, A, 18)}:{lbl(G, C, 18)}  ::  {lbl(G, D, 18)}:{lbl(G, F, 18)}",
                    "detail": f"shared motif  --{r1}--> · --{r2}-->"
                              + (f"   (parallelism={par:.2f}, novelty={novelty:.2f})"
                                 if vecs else f"   (motif seen {len(insts)}x)"),
                    "instances": [[A, B, C], [D, E, F]], "signature": [r1, r2],
                    "parallelism": (None if not vecs else float(par)),
                })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top]


def feedback_loops(G, vecs, top=10, max_len=6, max_cycles=4000):
    """Directed simple cycles = candidate feedback / self-reinforcing mechanisms
    (especially meaningful for self-healing / homeostatic systems). Enumeration is capped —
    dense graphs can contain exponentially many cycles."""
    try:
        gen = nx.simple_cycles(G, length_bound=max_len)
    except TypeError:                              # older networkx: no length_bound
        gen = (c for c in nx.simple_cycles(G) if len(c) <= max_len)
    cycles = []
    for c in gen:                                   # cap: don't materialize an exploding generator
        cycles.append(c)
        if len(cycles) >= max_cycles:
            break
    out = []
    for cyc in cycles:
        if len(cyc) < 2:
            continue
        rels = [G[cyc[k]][cyc[(k + 1) % len(cyc)]].get("relation", "related_to")
                for k in range(len(cyc))]
        # prefer shorter, semantically-coherent loops (a tight feedback motif)
        coh = (np.mean([_cos(vecs, cyc[k], cyc[(k + 1) % len(cyc)])
                        for k in range(len(cyc))]) if vecs else 0.5)
        score = float(coh) / len(cyc)
        chain = " -> ".join(f"[{lbl(G, n, 22)}]" for n in cyc) + " -> ↺"
        out.append({
            "kind": "feedback_loop",
            "score": score,
            "title": f"loop({len(cyc)}):  " + " → ".join(lbl(G, n, 16) for n in cyc),
            "detail": f"relations={rels}\n    {chain}",
            "cycle": list(cyc), "relations": rels, "length": len(cyc),
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top]


def semantic_dissonance(G, vecs, top=12, sim_thr=0.55, min_hops=3):
    """Concept pairs that are embedding-similar but graph-distant (>= min_hops apart or
    in different components): the model treats them as related yet never connected them."""
    if not vecs:
        return []
    U = G.to_undirected()
    nodes = list(G.nodes)
    N = len(nodes)
    X = np.stack([vecs[n] for n in nodes])
    rng = np.random.default_rng(1)
    # sample random index pairs (vectorized), keep the embedding-similar ones, then probe graph
    # distance only for those (capped) — avoids materializing all O(N^2) pairs + N^2 BFS calls.
    M = min(40000, N * (N - 1) // 2)
    ii = rng.integers(0, N, 2 * M); jj = rng.integers(0, N, 2 * M)
    m = ii != jj
    ii, jj = ii[m][:M], jj[m][:M]
    sims = np.einsum("ij,ij->i", X[ii], X[jj])
    hi = np.where(sims >= sim_thr)[0]
    hi = hi[np.argsort(sims[hi])[::-1]]                  # most-similar first
    out, checked = [], 0
    for k in hi:
        if checked >= max(2000, top * 50):              # cap expensive BFS probes
            break
        checked += 1
        a, b, s = nodes[ii[k]], nodes[jj[k]], float(sims[k])
        if nx.has_path(U, a, b):
            hops = nx.shortest_path_length(U, a, b)
            if hops < min_hops:
                continue
        else:
            hops = None                            # different components — maximally dissonant
        score = s * (1.0 + (0.5 if hops is None else min(hops, 8) / 8.0))
        out.append({
            "kind": "semantic_dissonance",
            "score": float(score),
            "title": f"{lbl(G, a, 32)}  ≈  {lbl(G, b, 32)}",
            "detail": f"cosine={s:.2f} but graph distance="
                      + ("∞ (separate clusters)" if hops is None else f"{hops} hops")
                      + " — related in meaning, unlinked in reasoning.",
            "pair": [a, b], "semantic": float(s),
            "graph_hops": (None if hops is None else int(hops)),
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top]


def broker_ideas(G, vecs, top=10):
    """High-betweenness, multi-community, low-constraint nodes: the interdisciplinary
    connectors where recombination/novelty concentrates (Burt's structural holes)."""
    U = G.to_undirected()
    nN = G.number_of_nodes()
    if nN <= 3:
        bet = {n: 0.0 for n in G}
    elif nN > 1500:                                # approx betweenness — exact is minutes on ~7k nodes
        print(f"  broker_ideas: approx betweenness (k=400) on {nN} nodes…", flush=True)
        bet = nx.betweenness_centrality(G, k=min(400, nN), seed=0)
    else:
        bet = nx.betweenness_centrality(G)
    try:
        comms = list(nx.community.greedy_modularity_communities(U))
        cmap = {n: i for i, c in enumerate(comms) for n in c}
    except Exception:
        cmap = {n: 0 for n in G}
    try:
        constraint = nx.constraint(U)             # Burt: low => spans structural holes
    except Exception:
        constraint = {n: float("nan") for n in G}
    out = []
    for n in G.nodes:
        nbr_comms = {cmap.get(m, -1) for m in U.neighbors(n)}
        ncomm = len(nbr_comms)
        if ncomm < 2 and bet[n] <= 0:
            continue
        c = constraint.get(n, float("nan"))
        cterm = 0.0 if (c != c) else (1.0 - min(c, 1.0))   # nan-safe
        score = bet[n] * (1.0 + ncomm) + 0.2 * cterm
        out.append({
            "kind": "broker_idea",
            "score": float(score),
            "title": lbl(G, n, 40),
            "detail": f"betweenness={bet[n]:.3f}, bridges {ncomm} idea-clusters, "
                      f"Burt constraint={c:.2f}" if c == c else
                      f"betweenness={bet[n]:.3f}, bridges {ncomm} idea-clusters",
            "node": n, "betweenness": float(bet[n]), "n_clusters_bridged": int(ncomm),
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top]


# (canonical_kind, display_header, miner_fn). canonical_kind matches the "kind" each
# miner stamps on its insight dicts — used as the stable key everywhere downstream.
MINERS = [
    ("conceptual_bridge", "Conceptual bridges (implied multi-step arguments)", conceptual_bridges),
    ("latent_link", "Latent links (relationships the graph is missing)", latent_links),
    ("open_triad", "Open triads (inferable A→C relations)", open_triads),
    ("relational_analogy", "Relational analogies (A:B :: C:D motifs)", relational_analogies),
    ("feedback_loop", "Feedback loops (self-reinforcing mechanisms)", feedback_loops),
    ("semantic_dissonance", "Semantic dissonance (similar but unlinked)", semantic_dissonance),
    ("broker_idea", "Broker ideas (interdisciplinary connectors)", broker_ideas),
]
KIND_ORDER = [k for k, _, _ in MINERS]
KIND_HEADER = {k: h for k, h, _ in MINERS}


def mine_all(G, vecs, top=10, log=False):
    """Run every miner and return [(kind, [insight, ...]), ...] in MINERS order.
    Reusable entry point for other tools (e.g. synthesize.py)."""
    results = []
    for kind, header, fn in MINERS:
        ins = fn(G, vecs, top=top)
        results.append((kind, ins))
        if log:
            print(f"  {header:<52s} {len(ins):>3d} candidates")
    return results


def load_insights_or_mine(run_dir, top=10, want_mine=False, embed_model=None):
    """Load a previously-written <run>/insights.json, or mine fresh if missing/requested.
    Returns (topic, G, results) where results is the mine_all() shape. `embed_model` (or the
    model recorded in the run's summary.json) is used when mining fresh."""
    G = load_graph(run_dir)
    topic = ""
    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        try:
            topic = json.load(open(sp)).get("topic", "")
        except Exception:
            pass
    jp = os.path.join(run_dir, "insights.json")
    if os.path.exists(jp) and not want_mine and MAX_ITER is None:   # cached json is uncapped
        data = json.load(open(jp))
        topic = topic or data.get("topic", "")
        miners = data.get("miners", {})
        results = [(k, miners.get(k, [])) for k in KIND_ORDER]   # always all 7, in order
        return topic, G, results
    from graphstore import resolve_embed_model
    vecs = embed_nodes(G, resolve_embed_model(run_dir, embed_model))
    return topic, G, mine_all(G, vecs, top=top, log=True)


# --------------------------------------------------------------------------- #
#  Optional LLM expansion of the top insights into concrete hypotheses
# --------------------------------------------------------------------------- #
def llm_expand(results, G, topic, config_path, per_kind=2):
    """Turn the highest-scoring structural insights into testable hypotheses using the
    generator (same model/endpoint as the loop). Best-effort; returns list of dicts."""
    try:
        import yaml
        from clients import Clients
        cfg = yaml.safe_load(open(config_path))
        clients = Clients(cfg)
    except Exception as e:
        print(f"[insights] --llm skipped (client unavailable: {e})")
        return []
    picks = []
    for _, ins in results:
        picks.extend(ins[:per_kind])
    expanded = []
    for ins in picks:
        prompt = (f"Topic: {topic}\n\nThe following non-obvious connection was mined from a "
                  f"knowledge graph built while reasoning about this topic:\n\n"
                  f"{ins['title']}\n{ins['detail']}\n\n"
                  f"State ONE specific, testable scientific hypothesis this connection suggests, "
                  f"and a one-sentence rationale. Be concrete.")
        try:
            out = clients.generate(prompt, previous_id=None)
            from parse import parse_trace
            ans = parse_trace(out["full"])["answer"]
        except Exception as e:
            ans = f"(generation failed: {e})"
        expanded.append({"insight": ins["title"], "kind": ins["kind"], "hypothesis": ans})
    return expanded


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def write_markdown(path, run_dir, topic, G, results, expanded):
    lines = [f"# Mined insights — {topic or os.path.basename(run_dir)}",
             "",
             f"Source graph: `{run_dir}/graph.graphml` "
             f"({G.number_of_nodes()} ideas, {G.number_of_edges()} links)",
             "",
             "Each section lists the highest-scoring candidates from one miner. These are "
             "*hypotheses generated from graph structure*, not asserted facts — they point at "
             "connections worth investigating or feeding back into the ideation loop.",
             ""]
    for kind, ins in results:
        lines.append(f"## {KIND_HEADER.get(kind, kind)}")
        if not ins:
            lines.append("_none found (needs embeddings or a denser graph)._\n")
            continue
        for k, x in enumerate(ins, 1):
            lines.append(f"{k}. **{x['title']}**  _(score {x['score']:.3f})_  ")
            lines.append(f"   {x['detail']}")
        lines.append("")
    if expanded:
        lines.append("## LLM-expanded hypotheses (top insights → testable claims)")
        for e in expanded:
            lines.append(f"- **[{e['kind']}]** {e['insight']}")
            lines.append(f"  - {e['hypothesis']}")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def insight_map(base, G, vecs, results):
    """Semantic PCA map of ideas with the top conceptual bridges drawn over it."""
    import matplotlib.pyplot as plt
    if not vecs:
        print("  (insight map skipped: no embeddings)")
        return
    nodes = list(G.nodes)
    X = np.stack([vecs[n] for n in nodes]); Xc = X - X.mean(0)
    try:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False); P = Xc @ Vt[:2].T
    except Exception:
        P = Xc[:, :2]
    pos = {n: P[i] for i, n in enumerate(nodes)}
    try:
        pr = nx.pagerank(G) if G.number_of_edges() else {n: 1.0 for n in G}
    except Exception:
        pr = {n: 1.0 / max(1, len(nodes)) for n in G}
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    for u, v in G.to_undirected().edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="0.85", lw=0.5, zorder=1)
    ax.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes],
               s=[20 + 3000 * pr[n] for n in nodes], color="#1f77b4", alpha=0.6,
               linewidths=0, zorder=2)
    bridges = next((ins for kind, ins in results if kind == "conceptual_bridge"), [])
    colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for i, b in enumerate(bridges[:5]):
        pth = b["path"]
        xs = [pos[n][0] for n in pth]; ys = [pos[n][1] for n in pth]
        ax.plot(xs, ys, color=colors[i % len(colors)], lw=2.2, zorder=3,
                label=f"{lbl(G, pth[0], 18)} ⇝ {lbl(G, pth[-1], 18)}")
        for n in (pth[0], pth[-1]):
            ax.annotate(lbl(G, n, 20), pos[n], fontsize=7, zorder=4,
                        xytext=(3, 3), textcoords="offset points")
    ax.legend(fontsize=7, frameon=False, loc="best")
    ax.set_title("Idea landscape + top conceptual bridges")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{base}_map.{ext}", bbox_inches="tight")
    print(f"wrote {base}_map.png/.svg/.pdf")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run dir produced by ideate.py")
    p.add_argument("--out", default=None, help="output basename (default: <run>/insights)")
    p.add_argument("--top", type=int, default=10, help="candidates per miner")
    p.add_argument("--no-fig", action="store_true", help="skip the insight-map figure")
    p.add_argument("--llm", action="store_true",
                   help="expand top insights into hypotheses via the generator (uses config)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--embed-model", dest="embed_model", default=None,
                   help="sentence-transformers id for re-embedding "
                        "(default: model recorded by the run, else embeddinggemma-300m)")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=None,
                   help="truncate the graph to iter <= this before mining (fair cross-run cutoff)")
    args = p.parse_args()
    out = args.out or os.path.join(args.run.rstrip("/"), "insights")

    global MAX_ITER
    MAX_ITER = args.max_iter
    if MAX_ITER is not None:
        print(f"[insights] truncating to iter <= {MAX_ITER}")

    G = load_graph(args.run)
    topic = ""
    sp = os.path.join(args.run, "summary.json")
    if os.path.exists(sp):
        try:
            topic = json.load(open(sp)).get("topic", "")
        except Exception:
            pass
    from graphstore import resolve_embed_model
    model = resolve_embed_model(args.run, args.embed_model)
    vecs = embed_nodes(G, model)
    print(f"[insights] mining {G.number_of_nodes()} ideas / {G.number_of_edges()} links"
          + (f"  (embed: {model})" if vecs else "  (structural miners only — no embeddings)"))

    results = mine_all(G, vecs, top=args.top, log=True)

    expanded = llm_expand(results, G, topic, args.config) if args.llm else []

    # structured JSON
    payload = {"run": args.run, "topic": topic,
               "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
               "miners": {kind: ins for kind, ins in results},
               "llm_expanded": expanded}
    with open(f"{out}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out}.json")
    write_markdown(f"{out}.md", args.run, topic, G, results, expanded)
    if not args.no_fig:
        insight_map(out, G, vecs, results)


if __name__ == "__main__":
    main()
