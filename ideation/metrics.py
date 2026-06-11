"""Ideation / creativity metrics over the accumulated graph.

Graph dynamics + semantic diversity (embeddings) + simple creativity proxies
(fluency / flexibility / elaboration). Originality vs a reference set and the
LLM-judge eval live in compare.py.
"""
import math
import numpy as np
import networkx as nx


def graph_metrics(store):
    G = store.G
    n, m = G.number_of_nodes(), G.number_of_edges()
    U = G.to_undirected()
    comps = list(nx.connected_components(U)) if n else []
    largest = max((len(c) for c in comps), default=0)
    out = {
        "nodes": n, "edges": m,
        "density": nx.density(G) if n > 1 else 0.0,
        "avg_degree": (2 * m / n) if n else 0.0,
        "n_components": len(comps),
        "largest_component_frac": (largest / n) if n else 0.0,
        "avg_clustering": nx.average_clustering(U) if n > 2 else 0.0,
    }
    try:
        out["longest_path"] = nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) else None
    except Exception:
        out["longest_path"] = None
    return out


def semantic_metrics(store):
    vecs = store.node_vectors()
    if len(vecs) < 2:
        return {"mean_pairwise_distance": 0.0, "embedding_spread": 0.0, "n_embedded": len(vecs)}
    X = np.stack(list(vecs.values()))
    sims = X @ X.T
    iu = np.triu_indices(len(X), k=1)
    mean_dist = float(1.0 - sims[iu].mean())          # higher => more diverse ideas
    spread = float(np.trace(np.cov(X.T)))             # variance over embedding dims
    return {"mean_pairwise_distance": mean_dist, "embedding_spread": spread, "n_embedded": len(X)}


def creativity_proxies(store, run_stats):
    """Torrance-style proxies mapped to the graph."""
    G = store.G
    n = G.number_of_nodes()
    calls = max(1, run_stats.get("calls", 1))
    # flexibility: # semantic clusters (cheap k via sqrt heuristic + kmeans if available)
    flexibility = None
    vecs = store.node_vectors()
    if len(vecs) >= 4:
        try:
            from sklearn.cluster import KMeans
            X = np.stack(list(vecs.values()))
            k = max(2, int(np.sqrt(len(X))))
            flexibility = int(len(set(KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X))))
        except Exception:
            flexibility = None
    return {
        "fluency": n,                                  # distinct ideas
        "ideas_per_call": n / calls,                   # ideation efficiency
        "elaboration": (G.number_of_edges() / n) if n else 0.0,   # edges per idea
        "flexibility_clusters": flexibility,
    }


def all_metrics(store, run_stats):
    return {**graph_metrics(store), **semantic_metrics(store),
            **creativity_proxies(store, run_stats)}


# ----------------------------------------------------------------------------
#  Advanced graph analytics (for the rich journal figure / graph_analysis.json)
# ----------------------------------------------------------------------------
def _largest_cc_undirected(G):
    U = G.to_undirected()
    if U.number_of_nodes() == 0:
        return U
    return U.subgraph(max(nx.connected_components(U), key=len)).copy()


def _avg_shortest_path(U, sample_above=1500, sources=300):
    """Mean shortest-path length on U. Exact for small graphs; on large ones, average over BFS
    from a random sample of sources (exact all-pairs is O(V*(V+E)) — minutes on ~7k nodes)."""
    import random as _random
    n = U.number_of_nodes()
    if n <= sample_above:
        return nx.average_shortest_path_length(U)
    src = _random.Random(0).sample(list(U), min(sources, n))
    print(f"    small-world: sampling avg path length from {len(src)} sources on {n} nodes…",
          flush=True)
    tot = cnt = 0
    for s in src:
        for d in nx.single_source_shortest_path_length(U, s).values():
            tot += d; cnt += 1
    return tot / max(1, cnt - len(src))                   # exclude the d=0 self terms


def small_worldness(G):
    """Analytic small-world coefficients on the largest connected component.
    sigma = (C/C_rand)/(L/L_rand) > 1 and omega ~ 0 indicate small-world structure.
    Uses Erdos-Renyi / ring-lattice analytic references (fast, deterministic)."""
    U = _largest_cc_undirected(G)
    n, m = U.number_of_nodes(), U.number_of_edges()
    if n < 4 or m < n - 1:
        return {}
    k = 2 * m / n
    C = nx.average_clustering(U)
    L = _avg_shortest_path(U)
    C_rand = k / (n - 1)
    L_rand = math.log(n) / math.log(k) if k > 1 else float("nan")
    C_latt = 3 * (k - 2) / (4 * (k - 1)) if k > 2 else float("nan")
    out = {"clustering": C, "avg_path_length": L}
    if C_rand > 0 and L_rand and L_rand > 0:
        out["small_world_sigma"] = (C / C_rand) / (L / L_rand)
    if C_latt and C_latt > 0 and L > 0:
        out["small_world_omega"] = L_rand / L - C / C_latt
    return out


def advanced_metrics(G):
    out = dict(small_worldness(G))
    U = G.to_undirected()
    for name, fn in [("transitivity", lambda: nx.transitivity(G)),
                     ("degree_assortativity", lambda: nx.degree_assortativity_coefficient(G)),
                     ("reciprocity", lambda: nx.reciprocity(G))]:
        try:
            out[name] = float(fn())
        except Exception:
            pass
    try:
        comms = list(nx.community.greedy_modularity_communities(U))
        out["modularity"] = float(nx.community.modularity(U, comms))
        out["n_communities"] = len(comms)
    except Exception:
        pass
    try:
        Ucc = _largest_cc_undirected(G)
        nU = Ucc.number_of_nodes()
        if nU <= 1:
            out["diameter"] = 0
        elif nU <= 1500:                                  # exact diameter is O(V*(V+E)) — skip on big graphs
            out["diameter"] = nx.diameter(Ucc)
    except Exception:
        pass
    out["density"] = nx.density(G) if G.number_of_nodes() > 1 else 0.0
    return out


def centralities(G, approx_above=1500, pivots=400):
    """Degree/betweenness/closeness/PageRank for the viz panels. On large graphs (> approx_above
    nodes) betweenness uses k-pivot sampling and closeness is computed on a node sample — exact
    versions are O(V*(V+E)) in pure-Python NetworkX (minutes on ~7k nodes), and these only feed
    distribution plots / the broker scatter, so the approximation changes no quantitative claim."""
    import random as _random
    n = G.number_of_nodes()
    big = n > approx_above
    deg = dict(G.degree())
    try:
        if n <= 2:
            bet = {x: 0.0 for x in G}
        elif big:
            print(f"    centralities: approx betweenness (k={pivots} pivots) on {n} nodes…", flush=True)
            bet = nx.betweenness_centrality(G, k=min(pivots, n), seed=0)
        else:
            print(f"    centralities: exact betweenness on {n} nodes…", flush=True)
            bet = nx.betweenness_centrality(G)
    except Exception:
        bet = {x: 0.0 for x in G}
    try:
        if big:                                        # closeness has no k-sampling; sample nodes
            samp = set(_random.Random(0).sample(list(G), min(pivots, n)))
            print(f"    centralities: closeness on a {len(samp)}-node sample…", flush=True)
            clo = {x: nx.closeness_centrality(G, u=x) for x in samp}
        else:
            clo = nx.closeness_centrality(G)
    except Exception:
        clo = {x: 0.0 for x in G}
    try:
        pr = nx.pagerank(G) if G.number_of_edges() else {x: 1.0 / max(1, n) for x in G}
    except Exception:
        pr = {x: 1.0 / max(1, n) for x in G}
    return {"degree": deg, "betweenness": bet, "closeness": clo, "pagerank": pr}
